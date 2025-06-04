#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_vae_classifier.py

Script para entrenar un Autoencoder Variacional (VAE) Convolucional sobre matrices de 
conectividad funcional fMRI y luego usar sus representaciones latentes para clasificar 
entre Controles Sanos (CN) y pacientes con Enfermedad de Alzheimer (AD).

Versión: 1.3.0 - Channel/Norm/Arch/Split Enhancements
Cambios desde v1.2.0:
- Añadida selección de canales de entrada específicos para el VAE (--channels_to_use).
- Implementada normalización MinMax fija para canales predefinidos si se seleccionan.
- Modificada la capa FC intermedia del VAE para que su dimensión pueda ser la mitad de la entrada.
- Añadido log para sugerir una dimensión latente basada en la capa previa.
- Mejorada la estratificación de folds del clasificador para incluir 'Sex' si está disponible.
- Definidos nombres de canales por defecto y parámetros de normalización fijos.
- Ajustes en el manejo de la normalización para canales seleccionados.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
import time
import argparse
import gc
import hashlib # Para hash de IDs
from typing import Optional, List, Dict, Tuple, Any, Union # Asegurar que Tuple está importado
import copy # For deepcopying model state

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split as sk_train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler as SklearnScaler # Para escalar features latentes
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, f1_score, average_precision_score

# --- Configuración del Logger ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

# --- Constantes y Configuraciones Globales ---
DEFAULT_CHANNEL_NAMES = [
    'Pearson_OMST_GCE_Signed_Weighted',
    'Pearson_Full_FisherZ_Signed',
    'MI_KNN_Symmetric',
    'dFC_AbsDiffMean',
    'dFC_StdDev',
    'Granger_F_lag1'
]

# Fixed normalization parameters for specific channels when using minmax_offdiag
# These will override data-driven min/max calculation for these specific channels if they are used.
FIXED_MINMAX_PARAMS_PER_CHANNEL = {
    'Pearson_OMST_GCE_Signed_Weighted': {'min': 0.0000, 'max': 6.2582},
    'Pearson_Full_FisherZ_Signed': {'min': -5.0878, 'max': 5.0575},
    'MI_KNN_Symmetric': {'min': -4.5142, 'max': 9.0464},
    'dFC_AbsDiffMean': {'min': -3.7656, 'max': 4.3263},
    'dFC_StdDev': {'min': -2.7555, 'max': 4.0289},
    'Granger_F_lag1': {'min': -0.9069, 'max': 30.7738}
}


# --- Definición del Modelo VAE Convolucional ---
class ConvolutionalVAE(nn.Module):
    def __init__(self, 
                 input_channels: int = 6, 
                 latent_dim: int = 128, 
                 image_size: int = 131, 
                 final_activation: str = 'sigmoid',
                 intermediate_fc_dim_config: Union[int, str] = 0, # Can be int or special string like "half"
                 dropout_rate: float = 0.2):
        super(ConvolutionalVAE, self).__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.final_activation_name = final_activation
        self.dropout_rate = dropout_rate

        # Encoder Convolutional Layers
        encoder_conv_layers = []
        current_ch_enc = input_channels
        # Fewer channels if input_channels is small
        base_conv_channels = [max(16, input_channels*2), max(32, input_channels*4), max(64, input_channels*8)] 
        conv_channels_enc = [min(c, 256) for c in base_conv_channels] # Cap channel numbers


        kernels = [7, 5, 3]
        paddings = [3, 2, 1] 

        for i in range(len(conv_channels_enc)):
            encoder_conv_layers.extend([
                nn.Conv2d(current_ch_enc, conv_channels_enc[i], kernel_size=kernels[i], stride=2, padding=paddings[i]),
                nn.ReLU(),
                nn.BatchNorm2d(conv_channels_enc[i]),
                nn.Dropout2d(p=self.dropout_rate)
            ])
            current_ch_enc = conv_channels_enc[i]
        self.encoder_conv = nn.Sequential(*encoder_conv_layers)
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, image_size, image_size)
            encoded_dummy = self.encoder_conv(dummy_input)
            self.final_conv_output_channels = encoded_dummy.shape[1]
            self.final_spatial_dim_encoder = encoded_dummy.shape[-1]
            self.flattened_size_after_conv = self.final_conv_output_channels * self.final_spatial_dim_encoder * self.final_spatial_dim_encoder
            logger.info(f"VAE Encoder: Input Channels: {input_channels}, Conv Channels: {conv_channels_enc}")
            logger.info(f"VAE Encoder: Calculated flattened size after conv = {self.flattened_size_after_conv} "
                        f"(final conv output channels: {self.final_conv_output_channels}, "
                        f"final spatial dim after encoder: {self.final_spatial_dim_encoder})")

        # Determine intermediate FC dimension
        if isinstance(intermediate_fc_dim_config, str) and intermediate_fc_dim_config.lower() == "half":
            self.intermediate_fc_dim = self.flattened_size_after_conv // 2
            logger.info(f"VAE: Intermediate FC dim set to 'half' of flattened conv output: {self.intermediate_fc_dim}")
        elif isinstance(intermediate_fc_dim_config, int) and intermediate_fc_dim_config > 0:
            self.intermediate_fc_dim = intermediate_fc_dim_config
        else: # 0 or invalid
            self.intermediate_fc_dim = 0
            
        # Encoder FC Layers
        if self.intermediate_fc_dim > 0:
            self.encoder_fc_intermediate = nn.Sequential(
                nn.Linear(self.flattened_size_after_conv, self.intermediate_fc_dim),
                nn.ReLU(),
                nn.BatchNorm1d(self.intermediate_fc_dim),
                nn.Dropout(p=self.dropout_rate)
            )
            fc_mu_logvar_input_dim = self.intermediate_fc_dim
        else:
            self.encoder_fc_intermediate = None
            fc_mu_logvar_input_dim = self.flattened_size_after_conv
        
        logger.info(f"VAE: Input dimension to fc_mu/fc_logvar: {fc_mu_logvar_input_dim}.")
        suggested_latent_dim = fc_mu_logvar_input_dim // 2
        logger.info(f"VAE: Based on this, a suggested latent_dim could be: {suggested_latent_dim} (current is {latent_dim}).")
            
        self.fc_mu = nn.Linear(fc_mu_logvar_input_dim, latent_dim)
        self.fc_logvar = nn.Linear(fc_mu_logvar_input_dim, latent_dim)

        # Decoder FC Layers
        decoder_fc_to_conv_input_dim = latent_dim
        if self.intermediate_fc_dim > 0: # Symmetric intermediate FC layer in decoder
            self.decoder_fc_intermediate = nn.Sequential(
                nn.Linear(latent_dim, self.intermediate_fc_dim),
                nn.ReLU(),
                nn.BatchNorm1d(self.intermediate_fc_dim),
                nn.Dropout(p=self.dropout_rate)
            )
            decoder_fc_to_conv_input_dim = self.intermediate_fc_dim # This will be input to the next Linear
        else:
            self.decoder_fc_intermediate = None
            # decoder_fc_to_conv_input_dim remains latent_dim
            
        self.decoder_fc_to_conv = nn.Linear(decoder_fc_to_conv_input_dim, self.flattened_size_after_conv)

        # Decoder Convolutional Transpose Layers
        decoder_conv_t_layers = []
        current_ch_dec = self.final_conv_output_channels 
        target_conv_t_channels = conv_channels_enc[-2::-1] + [input_channels] 
        
        # Output paddings calculated for image_size=131.
        # If image_size changes, these might need dynamic calculation or careful review.
        # The forward pass crop/pad/interpolate is a fallback.
        if image_size == 131:
            output_paddings = [0, 1, 0] 
        else:
            # For other image sizes, output_padding might need adjustment.
            # Defaulting to 0 or 1 if exact calculation is not performed.
            # This might lead to more reliance on the interpolate step.
            logger.warning(f"Image size is {image_size}, not 131. Hardcoded output_paddings {output_paddings} might not be optimal. "
                           "Relying on forward pass adjustment.")
            # A simple heuristic, may not be perfect:
            output_paddings = []
            temp_dim = self.final_spatial_dim_encoder
            target_dims = [self.final_spatial_dim_encoder * (2**i) for i in range(1, len(kernels)+1)] # Approximate target sizes
            target_dims[-1] = image_size # Final target is original image size
            
            # This is a placeholder for dynamic output_padding calculation if needed.
            # For now, using the [0,1,0] which was specific to 131->66->33->17 path.
            # A more robust way would be to calculate H_out for each ConvTranspose2d and adjust OP.
            # H_out = (H_in - 1)*S - 2*P + K + OP
            # OP = Target_H_out - [(H_in - 1)*S - 2*P + K]
            # For simplicity, we'll stick to the ones for 131 and let interpolate handle others.
            output_paddings = [0, 1, 0] # Keep as per original calculation for 131


        for i in range(len(target_conv_t_channels)):
            decoder_conv_t_layers.extend([
                nn.ConvTranspose2d(current_ch_dec, target_conv_t_channels[i], 
                                   kernel_size=kernels[len(kernels)-1-i], 
                                   stride=2, 
                                   padding=paddings[len(paddings)-1-i], 
                                   output_padding=output_paddings[i]),
                nn.ReLU() if i < len(target_conv_t_channels) - 1 else nn.Identity(),
                nn.BatchNorm2d(target_conv_t_channels[i]) if i < len(target_conv_t_channels) - 1 else nn.Identity(),
            ])
            current_ch_dec = target_conv_t_channels[i]
        
        if self.final_activation_name == 'sigmoid':
            decoder_conv_t_layers.append(nn.Sigmoid())
            logger.info("Decoder final activation: Sigmoid")
        elif self.final_activation_name == 'tanh':
            decoder_conv_t_layers.append(nn.Tanh())
            logger.info("Decoder final activation: Tanh")
        else: 
            logger.info("Decoder final activation: Linear (None)")
            
        self.decoder_conv = nn.Sequential(*decoder_conv_t_layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_conv(x)
        h = h.view(h.size(0), -1) 
        if self.encoder_fc_intermediate:
            h = self.encoder_fc_intermediate(h)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h_latent = z
        if self.decoder_fc_intermediate:
            h_latent = self.decoder_fc_intermediate(z)
        
        h = self.decoder_fc_to_conv(h_latent)
        h = h.view(h.size(0), self.final_conv_output_channels, self.final_spatial_dim_encoder, self.final_spatial_dim_encoder)
        return self.decoder_conv(h)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x_raw = self.decode(z)

        target_h, target_w = x.shape[2], x.shape[3]
        current_h, current_w = recon_x_raw.shape[2], recon_x_raw.shape[3]

        if current_h == target_h and current_w == target_w:
            recon_x = recon_x_raw
        else: 
            logger.debug(f"Shape mismatch before adjustment: Input {x.shape}, Recon_raw {recon_x_raw.shape}. Adjusting...")
            if current_h > target_h or current_w > target_w:
                h_start = max(0, (current_h - target_h) // 2)
                w_start = max(0, (current_w - target_w) // 2)
                recon_x = recon_x_raw[:, :, h_start:h_start + target_h, w_start:w_start + target_w]
            elif current_h < target_h or current_w < target_w:
                padding_h_total = target_h - current_h
                padding_w_total = target_w - current_w
                pad_top = padding_h_total // 2
                pad_bottom = padding_h_total - pad_top
                pad_left = padding_w_total // 2
                pad_right = padding_w_total - pad_left
                recon_x = nn.functional.pad(recon_x_raw, (pad_left, pad_right, pad_top, pad_bottom))
            else: 
                 recon_x = recon_x_raw
            
            if recon_x.shape[2] != target_h or recon_x.shape[3] != target_w :
                 logger.warning(f"Shape mismatch even after crop/pad: Input {x.shape}, Recon {recon_x.shape}. Attempting interpolate.")
                 recon_x = nn.functional.interpolate(recon_x_raw, size=(target_h, target_w), mode='bilinear', align_corners=False)
        
        return recon_x, mu, logvar

def vae_loss_function(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    recon_loss_mse = nn.functional.mse_loss(recon_x, x, reduction='sum') / x.shape[0] 
    kld_element = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kld = torch.mean(kld_element)
    return recon_loss_mse + beta * kld

# --- Funciones Auxiliares ---
def load_data(tensor_path: Path, metadata_path: Path) -> Tuple[Optional[np.ndarray], Optional[pd.DataFrame]]:
    logger.info(f"Cargando tensor global desde: {tensor_path}")
    if not tensor_path.exists():
        logger.error(f"Archivo de tensor global NO encontrado: {tensor_path}")
        return None, None
    try:
        data_npz = np.load(tensor_path)
        global_tensor = data_npz['global_tensor_data']
        subject_ids_tensor = data_npz['subject_ids'].astype(str) # Ensure string type
        logger.info(f"Tensor global cargado. Forma: {global_tensor.shape}")
        if global_tensor.shape[1] != len(DEFAULT_CHANNEL_NAMES):
            logger.warning(f"El tensor global tiene {global_tensor.shape[1]} canales, pero hay {len(DEFAULT_CHANNEL_NAMES)} nombres de canal por defecto definidos.")
    except Exception as e:
        logger.error(f"Error cargando tensor global: {e}")
        return None, None

    logger.info(f"Cargando metadatos desde: {metadata_path}")
    if not metadata_path.exists():
        logger.error(f"Archivo de metadatos NO encontrado: {metadata_path}")
        return None, None
    try:
        metadata_df = pd.read_csv(metadata_path)
        metadata_df['SubjectID'] = metadata_df['SubjectID'].astype(str).str.strip()
        logger.info(f"Metadatos cargados. Forma: {metadata_df.shape}")
    except Exception as e:
        logger.error(f"Error cargando metadatos: {e}")
        return None, None

    tensor_df = pd.DataFrame({'SubjectID': subject_ids_tensor})
    tensor_df['tensor_idx'] = np.arange(len(subject_ids_tensor))
    
    merged_df = pd.merge(metadata_df, tensor_df, on='SubjectID', how='inner')
    
    if len(merged_df) != len(subject_ids_tensor):
        logger.warning(f"Algunos sujetos del tensor no se encontraron en metadatos o viceversa. "
                       f"Tensor: {len(subject_ids_tensor)}, Merged: {len(merged_df)}")
    
    final_df = merged_df.copy()
    return global_tensor, final_df

def normalize_inter_channel_fold(
    data_tensor: np.ndarray, 
    train_indices_in_fold: np.ndarray, 
    mode: str = 'zscore_offdiag',
    selected_channel_original_names: Optional[List[str]] = None # Names of the channels in data_tensor
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    
    num_selected_channels = data_tensor.shape[1]
    logger.info(f"Aplicando normalización inter-canal (modo: {mode}) sobre {num_selected_channels} canales seleccionados, usando parámetros de entrenamiento.")
    
    normalized_tensor_fold = data_tensor.copy()
    norm_params_per_channel_list = [] # For the selected channels
    num_rois = data_tensor.shape[2]

    for c_idx_selected in range(num_selected_channels): # Iterate over the selected channels present in data_tensor
        off_diag_mask = ~np.eye(num_rois, dtype=bool)
        current_channel_original_name = selected_channel_original_names[c_idx_selected] if selected_channel_original_names else f"Channel_{c_idx_selected}"
        
        params = {'mode': mode, 'original_name': current_channel_original_name}

        # Check for fixed normalization parameters
        use_fixed_params = False
        if mode == 'minmax_offdiag' and current_channel_original_name in FIXED_MINMAX_PARAMS_PER_CHANNEL:
            fixed_p = FIXED_MINMAX_PARAMS_PER_CHANNEL[current_channel_original_name]
            params.update({'min': fixed_p['min'], 'max': fixed_p['max']})
            use_fixed_params = True
            logger.info(f"Canal '{current_channel_original_name}' (idx {c_idx_selected} en selección): Usando MinMax fijo (min={params['min']:.4f}, max={params['max']:.4f}).")

        if not use_fixed_params:
            # Calculate params from data_tensor[train_indices_in_fold] for this selected channel
            channel_data_train_for_norm_params = data_tensor[train_indices_in_fold, c_idx_selected, :, :]
            
            all_off_diag_train_values = []
            if channel_data_train_for_norm_params.ndim == 3: # multiple subjects
                for subj_idx in range(channel_data_train_for_norm_params.shape[0]):
                    all_off_diag_train_values.extend(channel_data_train_for_norm_params[subj_idx][off_diag_mask])
            elif channel_data_train_for_norm_params.ndim == 2: # single subject (should not happen for train_indices_in_fold)
                 all_off_diag_train_values.extend(channel_data_train_for_norm_params[off_diag_mask])

            all_off_diag_train_values = np.array(all_off_diag_train_values)

            if all_off_diag_train_values.size == 0:
                logger.warning(f"Canal '{current_channel_original_name}': No hay elementos fuera de la diagonal en el training set. No se escala.")
                params.update({'mean': 0.0, 'std': 1.0, 'min': 0.0, 'max': 1.0, 'no_scale': True})
            elif mode == 'zscore_offdiag':
                mean_val = np.mean(all_off_diag_train_values)
                std_val = np.std(all_off_diag_train_values)
                params.update({'mean': mean_val, 'std': std_val if std_val > 1e-9 else 1.0})
                if std_val <= 1e-9: logger.warning(f"Canal '{current_channel_original_name}': STD muy bajo ({std_val:.2e}). Usando STD=1.")
            elif mode == 'minmax_offdiag': # This path only if not in FIXED_MINMAX_PARAMS_PER_CHANNEL
                min_val = np.min(all_off_diag_train_values)
                max_val = np.max(all_off_diag_train_values)
                params.update({'min': min_val, 'max': max_val})
                if (max_val - min_val) <= 1e-9: logger.warning(f"Canal '{current_channel_original_name}': Rango (max-min) muy bajo ({(max_val - min_val):.2e}).")
            else:
                raise ValueError(f"Modo de normalización desconocido: {mode}")
        
        norm_params_per_channel_list.append(params)

        # Apply normalization to the current selected channel (c_idx_selected) across all subjects in data_tensor
        if not params.get('no_scale', False):
            for subj_glob_idx in range(data_tensor.shape[0]):
                current_matrix = data_tensor[subj_glob_idx, c_idx_selected, :, :]
                scaled_matrix_ch = current_matrix.copy()
                if off_diag_mask.any(): # Ensure there are off-diagonal elements
                    if mode == 'zscore_offdiag':
                        if params['std'] > 1e-9:
                            scaled_matrix_ch[off_diag_mask] = (current_matrix[off_diag_mask] - params['mean']) / params['std']
                    elif mode == 'minmax_offdiag':
                        range_val = params.get('max', 1.0) - params.get('min', 0.0)
                        if range_val > 1e-9: 
                            scaled_matrix_ch[off_diag_mask] = (current_matrix[off_diag_mask] - params['min']) / range_val
                        else: 
                            scaled_matrix_ch[off_diag_mask] = 0.0 
                normalized_tensor_fold[subj_glob_idx, c_idx_selected, :, :] = scaled_matrix_ch
            
            if not use_fixed_params: # Log only if params were calculated
                if mode == 'zscore_offdiag':
                    logger.info(f"Canal '{current_channel_original_name}': Off-diag Z-score (train_mean={params['mean']:.3f}, train_std={params['std']:.3f}).")
                elif mode == 'minmax_offdiag':
                    logger.info(f"Canal '{current_channel_original_name}': Off-diag MinMax (train_min={params['min']:.3f}, train_max={params['max']:.3f}).")
            
    return normalized_tensor_fold, norm_params_per_channel_list


def apply_normalization_params(data_tensor_subset: np.ndarray, 
                               norm_params_per_channel_list: List[Dict[str, float]]
                               ) -> np.ndarray:
    logger.info(f"Aplicando parámetros de normalización precalculados a subconjunto de datos ({data_tensor_subset.shape[1]} canales).")
    normalized_tensor_subset = data_tensor_subset.copy()
    num_selected_channels = data_tensor_subset.shape[1]
    num_rois = data_tensor_subset.shape[2]
    off_diag_mask = ~np.eye(num_rois, dtype=bool)

    if len(norm_params_per_channel_list) != num_selected_channels:
        raise ValueError(f"Mismatch in number of channels for normalization: data has {num_selected_channels}, params provided for {len(norm_params_per_channel_list)}")

    for c_idx_selected in range(num_selected_channels):
        params = norm_params_per_channel_list[c_idx_selected]
        mode = params.get('mode', 'zscore_offdiag') 
        channel_name = params.get('original_name', f"Channel_{c_idx_selected}")
        
        if params.get('no_scale', False):
            logger.info(f"Canal '{channel_name}': No se aplica escala (parámetros indicaron no escalar).")
            continue

        for subj_idx in range(data_tensor_subset.shape[0]):
            current_matrix = data_tensor_subset[subj_idx, c_idx_selected, :, :]
            scaled_matrix_ch = current_matrix.copy()
            if off_diag_mask.any():
                if mode == 'zscore_offdiag':
                    if params['std'] > 1e-9:
                        scaled_matrix_ch[off_diag_mask] = (current_matrix[off_diag_mask] - params['mean']) / params['std']
                elif mode == 'minmax_offdiag':
                    range_val = params.get('max', 1.0) - params.get('min', 0.0)
                    if range_val > 1e-9:
                        scaled_matrix_ch[off_diag_mask] = (current_matrix[off_diag_mask] - params['min']) / range_val
                    else:
                        scaled_matrix_ch[off_diag_mask] = 0.0 
            normalized_tensor_subset[subj_idx, c_idx_selected, :, :] = scaled_matrix_ch
    return normalized_tensor_subset

# --- Función Principal de Entrenamiento y Evaluación ---
def train_and_evaluate_pipeline(global_tensor_all_channels: np.ndarray, metadata_df: pd.DataFrame, args: argparse.Namespace):
    
    # --- Channel Selection ---
    selected_channel_indices: List[int] = []
    selected_channel_names: List[str] = []

    if args.channels_to_use:
        for ch_specifier in args.channels_to_use:
            try:
                ch_idx = int(ch_specifier)
                if 0 <= ch_idx < len(DEFAULT_CHANNEL_NAMES):
                    selected_channel_indices.append(ch_idx)
                    selected_channel_names.append(DEFAULT_CHANNEL_NAMES[ch_idx])
                else:
                    logger.warning(f"Índice de canal '{ch_idx}' fuera de rango. Ignorando.")
            except ValueError: # It's a name
                if ch_specifier in DEFAULT_CHANNEL_NAMES:
                    selected_channel_indices.append(DEFAULT_CHANNEL_NAMES.index(ch_specifier))
                    selected_channel_names.append(ch_specifier)
                else:
                    logger.warning(f"Nombre de canal '{ch_specifier}' desconocido. Ignorando. Canales disponibles: {DEFAULT_CHANNEL_NAMES}")
        
        if not selected_channel_indices:
            logger.error("No se seleccionaron canales válidos. Abortando.")
            return
        logger.info(f"Usando canales seleccionados (índices originales): {selected_channel_indices}")
        logger.info(f"Nombres de canales seleccionados: {selected_channel_names}")
        current_global_tensor = global_tensor_all_channels[:, selected_channel_indices, :, :]
    else:
        logger.info(f"Usando todos los {global_tensor_all_channels.shape[1]} canales disponibles.")
        current_global_tensor = global_tensor_all_channels
        selected_channel_names = list(DEFAULT_CHANNEL_NAMES) # Assume all default channels if tensor matches
        if global_tensor_all_channels.shape[1] != len(DEFAULT_CHANNEL_NAMES):
             selected_channel_names = [f"OrigChan{i}" for i in range(global_tensor_all_channels.shape[1])]


    num_input_channels_for_vae = current_global_tensor.shape[1]

    # 1. Filtrar para clasificación CN vs AD
    cn_ad_df = metadata_df[metadata_df['ResearchGroup'].isin(['CN', 'AD'])].copy()
    if cn_ad_df.empty:
        logger.error("No se encontraron sujetos CN o AD en los metadatos.")
        return

    label_mapping = {'CN': 0, 'AD': 1}
    cn_ad_df['label'] = cn_ad_df['ResearchGroup'].map(label_mapping)
    
    # Stratification key for classifier's outer CV
    # Try to stratify by ResearchGroup and Sex if 'Sex' column is available
    if 'Sex' in cn_ad_df.columns:
        # Handle potential NaNs in Sex column by converting to string 'Unknown'
        cn_ad_df['Sex_str'] = cn_ad_df['Sex'].fillna('Sex_Unknown').astype(str)
        cn_ad_df['stratify_key'] = cn_ad_df['ResearchGroup'].astype(str) + "_" + cn_ad_df['Sex_str']
        logger.info("Estratificando folds del clasificador por ResearchGroup y Sex.")
        stratify_labels = cn_ad_df['stratify_key']
    else:
        logger.info("Columna 'Sex' no encontrada en metadatos. Estratificando folds del clasificador solo por ResearchGroup.")
        stratify_labels = cn_ad_df['label']


    X_classifier_indices_global = cn_ad_df['tensor_idx'].values 
    y_classifier_labels_cn_ad = cn_ad_df['label'].values # Labels (0 or 1) for CN/AD subjects

    logger.info(f"Iniciando clasificación CN vs AD. Total sujetos CN/AD: {len(cn_ad_df)}. "
                f"CN: {sum(y_classifier_labels_cn_ad == 0)}, AD: {sum(y_classifier_labels_cn_ad == 1)}")

    # 2. Configuración de Nested Cross-Validation
    outer_cv_clf = StratifiedKFold(n_splits=args.outer_folds, shuffle=True, random_state=args.seed)
    
    fold_metrics = []
    
    for fold_idx, (train_dev_clf_local_indices, test_clf_local_indices) in enumerate(outer_cv_clf.split(np.zeros(len(stratify_labels)), stratify_labels)):
        logger.info(f"--- Iniciando Fold Externo {fold_idx + 1}/{args.outer_folds} ---")

        # Global tensor indices for classifier train/dev and test sets (these are indices in the original full metadata_df)
        global_indices_clf_train_dev = X_classifier_indices_global[train_dev_clf_local_indices]
        global_indices_clf_test = X_classifier_indices_global[test_clf_local_indices]
        
        # VAE training data: All subjects NOT in the current classifier *test* set.
        all_subject_global_indices_in_metadata = metadata_df['tensor_idx'].values # Indices from original metadata
        global_indices_vae_train_val = np.setdiff1d(all_subject_global_indices_in_metadata, global_indices_clf_test, assume_unique=True)
        
        vae_train_val_tensor_original_scale = current_global_tensor[global_indices_vae_train_val] # Use already channel-selected tensor
        
        if args.vae_val_split_ratio > 0 and len(global_indices_vae_train_val) > 10 :
             # Stratification for VAE val split is complex if non-CN/AD subjects are included. Keeping it random.
             # The VAE is unsupervised; its validation is for reconstruction quality.
             vae_train_indices_local_to_vae_set, vae_val_indices_local_to_vae_set = sk_train_test_split(
                np.arange(len(global_indices_vae_train_val)), # Indices local to vae_train_val_tensor_original_scale
                test_size=args.vae_val_split_ratio,
                random_state=args.seed + fold_idx 
            )
        else: 
            vae_train_indices_local_to_vae_set = np.arange(len(global_indices_vae_train_val))
            vae_val_indices_local_to_vae_set = np.array([], dtype=int)
            logger.info("No VAE validation split performed.")

        logger.info("Normalizando datos para VAE...")
        vae_train_val_tensor_norm, norm_params_fold_list = normalize_inter_channel_fold(
            vae_train_val_tensor_original_scale, 
            vae_train_indices_local_to_vae_set, 
            mode=args.norm_mode,
            selected_channel_original_names=selected_channel_names # Pass names of channels in current_global_tensor
        )

        vae_train_dataset_fold = TensorDataset(torch.from_numpy(vae_train_val_tensor_norm[vae_train_indices_local_to_vae_set]).float())
        vae_train_dataloader = DataLoader(vae_train_dataset_fold, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        
        vae_val_dataloader = None
        if len(vae_val_indices_local_to_vae_set) > 0:
            vae_val_dataset_fold = TensorDataset(torch.from_numpy(vae_train_val_tensor_norm[vae_val_indices_local_to_vae_set]).float())
            vae_val_dataloader = DataLoader(vae_val_dataset_fold, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Usando dispositivo: {device}")
        
        vae = ConvolutionalVAE(
            input_channels=num_input_channels_for_vae, 
            latent_dim=args.latent_dim, 
            image_size=current_global_tensor.shape[2], # ROI count
            final_activation=args.vae_final_activation,
            intermediate_fc_dim_config=args.intermediate_fc_dim_vae, # Pass the arg value
            dropout_rate=args.dropout_rate_vae
            ).to(device)
        optimizer_vae = optim.Adam(vae.parameters(), lr=args.lr_vae, weight_decay=args.weight_decay_vae)
        
        scheduler_vae = None
        if vae_val_dataloader and args.lr_scheduler_patience_vae > 0 :
            scheduler_vae = optim.lr_scheduler.ReduceLROnPlateau(optimizer_vae, 'min', 
                                                                 patience=args.lr_scheduler_patience_vae, 
                                                                 factor=0.1, verbose=True)
            logger.info(f"LR Scheduler ReduceLROnPlateau activado con paciencia {args.lr_scheduler_patience_vae}.")

        logger.info(f"Entrenando VAE para el fold {fold_idx + 1}...")
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_state_dict = None

        for epoch in range(args.epochs_vae):
            vae.train()
            epoch_train_loss = 0
            for batch_idx, (data,) in enumerate(vae_train_dataloader):
                data = data.to(device)
                optimizer_vae.zero_grad()
                recon_batch, mu, logvar = vae(data)
                loss = vae_loss_function(recon_batch, data, mu, logvar, beta=args.beta_vae)
                loss.backward()
                optimizer_vae.step()
                epoch_train_loss += loss.item() * data.size(0)
            
            avg_epoch_train_loss = epoch_train_loss / len(vae_train_dataloader.dataset)
            current_lr = optimizer_vae.param_groups[0]['lr']
            log_msg = f"Fold {fold_idx+1}, VAE Epoch {epoch+1}/{args.epochs_vae}, Train Loss: {avg_epoch_train_loss:.4f}, LR: {current_lr:.2e}"

            if vae_val_dataloader:
                vae.eval()
                epoch_val_loss = 0
                with torch.no_grad():
                    for val_data, in vae_val_dataloader:
                        val_data = val_data.to(device)
                        recon_val, mu_val, logvar_val = vae(val_data)
                        val_loss_batch = vae_loss_function(recon_val, val_data, mu_val, logvar_val, beta=args.beta_vae)
                        epoch_val_loss += val_loss_batch.item() * val_data.size(0)
                avg_epoch_val_loss = epoch_val_loss / len(vae_val_dataloader.dataset)
                log_msg += f", Val Loss: {avg_epoch_val_loss:.4f}"

                if scheduler_vae:
                    scheduler_vae.step(avg_epoch_val_loss)

                if avg_epoch_val_loss < best_val_loss:
                    best_val_loss = avg_epoch_val_loss
                    epochs_no_improve = 0
                    best_model_state_dict = copy.deepcopy(vae.state_dict())
                else:
                    epochs_no_improve += 1
                
                if args.early_stopping_patience_vae > 0 and epochs_no_improve >= args.early_stopping_patience_vae:
                    logger.info(f"Fold {fold_idx+1}: Early stopping VAE en epoch {epoch+1}. Mejor val_loss: {best_val_loss:.4f}")
                    break
            
            if (epoch + 1) % 10 == 0 or epoch == args.epochs_vae -1 : 
                logger.info(log_msg)
        
        if best_model_state_dict:
            logger.info(f"Fold {fold_idx+1}: Cargando mejor modelo VAE con val_loss: {best_val_loss:.4f}")
            vae.load_state_dict(best_model_state_dict)
        else:
            logger.info(f"Fold {fold_idx+1}: Usando el último modelo VAE entrenado.")

        vae_model_path = Path(args.output_dir) / f"vae_fold_{fold_idx+1}_ld{args.latent_dim}_beta{args.beta_vae}_ch{num_input_channels_for_vae}.pt"
        torch.save(vae.state_dict(), vae_model_path)
        logger.info(f"Modelo VAE del fold {fold_idx+1} guardado en: {vae_model_path}")
        
        # --- Classifier Stage ---
        clf_train_dev_tensor_original_scale = current_global_tensor[global_indices_clf_train_dev]
        clf_train_dev_tensor_norm = apply_normalization_params(clf_train_dev_tensor_original_scale, norm_params_fold_list)
        y_train_clf_labels = y_classifier_labels_cn_ad[train_dev_clf_local_indices]

        clf_test_tensor_original_scale = current_global_tensor[global_indices_clf_test]
        clf_test_tensor_norm = apply_normalization_params(clf_test_tensor_original_scale, norm_params_fold_list)
        y_test_clf_labels = y_classifier_labels_cn_ad[test_clf_local_indices]
        
        vae.eval()
        with torch.no_grad():
            mu_train_clf, _ = vae.encode(torch.from_numpy(clf_train_dev_tensor_norm).float().to(device))
            mu_test_clf, _ = vae.encode(torch.from_numpy(clf_test_tensor_norm).float().to(device))
        
        X_train_latent = mu_train_clf.cpu().numpy()
        X_test_latent = mu_test_clf.cpu().numpy()

        if X_train_latent.shape[0] == 0 or X_test_latent.shape[0] == 0:
            logger.warning(f"Fold {fold_idx+1}: No data for classifier. Skipping.")
            # ... (append NaN metrics)
            continue

        scaler_latent = SklearnScaler()
        X_train_latent_scaled = scaler_latent.fit_transform(X_train_latent)
        X_test_latent_scaled = scaler_latent.transform(X_test_latent)

        param_grid_svm = {'C': [0.01, 0.1, 1, 10, 100], 'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 'scale', 'auto'], 'kernel': ['rbf']}
        inner_cv_svm = StratifiedKFold(n_splits=args.inner_folds, shuffle=True, random_state=args.seed + fold_idx + 100) 
        
        # Use y_train_clf_labels for SVM training stratification
        svm_clf = GridSearchCV(SVC(probability=True, random_state=args.seed, class_weight='balanced' if args.class_weight_svm else None), 
                               param_grid_svm, cv=inner_cv_svm, scoring='roc_auc', n_jobs=1, verbose=0)
        
        logger.info(f"Entrenando clasificador SVM para el fold {fold_idx + 1}...")
        svm_clf.fit(X_train_latent_scaled, y_train_clf_labels) # Use CN/AD labels
        logger.info(f"Mejores hiperparámetros SVM para fold {fold_idx + 1}: {svm_clf.best_params_}")

        y_pred_proba_clf = svm_clf.predict_proba(X_test_latent_scaled)[:, 1]
        y_pred_clf = svm_clf.predict(X_test_latent_scaled)

        auc = roc_auc_score(y_test_clf_labels, y_pred_proba_clf)
        pr_auc = average_precision_score(y_test_clf_labels, y_pred_proba_clf)
        acc = accuracy_score(y_test_clf_labels, y_pred_clf)
        sens = recall_score(y_test_clf_labels, y_pred_clf, pos_label=1, zero_division=0) 
        spec = recall_score(y_test_clf_labels, y_pred_clf, pos_label=0, zero_division=0) 
        f1 = f1_score(y_test_clf_labels, y_pred_clf, pos_label=1, zero_division=0)
        
        logger.info(f"Fold {fold_idx + 1} - Resultados Clasificador:")
        logger.info(f"  AUC: {auc:.4f}, PR-AUC: {pr_auc:.4f}, Acc: {acc:.4f}")
        logger.info(f"  Sens (AD): {sens:.4f}, Spec (CN): {spec:.4f}, F1 (AD): {f1:.4f}")
        
        # Get SubjectIDs for logging
        train_subject_ids_fold = cn_ad_df.iloc[train_dev_clf_local_indices]['SubjectID'].tolist()
        test_subject_ids_fold = cn_ad_df.iloc[test_clf_local_indices]['SubjectID'].tolist()
        train_ids_str = ",".join(sorted(train_subject_ids_fold))
        test_ids_str = ",".join(sorted(test_subject_ids_fold))
        logger.info(f"  Fold {fold_idx+1} Train (Clf) IDs Hash: {hashlib.md5(train_ids_str.encode()).hexdigest()}")
        logger.info(f"  Fold {fold_idx+1} Test (Clf) IDs Hash: {hashlib.md5(test_ids_str.encode()).hexdigest()}")

        fold_metrics.append({'fold': fold_idx + 1, 'auc': auc, 'pr_auc': pr_auc, 'accuracy': acc, 
                             'sensitivity': sens, 'specificity': spec, 'f1_score': f1, 
                             'best_svm_params': svm_clf.best_params_,
                             'num_selected_channels': num_input_channels_for_vae,
                             'selected_channel_names': ";".join(selected_channel_names)})
        
        del vae, optimizer_vae, vae_train_dataloader, vae_val_dataloader
        del svm_clf, X_train_latent, X_test_latent, X_train_latent_scaled, X_test_latent_scaled
        gc.collect()
        if device.type == 'cuda': torch.cuda.empty_cache()

    if fold_metrics:
        metrics_df = pd.DataFrame(fold_metrics)
        logger.info("\n--- Resumen de Rendimiento (Promedio sobre Folds Externos) ---")
        for metric in ['auc', 'pr_auc', 'accuracy', 'sensitivity', 'specificity', 'f1_score']:
            mean_val = metrics_df[metric].mean()
            std_val = metrics_df[metric].std()
            logger.info(f"{metric.capitalize():<12}: {mean_val:.4f} +/- {std_val:.4f}")
        
        # Build filename
        fname_parts = ["clf_results_vae_svm", args.norm_mode, f"ld{args.latent_dim}", f"beta{args.beta_vae}"]
        if args.channels_to_use:
            fname_parts.append(f"ch{num_input_channels_for_vae}sel")
        else:
            fname_parts.append(f"ch{num_input_channels_for_vae}all")
        fname_parts.extend([f"intFC{args.intermediate_fc_dim_vae}", f"drop{args.dropout_rate_vae}", f"es{args.early_stopping_patience_vae}"])
        
        results_filename = "_".join(str(p) for p in fname_parts) + ".csv"
        results_path = Path(args.output_dir) / results_filename
        results_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(results_path, index=False)
        logger.info(f"Resultados de clasificación guardados en: {results_path}")
    else:
        logger.warning("No se pudieron calcular métricas para ningún fold.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de Entrenamiento VAE y Clasificador para AD vs CN")
    parser.add_argument("--global_tensor_path", type=str, required=True, help="Ruta al archivo .npz del tensor global.")
    parser.add_argument("--metadata_path", type=str, required=True, help="Ruta al archivo CSV de metadatos.")
    parser.add_argument("--output_dir", type=str, default="./vae_clf_output_v3", help="Directorio para guardar resultados.")
    
    parser.add_argument("--channels_to_use", type=str, nargs='*', default=None, 
                        help="Lista de nombres o índices de canales a usar (desde 0). Ej: 0 5 Pearson_Full_FisherZ_Signed. Si no se provee, usa todos.")

    parser.add_argument("--outer_folds", type=int, default=5)
    parser.add_argument("--inner_folds", type=int, default=3)
    
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--lr_vae", type=float, default=1e-4)
    parser.add_argument("--epochs_vae", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--beta_vae", type=float, default=1.0)
    parser.add_argument("--weight_decay_vae", type=float, default=1e-5)
    parser.add_argument("--vae_final_activation", type=str, default="sigmoid", choices=["sigmoid", "tanh", "linear"])
    parser.add_argument("--intermediate_fc_dim_vae", type=str, default="0", 
                        help="Dimensión de la capa FC intermedia en VAE. Entero > 0, '0' (deshabilitado), o 'half' (mitad de la entrada convolucional aplanada).")
    parser.add_argument("--dropout_rate_vae", type=float, default=0.2)

    parser.add_argument("--vae_val_split_ratio", type=float, default=0.15)
    parser.add_argument("--early_stopping_patience_vae", type=int, default=15)
    parser.add_argument("--lr_scheduler_patience_vae", type=int, default=10)

    parser.add_argument("--norm_mode", type=str, default="minmax_offdiag", choices=["zscore_offdiag", "minmax_offdiag"])
    parser.add_argument("--class_weight_svm", action='store_true')
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()

    # Convert intermediate_fc_dim_vae to int if it's a number, else keep as string (for "half")
    try:
        args.intermediate_fc_dim_vae = int(args.intermediate_fc_dim_vae)
    except ValueError:
        if args.intermediate_fc_dim_vae.lower() != "half":
            logger.error(f"Valor inválido para intermediate_fc_dim_vae: {args.intermediate_fc_dim_vae}. Usar entero, '0', o 'half'.")
            exit(1)
        # Keep as "half" string if that's the case

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    global_tensor_data, metadata_df_full = load_data(Path(args.global_tensor_path), Path(args.metadata_path))

    if global_tensor_data is not None and metadata_df_full is not None:
        logger.info(f"Canales a usar: {'Todos' if not args.channels_to_use else args.channels_to_use}")
        logger.info(f"Modo de normalización: {args.norm_mode}")
        logger.info(f"Activación final VAE: {args.vae_final_activation}")
        # ... (warnings for norm_mode and vae_final_activation consistency) ...
        
        if args.vae_val_split_ratio <=0 or args.vae_val_split_ratio >=1:
            logger.warning(f"vae_val_split_ratio ({args.vae_val_split_ratio}) es inválido. Deshabilitando validación VAE.")
            args.early_stopping_patience_vae = 0 
            args.lr_scheduler_patience_vae = 0

        train_and_evaluate_pipeline(global_tensor_data, metadata_df_full, args)
    else:
        logger.critical("No se pudieron cargar los datos. Abortando.")

    logger.info("Pipeline completado.")
    logger.info("--- Consideraciones sobre Normalización y Activación Final del VAE ---")
    logger.info("Normalización:")
    logger.info(" - 'minmax_offdiag': Escala los valores fuera de la diagonal de cada canal a [0,1] (usando params de train o fijos).")
    logger.info("   Ideal con activación final 'sigmoid' en el VAE.")
    logger.info(" - 'zscore_offdiag': Estandariza los valores fuera de la diagonal (media 0, std 1, usando params de train).")
    logger.info("   Puede ser mejor con activación final 'tanh' (para rango ~[-1,1]) o 'linear' (sin restricción de rango).")
    logger.info("Activación Final del Decoder VAE:")
    logger.info(" - 'sigmoid': Comprime la salida a [0,1]. Adecuada si los datos normalizados están en este rango.")
    logger.info(" - 'tanh': Comprime la salida a [-1,1]. Adecuada para datos normalizados a este rango (ej. Z-scores reescalados) o Z-scores.")
    logger.info(" - 'linear' (None): Sin activación final. La salida puede tomar cualquier valor. Útil si no se desea restringir el rango o si los datos normalizados (ej. Z-scores) no tienen un rango acotado claro.")
    logger.info("La elección depende de la distribución de tus datos después de la normalización y de los objetivos de la reconstrucción.")
    logger.info("Si usas los parámetros MinMax fijos para ciertos canales, estos se mapearán a [0,1], lo que es compatible con 'sigmoid'.")

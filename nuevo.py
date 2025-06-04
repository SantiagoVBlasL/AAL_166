#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_vae_classifier.py

Script para entrenar un Autoencoder Variacional (VAE) Convolucional sobre matrices de 
conectividad funcional fMRI y luego usar sus representaciones latentes para clasificar 
entre Controles Sanos (CN) y pacientes con Enfermedad de Alzheimer (AD).

Versión: 1.4.0 - Thesis Prep Enhancements
Cambios desde v1.3.0:
- Añadido RepeatedStratifiedKFold para la validación cruzada externa del clasificador.
- Implementado annealing cíclico para beta en el entrenamiento del VAE.
- Añadida opción para LayerNorm en las capas FC del VAE.
- Flexibilizada la elección del clasificador (SVM, Logistic Regression, MLP).
- Flexibilizada la métrica de scoring para GridSearchCV del clasificador.
- Mejorado el logging: git hash, descripción de métricas, configuración más detallada.
- Nuevos argumentos para controlar estas funcionalidades.
- Pequeñas refactorizaciones para claridad y extensibilidad.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
import time
import argparse
import gc
import hashlib # Para hash de IDs
from typing import Optional, List, Dict, Tuple, Any, Union
import copy # For deepcopying model state
import subprocess # For git hash

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import (
    StratifiedKFold, 
    GridSearchCV, 
    train_test_split as sk_train_test_split,
    RepeatedStratifiedKFold
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler as SklearnScaler
from sklearn.metrics import (
    roc_auc_score, 
    accuracy_score, 
    recall_score, 
    precision_score, 
    f1_score, 
    average_precision_score,
    balanced_accuracy_score # Added
)

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
                 intermediate_fc_dim_config: Union[int, str] = 0,
                 dropout_rate: float = 0.2,
                 use_layernorm_fc: bool = False): # New parameter
        super(ConvolutionalVAE, self).__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.final_activation_name = final_activation
        self.dropout_rate = dropout_rate
        self.use_layernorm_fc = use_layernorm_fc

        # Encoder Convolutional Layers
        encoder_conv_layers = []
        current_ch_enc = input_channels
        base_conv_channels = [max(16, input_channels*2), max(32, input_channels*4), max(64, input_channels*8)] 
        conv_channels_enc = [min(c, 256) for c in base_conv_channels]

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

        if isinstance(intermediate_fc_dim_config, str) and intermediate_fc_dim_config.lower() == "half":
            self.intermediate_fc_dim = self.flattened_size_after_conv // 2
        elif isinstance(intermediate_fc_dim_config, int) and intermediate_fc_dim_config > 0:
            self.intermediate_fc_dim = intermediate_fc_dim_config
        else:
            self.intermediate_fc_dim = 0
            
        # Encoder FC Layers
        encoder_fc_intermediate_layers = []
        if self.intermediate_fc_dim > 0:
            encoder_fc_intermediate_layers.append(nn.Linear(self.flattened_size_after_conv, self.intermediate_fc_dim))
            if self.use_layernorm_fc:
                encoder_fc_intermediate_layers.append(nn.LayerNorm(self.intermediate_fc_dim))
            encoder_fc_intermediate_layers.extend([
                nn.ReLU(),
                nn.BatchNorm1d(self.intermediate_fc_dim), # BatchNorm often used with ReLU
                nn.Dropout(p=self.dropout_rate)
            ])
            self.encoder_fc_intermediate = nn.Sequential(*encoder_fc_intermediate_layers)
            fc_mu_logvar_input_dim = self.intermediate_fc_dim
            logger.info(f"VAE: Intermediate FC dim (encoder): {self.intermediate_fc_dim}. Using LayerNorm: {self.use_layernorm_fc}")
        else:
            self.encoder_fc_intermediate = None
            fc_mu_logvar_input_dim = self.flattened_size_after_conv
            logger.info("VAE: No intermediate FC layer in encoder.")
        
        logger.info(f"VAE: Input dimension to fc_mu/fc_logvar: {fc_mu_logvar_input_dim}.")
        suggested_latent_dim = fc_mu_logvar_input_dim // 2
        logger.info(f"VAE: Based on this, a suggested latent_dim could be: {suggested_latent_dim} (current is {latent_dim}).")
            
        self.fc_mu = nn.Linear(fc_mu_logvar_input_dim, latent_dim)
        self.fc_logvar = nn.Linear(fc_mu_logvar_input_dim, latent_dim)

        # Decoder FC Layers
        decoder_fc_intermediate_layers = []
        decoder_fc_to_conv_input_dim = latent_dim
        if self.intermediate_fc_dim > 0:
            decoder_fc_intermediate_layers.append(nn.Linear(latent_dim, self.intermediate_fc_dim))
            if self.use_layernorm_fc:
                decoder_fc_intermediate_layers.append(nn.LayerNorm(self.intermediate_fc_dim))
            decoder_fc_intermediate_layers.extend([
                nn.ReLU(),
                nn.BatchNorm1d(self.intermediate_fc_dim),
                nn.Dropout(p=self.dropout_rate)
            ])
            self.decoder_fc_intermediate = nn.Sequential(*decoder_fc_intermediate_layers)
            decoder_fc_to_conv_input_dim = self.intermediate_fc_dim
            logger.info(f"VAE: Intermediate FC dim (decoder): {self.intermediate_fc_dim}. Using LayerNorm: {self.use_layernorm_fc}")
        else:
            self.decoder_fc_intermediate = None
            logger.info("VAE: No intermediate FC layer in decoder.")
            
        self.decoder_fc_to_conv = nn.Linear(decoder_fc_to_conv_input_dim, self.flattened_size_after_conv)

        # Decoder Convolutional Transpose Layers
        decoder_conv_t_layers = []
        current_ch_dec = self.final_conv_output_channels 
        target_conv_t_channels = conv_channels_enc[-2::-1] + [input_channels] 
        
        # Dynamic output_padding calculation (basic heuristic)
        # This is still tricky; precise calculation depends on kernel, stride, padding sequences.
        # The interpolate fallback is important.
        output_paddings_calc = []
        current_spatial_dim = self.final_spatial_dim_encoder
        for i in range(len(target_conv_t_channels)):
            k = kernels[len(kernels)-1-i]
            s = 2 # Stride is 2 for ConvTranspose
            p = paddings[len(paddings)-1-i]
            
            # Target H_out for this layer (approximate doubling, final layer targets image_size)
            if i < len(target_conv_t_channels) - 1:
                 # This logic needs to be more robust if image_size is not a power of 2 multiple of final_spatial_dim_encoder
                 # For image_size 131 and final_spatial_dim_encoder 17:
                 # Target dims are roughly 17 -> 33 -> 66 -> 131 (or whatever image_size is)
                 # Let's assume target is roughly current_spatial_dim * 2
                 # A more robust way would be to calculate the expected output dim of previous layers.
                 # For now, we'll use a simplified approach or stick to fixed if image_size is 131.
                 if image_size == 131 and self.final_spatial_dim_encoder == 17: # Specific known path
                     if i == 0: target_dim_layer = 33 # 17*2-1 = 33 (approx)
                     elif i == 1: target_dim_layer = 66 # 33*2 = 66
                     else: target_dim_layer = image_size
                 else: # General case, less precise
                    target_dim_layer = current_spatial_dim * 2 
                    if i == len(target_conv_t_channels) -1:
                        target_dim_layer = image_size

            else: # Last ConvTranspose layer
                target_dim_layer = image_size

            # H_out = (H_in - 1)*S - 2*P + K + OP
            # OP = Target_H_out - [(H_in - 1)*S - 2*P + K]
            op = target_dim_layer - ((current_spatial_dim - 1) * s - 2 * p + k)
            output_paddings_calc.append(max(0, op)) # OP must be non-negative and < stride or dilation
            
            # Update current_spatial_dim for next iteration's calculation (approximate)
            # This is the H_out if op was perfect.
            # current_spatial_dim = (current_spatial_dim - 1) * s - 2 * p + k + max(0,op)
            # For simplicity, let's use the target for the next step if it's not the last layer
            if i < len(target_conv_t_channels) -1 :
                current_spatial_dim = target_dim_layer
            
        if image_size == 131 and self.final_spatial_dim_encoder == 17 : # If original known path
            output_paddings = [0, 1, 0] 
            logger.info(f"Using fixed output_paddings for image_size=131: {output_paddings}")
        else:
            output_paddings = output_paddings_calc
            logger.warning(f"Image size is {image_size} (final encoder spatial dim: {self.final_spatial_dim_encoder}). Dynamically estimated output_paddings: {output_paddings}. "
                           "Effectiveness depends on architecture; interpolation fallback is crucial.")


        for i in range(len(target_conv_t_channels)):
            op_val = output_paddings[i] if i < len(output_paddings) else 0 # Fallback if list is short
            decoder_conv_t_layers.extend([
                nn.ConvTranspose2d(current_ch_dec, target_conv_t_channels[i], 
                                   kernel_size=kernels[len(kernels)-1-i], 
                                   stride=2, 
                                   padding=paddings[len(paddings)-1-i], 
                                   output_padding=op_val),
                nn.ReLU() if i < len(target_conv_t_channels) - 1 else nn.Identity(),
                nn.BatchNorm2d(target_conv_t_channels[i]) if i < len(target_conv_t_channels) - 1 else nn.Identity(),
            ])
            if i < len(target_conv_t_channels) -1: # Add dropout except for the last conv layer before activation
                 decoder_conv_t_layers.append(nn.Dropout2d(p=self.dropout_rate))
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
            # Using interpolate as the primary method for robust resizing
            logger.debug(f"Shape mismatch: Input {x.shape}, Recon_raw {recon_x_raw.shape}. Adjusting via interpolation.")
            recon_x = nn.functional.interpolate(recon_x_raw, size=(target_h, target_w), mode='bilinear', align_corners=False)
            
            # Fallback crop/pad if interpolation somehow still results in mismatch (should be rare)
            if recon_x.shape[2] != target_h or recon_x.shape[3] != target_w:
                logger.warning(f"Shape mismatch even after interpolate: Input {x.shape}, Recon_interpolated {recon_x.shape}. Attempting crop/pad as fallback.")
                current_h_interp, current_w_interp = recon_x.shape[2], recon_x.shape[3]
                if current_h_interp > target_h or current_w_interp > target_w:
                    h_start = max(0, (current_h_interp - target_h) // 2)
                    w_start = max(0, (current_w_interp - target_w) // 2)
                    recon_x = recon_x[:, :, h_start:h_start + target_h, w_start:w_start + target_w]
                elif current_h_interp < target_h or current_w_interp < target_w:
                    padding_h_total = target_h - current_h_interp
                    padding_w_total = target_w - current_w_interp
                    pad_top = padding_h_total // 2
                    pad_bottom = padding_h_total - pad_top
                    pad_left = padding_w_total // 2
                    pad_right = padding_w_total - pad_left
                    recon_x = nn.functional.pad(recon_x, (pad_left, pad_right, pad_top, pad_bottom))
        
        return recon_x, mu, logvar

# --- Funciones de Pérdida y Schedules ---
def vae_loss_function(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    recon_loss_mse = nn.functional.mse_loss(recon_x, x, reduction='sum') / x.shape[0] 
    kld_element = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kld = torch.mean(kld_element)
    return recon_loss_mse + beta * kld

def get_cyclical_beta_schedule(current_epoch: int, total_epochs: int, beta_max: float, n_cycles: int, ratio_increase: float = 0.5) -> float:
    """
    Calculates beta for cyclical annealing schedule.
    Beta ramps from 0 to beta_max over `ratio_increase` of each cycle, then stays at beta_max.
    """
    if n_cycles <= 0:
        return beta_max # Constant beta if no cycles

    epoch_per_cycle = total_epochs / n_cycles
    # current_cycle_num = current_epoch // epoch_per_cycle # Not directly needed for this schedule type
    epoch_in_current_cycle = current_epoch % epoch_per_cycle
    
    increase_phase_duration = epoch_per_cycle * ratio_increase
    
    if epoch_in_current_cycle < increase_phase_duration:
        # Linearly increase beta from 0 to beta_max
        beta = beta_max * (epoch_in_current_cycle / increase_phase_duration)
    else:
        # Keep beta at beta_max for the remainder of the cycle
        beta = beta_max
    return beta

# --- Funciones Auxiliares de Datos ---
def load_data(tensor_path: Path, metadata_path: Path) -> Tuple[Optional[np.ndarray], Optional[pd.DataFrame]]:
    logger.info(f"Cargando tensor global desde: {tensor_path}")
    if not tensor_path.exists():
        logger.error(f"Archivo de tensor global NO encontrado: {tensor_path}")
        return None, None
    try:
        data_npz = np.load(tensor_path)
        global_tensor = data_npz['global_tensor_data']
        subject_ids_tensor = data_npz['subject_ids'].astype(str) 
        logger.info(f"Tensor global cargado. Forma: {global_tensor.shape}")
        # Warning if channel names don't match expected length (can be ignored if using subset or tensor is different)
        # if global_tensor.shape[1] != len(DEFAULT_CHANNEL_NAMES):
        #     logger.warning(f"El tensor global tiene {global_tensor.shape[1]} canales, pero hay {len(DEFAULT_CHANNEL_NAMES)} nombres de canal por defecto definidos.")
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
    tensor_df['tensor_idx'] = np.arange(len(subject_ids_tensor)) # Index within the loaded tensor
    
    # Merge, ensuring SubjectIDs from tensor are the primary source for indices
    merged_df = pd.merge(tensor_df, metadata_df, on='SubjectID', how='left') # Keep all tensor subjects
    
    num_tensor_subjects = len(subject_ids_tensor)
    num_merged_subjects = len(merged_df)
    
    if num_merged_subjects < num_tensor_subjects:
         logger.warning(f"Algunos SubjectIDs del tensor no se encontraron en los metadatos. "
                       f"Tensor: {num_tensor_subjects}, Merged: {num_merged_subjects}. "
                       "Estos sujetos podrían ser excluidos si la clasificación depende de metadatos faltantes.")
    elif num_merged_subjects > num_tensor_subjects and pd.merge(tensor_df, metadata_df, on='SubjectID', how='inner').shape[0] < num_tensor_subjects:
        # This case implies some IDs in metadata_df might have matched multiple times or other merge issues
        logger.warning(f"Discrepancia en el merge. Tensor subjects: {num_tensor_subjects}, Merged df: {num_merged_subjects}. "
                       f"Inner join count: {pd.merge(tensor_df, metadata_df, on='SubjectID', how='inner').shape[0]}. Verificar IDs duplicados.")

    # Filter out subjects from tensor that are not in metadata if metadata is crucial for all steps
    # However, for VAE training, we might want to use all tensor data if labels are not needed.
    # For now, keep all subjects from the tensor, and handle missing metadata downstream.
    final_df = merged_df.copy()
    return global_tensor, final_df


def normalize_inter_channel_fold(
    data_tensor: np.ndarray, 
    train_indices_in_fold: np.ndarray, 
    mode: str = 'zscore_offdiag',
    selected_channel_original_names: Optional[List[str]] = None
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    
    num_subjects_total, num_selected_channels, num_rois, _ = data_tensor.shape
    logger.info(f"Aplicando normalización inter-canal (modo: {mode}) sobre {num_selected_channels} canales seleccionados.")
    logger.info(f"Parámetros de normalización se calcularán usando {len(train_indices_in_fold)} sujetos de entrenamiento.")
    
    normalized_tensor_fold = data_tensor.copy()
    norm_params_per_channel_list = []
    off_diag_mask = ~np.eye(num_rois, dtype=bool)

    for c_idx_selected in range(num_selected_channels):
        current_channel_original_name = selected_channel_original_names[c_idx_selected] if selected_channel_original_names and c_idx_selected < len(selected_channel_original_names) else f"Channel_{c_idx_selected}"
        params = {'mode': mode, 'original_name': current_channel_original_name}
        use_fixed_params = False

        if mode == 'minmax_offdiag' and current_channel_original_name in FIXED_MINMAX_PARAMS_PER_CHANNEL:
            fixed_p = FIXED_MINMAX_PARAMS_PER_CHANNEL[current_channel_original_name]
            params.update({'min': fixed_p['min'], 'max': fixed_p['max']})
            use_fixed_params = True
            logger.info(f"Canal '{current_channel_original_name}': Usando MinMax fijo (min={params['min']:.4f}, max={params['max']:.4f}).")

        if not use_fixed_params:
            channel_data_train_for_norm_params = data_tensor[train_indices_in_fold, c_idx_selected, :, :]
            all_off_diag_train_values = channel_data_train_for_norm_params[:, off_diag_mask].flatten()

            if all_off_diag_train_values.size == 0:
                logger.warning(f"Canal '{current_channel_original_name}': No hay elementos fuera de la diagonal en el training set. No se escala.")
                params.update({'mean': 0.0, 'std': 1.0, 'min': 0.0, 'max': 1.0, 'no_scale': True})
            elif mode == 'zscore_offdiag':
                mean_val = np.mean(all_off_diag_train_values)
                std_val = np.std(all_off_diag_train_values)
                params.update({'mean': mean_val, 'std': std_val if std_val > 1e-9 else 1.0})
                if std_val <= 1e-9: logger.warning(f"Canal '{current_channel_original_name}': STD muy bajo ({std_val:.2e}). Usando STD=1.")
            elif mode == 'minmax_offdiag':
                min_val = np.min(all_off_diag_train_values)
                max_val = np.max(all_off_diag_train_values)
                params.update({'min': min_val, 'max': max_val})
                if (max_val - min_val) <= 1e-9: logger.warning(f"Canal '{current_channel_original_name}': Rango (max-min) muy bajo ({(max_val - min_val):.2e}).")
            else:
                raise ValueError(f"Modo de normalización desconocido: {mode}")
        
        norm_params_per_channel_list.append(params)

        if not params.get('no_scale', False):
            # Apply to all subjects in data_tensor (train, val, test for this VAE fold)
            current_channel_data_all_subjects = data_tensor[:, c_idx_selected, :, :]
            scaled_channel_data = current_channel_data_all_subjects.copy()
            
            if off_diag_mask.any():
                if mode == 'zscore_offdiag':
                    if params['std'] > 1e-9:
                        scaled_channel_data[:, off_diag_mask] = (current_channel_data_all_subjects[:, off_diag_mask] - params['mean']) / params['std']
                elif mode == 'minmax_offdiag':
                    range_val = params.get('max', 1.0) - params.get('min', 0.0)
                    if range_val > 1e-9: 
                        scaled_channel_data[:, off_diag_mask] = (current_channel_data_all_subjects[:, off_diag_mask] - params['min']) / range_val
                    else: 
                        # Avoid division by zero, map to 0 or 0.5 depending on desired behavior for constant data
                        scaled_channel_data[:, off_diag_mask] = 0.0 
            normalized_tensor_fold[:, c_idx_selected, :, :] = scaled_channel_data
            
            if not use_fixed_params:
                log_msg_params = f"Canal '{current_channel_original_name}': Off-diag {mode} (train_params: "
                if mode == 'zscore_offdiag': log_msg_params += f"mean={params['mean']:.3f}, std={params['std']:.3f})"
                elif mode == 'minmax_offdiag': log_msg_params += f"min={params['min']:.3f}, max={params['max']:.3f})"
                logger.info(log_msg_params)
            
    return normalized_tensor_fold, norm_params_per_channel_list


def apply_normalization_params(data_tensor_subset: np.ndarray, 
                               norm_params_per_channel_list: List[Dict[str, float]]
                               ) -> np.ndarray:
    num_subjects, num_selected_channels, num_rois, _ = data_tensor_subset.shape
    logger.info(f"Aplicando parámetros de normalización precalculados a subconjunto de datos ({num_subjects} sujetos, {num_selected_channels} canales).")
    normalized_tensor_subset = data_tensor_subset.copy()
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

        current_channel_data = data_tensor_subset[:, c_idx_selected, :, :]
        scaled_channel_data_subset = current_channel_data.copy()

        if off_diag_mask.any():
            if mode == 'zscore_offdiag':
                if params['std'] > 1e-9:
                    scaled_channel_data_subset[:, off_diag_mask] = (current_channel_data[:, off_diag_mask] - params['mean']) / params['std']
            elif mode == 'minmax_offdiag':
                range_val = params.get('max', 1.0) - params.get('min', 0.0)
                if range_val > 1e-9:
                    scaled_channel_data_subset[:, off_diag_mask] = (current_channel_data[:, off_diag_mask] - params['min']) / range_val
                else:
                    scaled_channel_data_subset[:, off_diag_mask] = 0.0 
        normalized_tensor_subset[:, c_idx_selected, :, :] = scaled_channel_data_subset
    return normalized_tensor_subset

# --- Función Principal de Entrenamiento y Evaluación ---
def train_and_evaluate_pipeline(global_tensor_all_channels: np.ndarray, 
                                metadata_df_full: pd.DataFrame, # Renamed for clarity
                                args: argparse.Namespace):
    
    # --- Channel Selection ---
    selected_channel_indices: List[int] = []
    selected_channel_names_in_tensor: List[str] = [] # Names corresponding to selected channels in current_global_tensor

    if args.channels_to_use:
        for ch_specifier in args.channels_to_use:
            try:
                ch_idx = int(ch_specifier) # Attempt to interpret as index
                if 0 <= ch_idx < global_tensor_all_channels.shape[1]: # Check against actual tensor dimension
                    selected_channel_indices.append(ch_idx)
                    # Try to get name from DEFAULT_CHANNEL_NAMES if original tensor had that many channels
                    if ch_idx < len(DEFAULT_CHANNEL_NAMES):
                         selected_channel_names_in_tensor.append(DEFAULT_CHANNEL_NAMES[ch_idx])
                    else:
                         selected_channel_names_in_tensor.append(f"RawChan{ch_idx}")
                else:
                    logger.warning(f"Índice de canal '{ch_idx}' fuera de rango para el tensor (0 a {global_tensor_all_channels.shape[1]-1}). Ignorando.")
            except ValueError: # Interpret as name
                if ch_specifier in DEFAULT_CHANNEL_NAMES:
                    try:
                        original_idx = DEFAULT_CHANNEL_NAMES.index(ch_specifier)
                        if original_idx < global_tensor_all_channels.shape[1]:
                             selected_channel_indices.append(original_idx)
                             selected_channel_names_in_tensor.append(ch_specifier)
                        else:
                            logger.warning(f"Nombre de canal '{ch_specifier}' (índice {original_idx}) fuera de rango para el tensor. Ignorando.")
                    except ValueError: # Should not happen if ch_specifier in DEFAULT_CHANNEL_NAMES
                        logger.warning(f"Nombre de canal '{ch_specifier}' en DEFAULT_CHANNEL_NAMES pero no se pudo obtener índice. Ignorando.")
                else:
                    logger.warning(f"Nombre de canal '{ch_specifier}' desconocido. Canales por defecto: {DEFAULT_CHANNEL_NAMES}. Ignorando.")
        
        if not selected_channel_indices:
            logger.error("No se seleccionaron canales válidos. Abortando.")
            return
        
        # Ensure selected_channel_indices are unique and sorted for consistent slicing
        selected_channel_indices = sorted(list(set(selected_channel_indices)))
        # Rebuild names based on sorted unique indices
        temp_names = []
        for idx in selected_channel_indices:
            if idx < len(DEFAULT_CHANNEL_NAMES): temp_names.append(DEFAULT_CHANNEL_NAMES[idx])
            else: temp_names.append(f"RawChan{idx}")
        selected_channel_names_in_tensor = temp_names

        logger.info(f"Usando canales seleccionados (índices en tensor original): {selected_channel_indices}")
        logger.info(f"Nombres de canales seleccionados: {selected_channel_names_in_tensor}")
        current_global_tensor = global_tensor_all_channels[:, selected_channel_indices, :, :]
    else:
        logger.info(f"Usando todos los {global_tensor_all_channels.shape[1]} canales disponibles del tensor.")
        current_global_tensor = global_tensor_all_channels
        # Try to assign default names if tensor matches, otherwise generic names
        if global_tensor_all_channels.shape[1] <= len(DEFAULT_CHANNEL_NAMES):
            selected_channel_names_in_tensor = DEFAULT_CHANNEL_NAMES[:global_tensor_all_channels.shape[1]]
        else:
            selected_channel_names_in_tensor = [f"RawChan{i}" for i in range(global_tensor_all_channels.shape[1])]
        selected_channel_indices = list(range(global_tensor_all_channels.shape[1]))


    num_input_channels_for_vae = current_global_tensor.shape[1]

    # 1. Filtrar para clasificación CN vs AD y preparar etiquetas
    # Ensure 'ResearchGroup' exists
    if 'ResearchGroup' not in metadata_df_full.columns:
        logger.error("'ResearchGroup' no encontrado en metadatos. No se puede proceder con la clasificación CN/AD.")
        return
        
    cn_ad_df = metadata_df_full[metadata_df_full['ResearchGroup'].isin(['CN', 'AD'])].copy()
    if cn_ad_df.empty:
        logger.error("No se encontraron sujetos CN o AD en los metadatos (columna 'ResearchGroup').")
        return
    if 'tensor_idx' not in cn_ad_df.columns: # Should be there from load_data
        logger.error("'tensor_idx' no encontrado en cn_ad_df después del merge. Error en carga/merge de datos.")
        return

    label_mapping = {'CN': 0, 'AD': 1}
    cn_ad_df['label'] = cn_ad_df['ResearchGroup'].map(label_mapping)
    
    # Stratification key for classifier's outer CV
    stratification_columns_present = ['ResearchGroup'] # Always stratify by group
    if args.classifier_stratify_cols:
        for col in args.classifier_stratify_cols:
            if col in cn_ad_df.columns:
                # Handle NaNs by converting to a string placeholder
                cn_ad_df[col] = cn_ad_df[col].fillna(f"{col}_Unknown").astype(str)
                stratification_columns_present.append(col)
            else:
                logger.warning(f"Columna de estratificación '{col}' no encontrada en metadatos. Se ignorará.")
    
    if len(stratification_columns_present) > 1:
        cn_ad_df['stratify_key'] = cn_ad_df[stratification_columns_present].apply(lambda x: '_'.join(x.astype(str)), axis=1)
        logger.info(f"Estratificando folds del clasificador por: {stratification_columns_present}")
    else:
        cn_ad_df['stratify_key'] = cn_ad_df['label'].astype(str) # Fallback to just label
        logger.info("Estratificando folds del clasificador solo por ResearchGroup (label).")
    
    stratify_labels_for_cv = cn_ad_df['stratify_key']
    X_classifier_indices_global = cn_ad_df['tensor_idx'].values # Indices into current_global_tensor
    y_classifier_labels_cn_ad = cn_ad_df['label'].values

    logger.info(f"Iniciando clasificación CN vs AD. Total sujetos CN/AD disponibles en metadatos y tensor: {len(cn_ad_df)}. "
                f"CN: {sum(y_classifier_labels_cn_ad == 0)}, AD: {sum(y_classifier_labels_cn_ad == 1)}")

    # 2. Configuración de Nested Cross-Validation
    if args.repeated_outer_folds_n_repeats > 1:
        outer_cv_clf = RepeatedStratifiedKFold(n_splits=args.outer_folds, 
                                               n_repeats=args.repeated_outer_folds_n_repeats, 
                                               random_state=args.seed)
        total_outer_iterations = args.outer_folds * args.repeated_outer_folds_n_repeats
        logger.info(f"Usando RepeatedStratifiedKFold con {args.outer_folds} splits y {args.repeated_outer_folds_n_repeats} repeats.")
    else:
        outer_cv_clf = StratifiedKFold(n_splits=args.outer_folds, shuffle=True, random_state=args.seed)
        total_outer_iterations = args.outer_folds
        logger.info(f"Usando StratifiedKFold con {args.outer_folds} splits.")
    
    fold_metrics_list = [] # Renamed for clarity
    
    # The split is on an array of zeros, using stratify_labels_for_cv for grouping
    for fold_idx, (train_dev_clf_local_indices, test_clf_local_indices) in enumerate(outer_cv_clf.split(np.zeros(len(cn_ad_df)), stratify_labels_for_cv)):
        logger.info(f"--- Iniciando Fold Externo {fold_idx + 1}/{total_outer_iterations} ---")

        # Map local indices (from cn_ad_df) to global tensor indices
        global_indices_clf_train_dev = X_classifier_indices_global[train_dev_clf_local_indices]
        global_indices_clf_test = X_classifier_indices_global[test_clf_local_indices]
        
        # VAE training data: All subjects from metadata_df_full (potentially including non-CN/AD) 
        # EXCEPT those in the current classifier *test* set (global_indices_clf_test).
        # This ensures VAE learns general features without peeking at classifier test set.
        all_subject_global_indices_in_metadata = metadata_df_full['tensor_idx'].values
        
        # Ensure all_subject_global_indices_in_metadata are valid indices for current_global_tensor
        valid_max_idx = current_global_tensor.shape[0] -1
        all_subject_global_indices_in_metadata = all_subject_global_indices_in_metadata[all_subject_global_indices_in_metadata <= valid_max_idx]

        global_indices_vae_train_val = np.setdiff1d(all_subject_global_indices_in_metadata, global_indices_clf_test, assume_unique=True)
        
        if len(global_indices_vae_train_val) == 0:
            logger.error(f"Fold {fold_idx+1}: No subjects remaining for VAE training after excluding classifier test set. Skipping fold.")
            continue

        vae_train_val_tensor_original_scale = current_global_tensor[global_indices_vae_train_val]
        
        # Indices for splitting vae_train_val_tensor_original_scale
        vae_train_indices_local, vae_val_indices_local = [], []
        if args.vae_val_split_ratio > 0 and len(global_indices_vae_train_val) > 10 :
             # Stratification for VAE val split is complex if non-CN/AD subjects are included.
             # For simplicity, using random split for VAE validation.
             # The VAE is unsupervised; its validation is for reconstruction quality.
             try:
                vae_train_indices_local, vae_val_indices_local = sk_train_test_split(
                    np.arange(len(global_indices_vae_train_val)), 
                    test_size=args.vae_val_split_ratio,
                    random_state=args.seed + fold_idx # Vary seed per fold for VAE split
                )
             except ValueError as e: # Handles cases where split is not possible (e.g. too few samples)
                logger.warning(f"Could not perform VAE validation split for fold {fold_idx+1}: {e}. Using all data for VAE training.")
                vae_train_indices_local = np.arange(len(global_indices_vae_train_val))
        else: 
            vae_train_indices_local = np.arange(len(global_indices_vae_train_val))
            logger.info("No VAE validation split performed (ratio <=0 or too few samples).")

        logger.info("Normalizando datos para VAE...")
        vae_train_val_tensor_norm, norm_params_fold_list = normalize_inter_channel_fold(
            vae_train_val_tensor_original_scale, 
            vae_train_indices_local, # Params from VAE training portion
            mode=args.norm_mode,
            selected_channel_original_names=selected_channel_names_in_tensor
        )

        vae_train_dataset_fold = TensorDataset(torch.from_numpy(vae_train_val_tensor_norm[vae_train_indices_local]).float())
        vae_train_dataloader = DataLoader(vae_train_dataset_fold, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        
        vae_val_dataloader = None
        if len(vae_val_indices_local) > 0:
            vae_val_dataset_fold = TensorDataset(torch.from_numpy(vae_train_val_tensor_norm[vae_val_indices_local]).float())
            vae_val_dataloader = DataLoader(vae_val_dataset_fold, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Usando dispositivo: {device}")
        
        vae = ConvolutionalVAE(
            input_channels=num_input_channels_for_vae, 
            latent_dim=args.latent_dim, 
            image_size=current_global_tensor.shape[-1], # ROI count
            final_activation=args.vae_final_activation,
            intermediate_fc_dim_config=args.intermediate_fc_dim_vae,
            dropout_rate=args.dropout_rate_vae,
            use_layernorm_fc=args.use_layernorm_vae_fc
            ).to(device)
        optimizer_vae = optim.Adam(vae.parameters(), lr=args.lr_vae, weight_decay=args.weight_decay_vae)
        
        scheduler_vae = None
        if vae_val_dataloader and args.lr_scheduler_patience_vae > 0 :
            scheduler_vae = optim.lr_scheduler.ReduceLROnPlateau(optimizer_vae, 'min', 
                                                                 patience=args.lr_scheduler_patience_vae, 
                                                                 factor=0.1, verbose=False) # Verbose can be noisy
            logger.info(f"LR Scheduler ReduceLROnPlateau activado con paciencia {args.lr_scheduler_patience_vae}.")

        logger.info(f"Entrenando VAE para el fold {fold_idx + 1}...")
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_state_dict = None

        for epoch in range(args.epochs_vae):
            vae.train()
            epoch_train_loss = 0
            
            current_beta_vae = get_cyclical_beta_schedule(
                epoch, args.epochs_vae, args.beta_vae, 
                args.cyclical_beta_n_cycles, args.cyclical_beta_ratio_increase
            )

            for batch_idx, (data,) in enumerate(vae_train_dataloader):
                data = data.to(device)
                optimizer_vae.zero_grad()
                recon_batch, mu, logvar = vae(data)
                loss = vae_loss_function(recon_batch, data, mu, logvar, beta=current_beta_vae)
                loss.backward()
                optimizer_vae.step()
                epoch_train_loss += loss.item() * data.size(0) # loss.item() is avg loss for batch
            
            avg_epoch_train_loss = epoch_train_loss / len(vae_train_dataloader.dataset)
            current_lr = optimizer_vae.param_groups[0]['lr']
            log_msg = (f"Fold {fold_idx+1}, VAE Epoch {epoch+1}/{args.epochs_vae}, "
                       f"Train Loss: {avg_epoch_train_loss:.4f}, Beta: {current_beta_vae:.4f}, LR: {current_lr:.2e}")

            if vae_val_dataloader:
                vae.eval()
                epoch_val_loss = 0
                with torch.no_grad():
                    for val_data, in vae_val_dataloader:
                        val_data = val_data.to(device)
                        recon_val, mu_val, logvar_val = vae(val_data)
                        # Use current_beta_vae for val loss consistency, or args.beta_vae if preferred for stable val metric
                        val_loss_batch = vae_loss_function(recon_val, val_data, mu_val, logvar_val, beta=current_beta_vae) 
                        epoch_val_loss += val_loss_batch.item() * val_data.size(0)
                avg_epoch_val_loss = epoch_val_loss / len(vae_val_dataloader.dataset)
                log_msg += f", Val Loss: {avg_epoch_val_loss:.4f}"

                if scheduler_vae:
                    scheduler_vae.step(avg_epoch_val_loss)

                if avg_epoch_val_loss < best_val_loss:
                    best_val_loss = avg_epoch_val_loss
                    epochs_no_improve = 0
                    best_model_state_dict = copy.deepcopy(vae.state_dict())
                    logger.debug(f"Fold {fold_idx+1}, Epoch {epoch+1}: New best VAE val_loss: {best_val_loss:.4f}")
                else:
                    epochs_no_improve += 1
                
                if args.early_stopping_patience_vae > 0 and epochs_no_improve >= args.early_stopping_patience_vae:
                    logger.info(f"Fold {fold_idx+1}: Early stopping VAE en epoch {epoch+1}. Mejor val_loss: {best_val_loss:.4f}")
                    break
            
            if (epoch + 1) % args.log_interval_epochs_vae == 0 or epoch == args.epochs_vae -1 : 
                logger.info(log_msg)
        
        if best_model_state_dict and vae_val_dataloader: # Only load best if validation was performed
            logger.info(f"Fold {fold_idx+1}: Cargando mejor modelo VAE con val_loss: {best_val_loss:.4f}")
            vae.load_state_dict(best_model_state_dict)
        else:
            logger.info(f"Fold {fold_idx+1}: Usando el último modelo VAE entrenado (sin validación VAE o sin mejora).")

        # Save VAE model for this fold
        vae_model_fname = (f"vae_fold_{fold_idx+1}_ld{args.latent_dim}_beta{args.beta_vae}_"
                           f"ch{num_input_channels_for_vae}_{args.norm_mode}.pt")
        vae_model_path = Path(args.output_dir) / vae_model_fname
        torch.save(vae.state_dict(), vae_model_path)
        logger.info(f"Modelo VAE del fold {fold_idx+1} guardado en: {vae_model_path}")
        
        # --- Classifier Stage ---
        # Prepare data for classifier: use global_indices_clf_train_dev and global_indices_clf_test
        # These indices are for current_global_tensor
        clf_train_dev_tensor_original_scale = current_global_tensor[global_indices_clf_train_dev]
        clf_train_dev_tensor_norm = apply_normalization_params(clf_train_dev_tensor_original_scale, norm_params_fold_list)
        y_train_clf_labels = y_classifier_labels_cn_ad[train_dev_clf_local_indices] # Corresponding labels

        clf_test_tensor_original_scale = current_global_tensor[global_indices_clf_test]
        clf_test_tensor_norm = apply_normalization_params(clf_test_tensor_original_scale, norm_params_fold_list)
        y_test_clf_labels = y_classifier_labels_cn_ad[test_clf_local_indices] # Corresponding labels
        
        vae.eval()
        with torch.no_grad():
            mu_train_clf, _ = vae.encode(torch.from_numpy(clf_train_dev_tensor_norm).float().to(device))
            mu_test_clf, _ = vae.encode(torch.from_numpy(clf_test_tensor_norm).float().to(device))
        
        X_train_latent = mu_train_clf.cpu().numpy()
        X_test_latent = mu_test_clf.cpu().numpy()

        if X_train_latent.shape[0] == 0 or X_test_latent.shape[0] == 0:
            logger.warning(f"Fold {fold_idx+1}: No data for classifier (train shape {X_train_latent.shape}, test shape {X_test_latent.shape}). Skipping.")
            fold_metrics_list.append({'fold': fold_idx + 1, 'auc': np.nan, 'pr_auc': np.nan, 'accuracy': np.nan, 
                                 'balanced_accuracy': np.nan, 'sensitivity': np.nan, 'specificity': np.nan, 
                                 'f1_score': np.nan, 'best_clf_params': {}, 'classifier_type': args.classifier_type,
                                 'num_selected_channels': num_input_channels_for_vae,
                                 'selected_channel_names': ";".join(selected_channel_names_in_tensor)})
            continue

        scaler_latent = SklearnScaler()
        X_train_latent_scaled = scaler_latent.fit_transform(X_train_latent)
        X_test_latent_scaled = scaler_latent.transform(X_test_latent)

        # Classifier definition and param grid
        classifier, param_grid_clf = None, {}
        clf_class_weight = 'balanced' if args.classifier_use_class_weight else None

        if args.classifier_type == 'svm':
            classifier = SVC(probability=True, random_state=args.seed, class_weight=clf_class_weight)
            param_grid_clf = {'C': [0.01, 0.1, 1, 10, 100], 'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 'scale', 'auto'], 'kernel': ['rbf']}
        elif args.classifier_type == 'logreg':
            classifier = LogisticRegression(random_state=args.seed, class_weight=clf_class_weight, solver='liblinear', max_iter=1000)
            param_grid_clf = {'C': [0.01, 0.1, 1, 10, 100]}
        elif args.classifier_type == 'mlp':
            hidden_layer_sizes = tuple(map(int, args.mlp_classifier_hidden_layers.split(',')))
            classifier = MLPClassifier(random_state=args.seed, hidden_layer_sizes=hidden_layer_sizes, 
                                       max_iter=500, early_stopping=True, n_iter_no_change=10)
            param_grid_clf = {'alpha': [0.0001, 0.001, 0.01], 'learning_rate_init': [0.001, 0.005, 0.01]}
        else:
            raise ValueError(f"Unsupported classifier_type: {args.classifier_type}")

        inner_cv_clf = StratifiedKFold(n_splits=args.inner_folds, shuffle=True, random_state=args.seed + fold_idx + 100) 
        
        # Stratify inner CV for GridSearchCV using the same stratify_key logic if possible, else just labels
        # This needs labels corresponding to X_train_latent_scaled, which are y_train_clf_labels
        # And the stratification key for these samples: stratify_labels_for_cv[train_dev_clf_local_indices]
        grid_search_stratify_labels = stratify_labels_for_cv.iloc[train_dev_clf_local_indices]

        clf_gscv = GridSearchCV(classifier, param_grid_clf, cv=inner_cv_clf, 
                                scoring=args.gridsearch_scoring, n_jobs=args.num_workers, verbose=0) # n_jobs=-1 can be problematic with some setups
        
        logger.info(f"Entrenando clasificador {args.classifier_type} para el fold {fold_idx + 1} (scoring: {args.gridsearch_scoring})...")
        clf_gscv.fit(X_train_latent_scaled, y_train_clf_labels) # Pass stratify_labels for inner CV if GridSearchCV supports it directly (it doesn't, StratifiedKFold handles it)
        logger.info(f"Mejores hiperparámetros {args.classifier_type} para fold {fold_idx + 1}: {clf_gscv.best_params_}")

        y_pred_proba_clf_test = clf_gscv.predict_proba(X_test_latent_scaled)[:, 1]
        y_pred_clf_test = clf_gscv.predict(X_test_latent_scaled)

        # Calculate metrics
        auc = roc_auc_score(y_test_clf_labels, y_pred_proba_clf_test)
        pr_auc = average_precision_score(y_test_clf_labels, y_pred_proba_clf_test)
        acc = accuracy_score(y_test_clf_labels, y_pred_clf_test)
        bal_acc = balanced_accuracy_score(y_test_clf_labels, y_pred_clf_test)
        sens = recall_score(y_test_clf_labels, y_pred_clf_test, pos_label=1, zero_division=0) 
        spec = recall_score(y_test_clf_labels, y_pred_clf_test, pos_label=0, zero_division=0) 
        f1 = f1_score(y_test_clf_labels, y_pred_clf_test, pos_label=1, zero_division=0)
        
        logger.info(f"Fold {fold_idx + 1} - Resultados Clasificador ({args.classifier_type}):")
        logger.info(f"  AUC: {auc:.4f}, PR-AUC: {pr_auc:.4f}, Acc: {acc:.4f}, Bal. Acc: {bal_acc:.4f}")
        logger.info(f"  Sens (AD): {sens:.4f}, Spec (CN): {spec:.4f}, F1 (AD): {f1:.4f}")
        
        # Subject ID hashing for traceability (using SubjectIDs from cn_ad_df for this fold)
        train_subject_ids_fold = cn_ad_df.iloc[train_dev_clf_local_indices]['SubjectID'].tolist()
        test_subject_ids_fold = cn_ad_df.iloc[test_clf_local_indices]['SubjectID'].tolist()
        train_ids_str = ",".join(sorted(train_subject_ids_fold))
        test_ids_str = ",".join(sorted(test_subject_ids_fold))
        logger.info(f"  Fold {fold_idx+1} Train (Clf) IDs Hash: {hashlib.md5(train_ids_str.encode()).hexdigest()}")
        logger.info(f"  Fold {fold_idx+1} Test (Clf) IDs Hash: {hashlib.md5(test_ids_str.encode()).hexdigest()}")

        fold_metrics_list.append({'fold': fold_idx + 1, 'auc': auc, 'pr_auc': pr_auc, 'accuracy': acc, 
                             'balanced_accuracy': bal_acc, 'sensitivity': sens, 'specificity': spec, 'f1_score': f1, 
                             'best_clf_params': clf_gscv.best_params_,
                             'classifier_type': args.classifier_type,
                             'num_selected_channels': num_input_channels_for_vae,
                             'selected_channel_names': ";".join(selected_channel_names_in_tensor)})
        
        del vae, optimizer_vae, vae_train_dataloader, vae_val_dataloader, scheduler_vae
        del clf_gscv, classifier, X_train_latent, X_test_latent, X_train_latent_scaled, X_test_latent_scaled
        gc.collect()
        if device.type == 'cuda': torch.cuda.empty_cache()

    if fold_metrics_list:
        metrics_df = pd.DataFrame(fold_metrics_list)
        logger.info("\n--- Resumen de Rendimiento (Promedio sobre Folds Externos) ---")
        for metric in ['auc', 'pr_auc', 'accuracy', 'balanced_accuracy', 'sensitivity', 'specificity', 'f1_score']:
            if metric in metrics_df.columns: # Check if metric exists (e.g. if all were NaN)
                mean_val = metrics_df[metric].mean()
                std_val = metrics_df[metric].std()
                logger.info(f"{metric.capitalize():<20}: {mean_val:.4f} +/- {std_val:.4f}")
        
        # Build filename
        fname_parts = [
            "clf_results", args.classifier_type,
            "vae", args.norm_mode, 
            f"ld{args.latent_dim}", f"beta{args.beta_vae}",
            f"cycBeta{args.cyclical_beta_n_cycles}" if args.cyclical_beta_n_cycles > 0 else "",
            f"ch{num_input_channels_for_vae}{'sel' if args.channels_to_use else 'all'}",
            f"intFC{args.intermediate_fc_dim_vae}",
            f"dropVAE{args.dropout_rate_vae}",
            f"lnFC{args.use_layernorm_vae_fc}",
            f"esVAE{args.early_stopping_patience_vae}",
            f"outer{args.outer_folds}x{args.repeated_outer_folds_n_repeats if args.repeated_outer_folds_n_repeats > 1 else '1'}",
            f"score{args.gridsearch_scoring}"
        ]
        # Filter out empty strings from fname_parts that might occur if a boolean arg leads to empty string
        results_filename = "_".join(str(p) for p in fname_parts if str(p)) + ".csv"
        results_path = Path(args.output_dir) / results_filename
        results_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(results_path, index=False)
        logger.info(f"Resultados de clasificación guardados en: {results_path}")

        # Save summary statistics of metrics
        summary_stats_path = Path(args.output_dir) / ("summary_stats_" + results_filename.replace(".csv", ".txt"))
        with open(summary_stats_path, 'w') as f:
            f.write(f"Run Arguments:\n{vars(args)}\n\n")
            f.write(f"Git Commit Hash: {args.git_hash}\n\n")
            f.write("Metrics Summary Statistics:\n")
            f.write(metrics_df.describe().to_string())
        logger.info(f"Sumario estadístico de métricas guardado en: {summary_stats_path}")

    else:
        logger.warning("No se pudieron calcular métricas para ningún fold.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de Entrenamiento VAE y Clasificador para AD vs CN (v1.4.0)",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # --- Paths and Data Arguments ---
    group_data = parser.add_argument_group('Data and Paths')
    group_data.add_argument("--global_tensor_path", type=str, required=True, help="Ruta al archivo .npz del tensor global.")
    group_data.add_argument("--metadata_path", type=str, required=True, help="Ruta al archivo CSV de metadatos.")
    group_data.add_argument("--output_dir", type=str, default="./vae_clf_output_v1.4", help="Directorio para guardar resultados.")
    group_data.add_argument("--channels_to_use", type=str, nargs='*', default=None, 
                        help="Lista de nombres o índices de canales a usar (desde 0). Ej: 0 5 Pearson_Full_FisherZ_Signed. Si no se provee, usa todos los del tensor.")

    # --- Cross-validation Arguments ---
    group_cv = parser.add_argument_group('Cross-validation')
    group_cv.add_argument("--outer_folds", type=int, default=5, help="Número de folds para CV externa del clasificador.")
    group_cv.add_argument("--repeated_outer_folds_n_repeats", type=int, default=1, help="Número de repeticiones para RepeatedStratifiedKFold (si >1).")
    group_cv.add_argument("--inner_folds", type=int, default=3, help="Número de folds para CV interna (GridSearchCV).")
    group_cv.add_argument("--classifier_stratify_cols", type=str, nargs='*', default=['Sex'], help="Columnas adicionales en metadatos para estratificación del clasificador (además de ResearchGroup). Ej: Sex AgeGroup Site. Default: Sex.")

    # --- VAE Model and Training Arguments ---
    group_vae = parser.add_argument_group('VAE Model and Training')
    group_vae.add_argument("--latent_dim", type=int, default=64, help="Dimensión del espacio latente del VAE.")
    group_vae.add_argument("--lr_vae", type=float, default=1e-4, help="Tasa de aprendizaje para el VAE.")
    group_vae.add_argument("--epochs_vae", type=int, default=150, help="Número máximo de épocas para entrenar el VAE.")
    group_vae.add_argument("--batch_size", type=int, default=16, help="Tamaño del batch para entrenamiento VAE y clasificador (donde aplique).")
    group_vae.add_argument("--beta_vae", type=float, default=1.0, help="Peso del término KLD en la pérdida del VAE (beta_max para annealing).")
    group_vae.add_argument("--cyclical_beta_n_cycles", type=int, default=0, help="Número de ciclos para annealing de beta (0 para beta constante).")
    group_vae.add_argument("--cyclical_beta_ratio_increase", type=float, default=0.5, help="Proporción de cada ciclo para aumentar beta (si cyclical_beta_n_cycles > 0).")
    group_vae.add_argument("--weight_decay_vae", type=float, default=1e-5, help="Decaimiento de peso (L2 reg) para el optimizador del VAE.")
    group_vae.add_argument("--vae_final_activation", type=str, default="tanh", choices=["sigmoid", "tanh", "linear"], help="Activación final del decoder del VAE.")
    group_vae.add_argument("--intermediate_fc_dim_vae", type=str, default="0", 
                        help="Dimensión de la capa FC intermedia en VAE. Entero > 0, '0' (deshabilitado), o 'half'.")
    group_vae.add_argument("--dropout_rate_vae", type=float, default=0.2, help="Tasa de dropout en el VAE.")
    group_vae.add_argument("--use_layernorm_vae_fc", action='store_true', help="Usar LayerNorm en las capas FC del VAE.")
    group_vae.add_argument("--vae_val_split_ratio", type=float, default=0.15, help="Proporción de datos para validación del VAE (si >0).")
    group_vae.add_argument("--early_stopping_patience_vae", type=int, default=20, help="Paciencia para early stopping en VAE (0 para deshabilitar).")
    group_vae.add_argument("--lr_scheduler_patience_vae", type=int, default=10, help="Paciencia para scheduler ReduceLROnPlateau en VAE (0 para deshabilitar).")

    # --- Classifier Arguments ---
    group_clf = parser.add_argument_group('Classifier')
    group_clf.add_argument("--classifier_type", type=str, default="svm", choices=["svm", "logreg", "mlp"], help="Tipo de clasificador a usar.")
    group_clf.add_argument("--gridsearch_scoring", type=str, default="roc_auc", choices=["roc_auc", "accuracy", "balanced_accuracy", "f1"], 
                           help="Métrica para optimizar en GridSearchCV.")
    group_clf.add_argument("--classifier_use_class_weight", action='store_true', help="Usar class_weight='balanced' en el clasificador (si soportado).")
    group_clf.add_argument("--mlp_classifier_hidden_layers", type=str, default="100", help="Tamaños de capas ocultas para MLP, separados por coma. Ej: '100,50'.")

    # --- General Arguments ---
    group_general = parser.add_argument_group('General Settings')
    group_general.add_argument("--norm_mode", type=str, default="zscore_offdiag", choices=["zscore_offdiag", "minmax_offdiag"], help="Modo de normalización inter-canal.")
    group_general.add_argument("--seed", type=int, default=42, help="Semilla para reproducibilidad.")
    group_general.add_argument("--num_workers", type=int, default=2, help="Número de workers para DataLoader.")
    group_general.add_argument("--log_interval_epochs_vae", type=int, default=10, help="Intervalo de épocas para loguear entrenamiento VAE.")
    
    args = parser.parse_args()

    # --- Post-processing and Validation of Arguments ---
    # Convert intermediate_fc_dim_vae
    try:
        args.intermediate_fc_dim_vae = int(args.intermediate_fc_dim_vae)
    except ValueError:
        if args.intermediate_fc_dim_vae.lower() != "half":
            logger.error(f"Valor inválido para intermediate_fc_dim_vae: {args.intermediate_fc_dim_vae}. Usar entero, '0', o 'half'. Abortando.")
            exit(1)
    
    # Validate VAE validation split ratio
    if not (0 < args.vae_val_split_ratio < 1):
        logger.info(f"vae_val_split_ratio ({args.vae_val_split_ratio}) fuera de (0,1). Deshabilitando validación VAE, early stopping y LR scheduler para VAE.")
        args.vae_val_split_ratio = 0
        args.early_stopping_patience_vae = 0 
        args.lr_scheduler_patience_vae = 0
    
    # --- Setup Seeds and Output Directory ---
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True # May impact performance
        torch.backends.cudnn.benchmark = False   # Ensure reproducibility
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # --- Log Git Hash ---
    try:
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        args.git_hash = git_hash # Store in args for saving
        logger.info(f"Git commit hash: {git_hash}")
    except Exception as e:
        logger.warning(f"No se pudo obtener el git hash: {e}")
        args.git_hash = "N/A"

    # --- Log Run Configuration ---
    logger.info("--- Configuración de la Ejecución ---")
    for arg, value in sorted(vars(args).items()):
        logger.info(f"{arg}: {value}")
    logger.info("------------------------------------")

    # --- Load Data ---
    global_tensor_data, metadata_df_full = load_data(Path(args.global_tensor_path), Path(args.metadata_path))

    if global_tensor_data is not None and metadata_df_full is not None:
        train_and_evaluate_pipeline(global_tensor_data, metadata_df_full, args)
    else:
        logger.critical("No se pudieron cargar los datos. Abortando.")

    logger.info("Pipeline completado.")
    logger.info("--- Consideraciones sobre Normalización y Activación Final del VAE (Recordatorio) ---")
    logger.info("Normalización: 'minmax_offdiag' -> [0,1] (ideal con sigmoid), 'zscore_offdiag' -> media 0, std 1 (mejor con tanh/linear).")
    logger.info("Activación Final VAE: 'sigmoid' -> [0,1], 'tanh' -> [-1,1], 'linear' -> sin restricción.")
    logger.info("Asegurar compatibilidad entre normalización y activación es clave.")

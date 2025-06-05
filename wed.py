#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_vae_classifier.py

Script para entrenar un Autoencoder Variacional (VAE) Convolucional sobre matrices de 
conectividad funcional fMRI y luego usar sus representaciones latentes para clasificar 
entre Controles Sanos (CN) y pacientes con Enfermedad de Alzheimer (AD).

Versión: 1.5.1 - Multi-Classifier Training & Arch Refinement
Cambios desde v1.5.0:
- Implementada capacidad para entrenar y evaluar múltiples tipos de clasificadores por VAE de cada fold.
- Nuevo argumento --classifier_types (lista) y --classifier_hp_tune_ratio.
- Arquitectura VAE: kernels [7,5,5,3] y padding [1,1,1,1] para encoder según especificación.
- Mejorado cálculo de output_padding para decoder 'convtranspose' con logging detallado.
- Los artefactos guardados (modelos, scalers) incluyen el tipo de clasificador en el nombre.
- Correcciones menores en la lógica de obtención de features latentes y manejo de datos.
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
import joblib # For saving sklearn models/scalers

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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler as SklearnScaler
from sklearn.metrics import (
    roc_auc_score, 
    accuracy_score, 
    recall_score, 
    precision_score, 
    f1_score, 
    average_precision_score,
    balanced_accuracy_score
)
import matplotlib.pyplot as plt # For saving plots

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
                 final_activation: str = 'tanh',
                 intermediate_fc_dim_config: Union[int, str] = "0",
                 dropout_rate: float = 0.2,
                 use_layernorm_fc: bool = False,
                 num_conv_layers_encoder: int = 3, 
                 decoder_type: str = 'upsample_conv' 
                 ):
        super(ConvolutionalVAE, self).__init__()
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.final_activation_name = final_activation
        self.dropout_rate = dropout_rate
        self.use_layernorm_fc = use_layernorm_fc
        self.num_conv_layers_encoder = num_conv_layers_encoder
        self.decoder_type = decoder_type

        if self.num_conv_layers_encoder not in [3, 4]:
            raise ValueError("num_conv_layers_encoder must be 3 or 4.")
        if self.decoder_type not in ['upsample_conv', 'convtranspose']:
            raise ValueError("decoder_type must be 'upsample_conv' or 'convtranspose'.")

        # --- Encoder ---
        encoder_conv_layers_list = []
        current_ch_enc = input_channels
        
        base_conv_channels = [max(16, input_channels*2), max(32, input_channels*4), max(64, input_channels*8), max(128, input_channels*16)]
        conv_channels_enc_all = [min(c, 256) for c in base_conv_channels]
        
        kernels_all = [7, 5, 5, 3]  # Specified by user
        paddings_all = [1, 1, 1, 1] # Specified by user
        strides_all = [2, 2, 2, 2]  # Specified by user

        self.encoder_kernels = kernels_all[:self.num_conv_layers_encoder]
        self.encoder_paddings = paddings_all[:self.num_conv_layers_encoder]
        self.encoder_strides = strides_all[:self.num_conv_layers_encoder]
        self.conv_channels_encoder = conv_channels_enc_all[:self.num_conv_layers_encoder]

        self.encoder_spatial_dims = [self.image_size]
        current_dim_calc = self.image_size

        for i in range(self.num_conv_layers_encoder):
            encoder_conv_layers_list.extend([
                nn.Conv2d(current_ch_enc, self.conv_channels_encoder[i], 
                          kernel_size=self.encoder_kernels[i], 
                          stride=self.encoder_strides[i], 
                          padding=self.encoder_paddings[i]),
                nn.ReLU(),
                nn.BatchNorm2d(self.conv_channels_encoder[i]),
                nn.Dropout2d(p=self.dropout_rate)
            ])
            current_ch_enc = self.conv_channels_encoder[i]
            current_dim_calc = ((current_dim_calc + 2 * self.encoder_paddings[i] - self.encoder_kernels[i]) // self.encoder_strides[i]) + 1
            self.encoder_spatial_dims.append(current_dim_calc)
        
        self.encoder_conv = nn.Sequential(*encoder_conv_layers_list)
        
        self.final_conv_output_channels = self.conv_channels_encoder[-1]
        self.final_spatial_dim_encoder = self.encoder_spatial_dims[-1]
        self.flattened_size_after_conv = self.final_conv_output_channels * self.final_spatial_dim_encoder * self.final_spatial_dim_encoder
        
        logger.info(f"VAE Encoder: Input Channels: {input_channels}, Num Conv Layers: {self.num_conv_layers_encoder}")
        logger.info(f"VAE Encoder: Conv Channels: {self.conv_channels_encoder}")
        logger.info(f"VAE Encoder: Kernels: {self.encoder_kernels}, Strides: {self.encoder_strides}, Paddings: {self.encoder_paddings}")
        logger.info(f"VAE Encoder: Spatial Dims (Input to Output): {self.encoder_spatial_dims}")
        logger.info(f"VAE Encoder: Calculated flattened size after conv = {self.flattened_size_after_conv} "
                    f"(final conv output channels: {self.final_conv_output_channels}, "
                    f"final spatial dim after encoder: {self.final_spatial_dim_encoder})")

        if isinstance(intermediate_fc_dim_config, str):
            if intermediate_fc_dim_config.lower() == "half":
                self.intermediate_fc_dim = self.flattened_size_after_conv // 2
            elif intermediate_fc_dim_config.lower() == "quarter":
                self.intermediate_fc_dim = self.flattened_size_after_conv // 4
            elif intermediate_fc_dim_config.lower() == "0":
                 self.intermediate_fc_dim = 0
            else: 
                try:
                    self.intermediate_fc_dim = int(intermediate_fc_dim_config)
                except ValueError:
                    logger.warning(f"Invalid string for intermediate_fc_dim_config: {intermediate_fc_dim_config}. Defaulting to 0.")
                    self.intermediate_fc_dim = 0
        elif isinstance(intermediate_fc_dim_config, int):
            self.intermediate_fc_dim = intermediate_fc_dim_config
        else:
            self.intermediate_fc_dim = 0
            
        encoder_fc_intermediate_layers = []
        if self.intermediate_fc_dim > 0:
            encoder_fc_intermediate_layers.append(nn.Linear(self.flattened_size_after_conv, self.intermediate_fc_dim))
            if self.use_layernorm_fc:
                encoder_fc_intermediate_layers.append(nn.LayerNorm(self.intermediate_fc_dim))
            encoder_fc_intermediate_layers.extend([
                nn.ReLU(),
                nn.BatchNorm1d(self.intermediate_fc_dim),
                nn.Dropout(p=self.dropout_rate)
            ])
            self.encoder_fc_intermediate = nn.Sequential(*encoder_fc_intermediate_layers)
            fc_mu_logvar_input_dim = self.intermediate_fc_dim
            logger.info(f"VAE: Intermediate FC dim (encoder): {self.intermediate_fc_dim}. Using LayerNorm: {self.use_layernorm_fc}")
        else:
            self.encoder_fc_intermediate = nn.Identity() 
            fc_mu_logvar_input_dim = self.flattened_size_after_conv
            logger.info("VAE: No intermediate FC layer in encoder.")
        
        self.fc_mu = nn.Linear(fc_mu_logvar_input_dim, latent_dim)
        self.fc_logvar = nn.Linear(fc_mu_logvar_input_dim, latent_dim)

        # --- Decoder ---
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
            self.decoder_fc_intermediate = nn.Identity()
            logger.info("VAE: No intermediate FC layer in decoder.")
            
        self.decoder_fc_to_conv = nn.Linear(decoder_fc_to_conv_input_dim, self.flattened_size_after_conv)

        decoder_conv_layers_list = []
        if self.decoder_type == 'upsample_conv':
            logger.info("Using Upsample-Conv decoder.")
            current_ch_dec = self.final_conv_output_channels
            target_decoder_conv_channels = self.conv_channels_encoder[-2::-1] + [self.input_channels]

            for i in range(self.num_conv_layers_encoder):
                target_spatial_dim = self.encoder_spatial_dims[-(i + 2)]
                decoder_conv_layers_list.append(nn.Upsample(size=target_spatial_dim, mode='bilinear', align_corners=False))
                out_ch_this_layer = target_decoder_conv_channels[i]
                decoder_conv_layers_list.extend([
                    nn.Conv2d(current_ch_dec, out_ch_this_layer, kernel_size=3, stride=1, padding=1),
                    nn.ReLU() if i < self.num_conv_layers_encoder - 1 else nn.Identity(),
                    nn.BatchNorm2d(out_ch_this_layer) if i < self.num_conv_layers_encoder - 1 else nn.Identity(),
                ])
                if i < self.num_conv_layers_encoder - 1:
                    decoder_conv_layers_list.append(nn.Dropout2d(p=self.dropout_rate))
                current_ch_dec = out_ch_this_layer

        elif self.decoder_type == 'convtranspose':
            logger.info("Using ConvTranspose decoder.")
            current_ch_dec = self.final_conv_output_channels 
            target_conv_t_channels = self.conv_channels_encoder[-2::-1] + [self.input_channels] 
            
            decoder_kernels = self.encoder_kernels[::-1]
            decoder_paddings = self.encoder_paddings[::-1]
            decoder_strides = self.encoder_strides[::-1] 
            
            output_paddings_calc = []
            current_spatial_dim_for_op_calc = self.final_spatial_dim_encoder
            logger.info(f"ConvTranspose Decoder - Initial spatial dim: {current_spatial_dim_for_op_calc}")
            logger.info(f"  Target spatial dims (from encoder, reversed): {self.encoder_spatial_dims[-2::-1] + [self.encoder_spatial_dims[0]]}")


            for i in range(self.num_conv_layers_encoder):
                k = decoder_kernels[i]
                s = decoder_strides[i] 
                p = decoder_paddings[i]
                
                target_dim_layer = self.encoder_spatial_dims[-(i + 2)] 
                
                op = target_dim_layer - ((current_spatial_dim_for_op_calc - 1) * s - 2 * p + k)
                op = max(0, min(s - 1, op)) 
                output_paddings_calc.append(op)
                
                h_out_actual_this_layer = (current_spatial_dim_for_op_calc - 1) * s - 2 * p + k + op
                logger.info(f"  Layer {self.num_conv_layers_encoder-1-i} (Decoder): K={k}, S={s}, P={p}, InDim={current_spatial_dim_for_op_calc}, TargetOut={target_dim_layer}, CalcOP={op}, ActualOut={h_out_actual_this_layer}")
                current_spatial_dim_for_op_calc = h_out_actual_this_layer
            
            output_paddings = output_paddings_calc
            logger.info(f"ConvTranspose: Calculated output_paddings: {output_paddings}")

            current_ch_dec = self.final_conv_output_channels 
            for i in range(len(target_conv_t_channels)):
                op_val = output_paddings[i]
                decoder_conv_layers_list.extend([
                    nn.ConvTranspose2d(current_ch_dec, target_conv_t_channels[i], 
                                       kernel_size=decoder_kernels[i], 
                                       stride=decoder_strides[i], 
                                       padding=decoder_paddings[i], 
                                       output_padding=op_val),
                    nn.ReLU() if i < len(target_conv_t_channels) - 1 else nn.Identity(),
                    nn.BatchNorm2d(target_conv_t_channels[i]) if i < len(target_conv_t_channels) - 1 else nn.Identity(),
                ])
                if i < len(target_conv_t_channels) -1:
                     decoder_conv_layers_list.append(nn.Dropout2d(p=self.dropout_rate))
                current_ch_dec = target_conv_t_channels[i]
        
        if self.final_activation_name == 'sigmoid':
            decoder_conv_layers_list.append(nn.Sigmoid())
            logger.info("Decoder final activation: Sigmoid")
        elif self.final_activation_name == 'tanh':
            decoder_conv_layers_list.append(nn.Tanh())
            logger.info("Decoder final activation: Tanh")
        else: 
            logger.info("Decoder final activation: Linear (None)")
            
        self.decoder_conv = nn.Sequential(*decoder_conv_layers_list)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_conv(x)
        h = h.view(h.size(0), -1) 
        h_intermediate = self.encoder_fc_intermediate(h)
        return self.fc_mu(h_intermediate), self.fc_logvar(h_intermediate)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h_latent_intermediate = self.decoder_fc_intermediate(z)
        h_fc_out = self.decoder_fc_to_conv(h_latent_intermediate)
        h_reshaped = h_fc_out.view(h_fc_out.size(0), self.final_conv_output_channels, 
                                   self.final_spatial_dim_encoder, self.final_spatial_dim_encoder)
        return self.decoder_conv(h_reshaped)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)

        if recon_x.shape != x.shape:
            if self.decoder_type == 'upsample_conv':
                 logger.warning(f"Shape mismatch with upsample_conv decoder! Input: {x.shape}, Recon: {recon_x.shape}. This should not happen if arch is correct. Interpolating.")
            else: 
                 logger.info(f"Shape mismatch with {self.decoder_type} decoder: Input {x.shape}, Recon {recon_x.shape}. Adjusting via interpolation.")
            
            recon_x = nn.functional.interpolate(recon_x, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
            
            if recon_x.shape[2] != x.shape[2] or recon_x.shape[3] != x.shape[3]:
                logger.error(f"CRITICAL: Shape mismatch even after interpolate: Input {x.shape}, Recon_interpolated {recon_x.shape}.")
        
        return recon_x, mu, logvar, z

# --- Funciones de Pérdida y Schedules ---
def vae_loss_function(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    recon_loss_mse = nn.functional.mse_loss(recon_x, x, reduction='sum') / x.shape[0] 
    kld_element = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kld = torch.mean(kld_element)
    return recon_loss_mse + beta * kld

def get_cyclical_beta_schedule(current_epoch: int, total_epochs: int, beta_max: float, n_cycles: int, ratio_increase: float = 0.5) -> float:
    if n_cycles <= 0: return beta_max
    epoch_per_cycle = total_epochs / n_cycles
    epoch_in_current_cycle = current_epoch % epoch_per_cycle
    increase_phase_duration = epoch_per_cycle * ratio_increase
    if epoch_in_current_cycle < increase_phase_duration:
        return beta_max * (epoch_in_current_cycle / increase_phase_duration)
    else:
        return beta_max

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
    merged_df = pd.merge(tensor_df, metadata_df, on='SubjectID', how='left')
    
    num_tensor_subjects = len(subject_ids_tensor)
    if len(merged_df) < num_tensor_subjects:
         logger.warning(f"Algunos SubjectIDs del tensor ({num_tensor_subjects}) no se encontraron en los metadatos. Merged: {len(merged_df)}.")
    return global_tensor, merged_df

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
    # logger.info(f"Aplicando parámetros de normalización precalculados a subconjunto de datos ({num_subjects} sujetos, {num_selected_channels} canales).")
    normalized_tensor_subset = data_tensor_subset.copy()
    off_diag_mask = ~np.eye(num_rois, dtype=bool)

    if len(norm_params_per_channel_list) != num_selected_channels:
        raise ValueError(f"Mismatch in number of channels for normalization: data has {num_selected_channels}, params provided for {len(norm_params_per_channel_list)}")

    for c_idx_selected in range(num_selected_channels):
        params = norm_params_per_channel_list[c_idx_selected]
        mode = params.get('mode', 'zscore_offdiag') 
        # channel_name = params.get('original_name', f"Channel_{c_idx_selected}")
        if params.get('no_scale', False):
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
                                metadata_df_full: pd.DataFrame,
                                args: argparse.Namespace):
    
    output_base_dir = Path(args.output_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    selected_channel_indices: List[int] = []
    selected_channel_names_in_tensor: List[str] = []
    if args.channels_to_use:
        for ch_specifier in args.channels_to_use:
            try:
                ch_idx = int(ch_specifier)
                if 0 <= ch_idx < global_tensor_all_channels.shape[1]:
                    selected_channel_indices.append(ch_idx)
                else:
                    logger.warning(f"Índice de canal '{ch_idx}' fuera de rango. Ignorando.")
            except ValueError:
                if ch_specifier in DEFAULT_CHANNEL_NAMES:
                    try:
                        original_idx = DEFAULT_CHANNEL_NAMES.index(ch_specifier)
                        if original_idx < global_tensor_all_channels.shape[1]:
                            selected_channel_indices.append(original_idx)
                        else:
                             logger.warning(f"Nombre de canal '{ch_specifier}' (índice {original_idx}) fuera de rango. Ignorando.")
                    except ValueError: pass 
                else:
                    logger.warning(f"Nombre de canal '{ch_specifier}' desconocido. Ignorando.")
        
        if not selected_channel_indices:
            logger.error("No se seleccionaron canales válidos. Abortando.")
            return
        selected_channel_indices = sorted(list(set(selected_channel_indices)))
        temp_names = []
        for idx in selected_channel_indices:
            if idx < len(DEFAULT_CHANNEL_NAMES): temp_names.append(DEFAULT_CHANNEL_NAMES[idx])
            else: temp_names.append(f"RawChan{idx}")
        selected_channel_names_in_tensor = temp_names
        current_global_tensor = global_tensor_all_channels[:, selected_channel_indices, :, :]
        logger.info(f"Usando canales seleccionados (índices): {selected_channel_indices}")
        logger.info(f"Nombres de canales seleccionados: {selected_channel_names_in_tensor}")
    else:
        current_global_tensor = global_tensor_all_channels
        selected_channel_names_in_tensor = [DEFAULT_CHANNEL_NAMES[i] if i < len(DEFAULT_CHANNEL_NAMES) else f"RawChan{i}" for i in range(current_global_tensor.shape[1])]
        logger.info(f"Usando todos los {current_global_tensor.shape[1]} canales.")
    
    num_input_channels_for_vae = current_global_tensor.shape[1]

    if 'ResearchGroup' not in metadata_df_full.columns:
        logger.error("'ResearchGroup' no encontrado. Abortando.")
        return
    cn_ad_df = metadata_df_full[metadata_df_full['ResearchGroup'].isin(['CN', 'AD'])].copy()
    if cn_ad_df.empty or 'tensor_idx' not in cn_ad_df.columns:
        logger.error("No hay sujetos CN/AD o falta 'tensor_idx'. Abortando.")
        return
    cn_ad_df['label'] = cn_ad_df['ResearchGroup'].map({'CN': 0, 'AD': 1})
    
    strat_cols = ['ResearchGroup']
    if args.classifier_stratify_cols:
        for col in args.classifier_stratify_cols:
            if col in cn_ad_df.columns:
                cn_ad_df[col] = cn_ad_df[col].fillna(f"{col}_Unknown").astype(str)
                strat_cols.append(col)
            else: logger.warning(f"Columna de estratificación '{col}' no encontrada.")
    if len(strat_cols) > 1:
        cn_ad_df['stratify_key'] = cn_ad_df[strat_cols].apply(lambda x: '_'.join(x.astype(str)), axis=1)
    else:
        cn_ad_df['stratify_key'] = cn_ad_df['label'].astype(str)
    logger.info(f"Estratificando folds del clasificador por: {strat_cols}")

    X_classifier_indices_global = cn_ad_df['tensor_idx'].values
    y_classifier_labels_cn_ad = cn_ad_df['label'].values
    stratify_labels_for_cv = cn_ad_df['stratify_key']
    logger.info(f"Sujetos CN/AD para clasificación: {len(cn_ad_df)}. CN: {sum(y_classifier_labels_cn_ad == 0)}, AD: {sum(y_classifier_labels_cn_ad == 1)}")

    if args.repeated_outer_folds_n_repeats > 1:
        outer_cv_clf = RepeatedStratifiedKFold(n_splits=args.outer_folds, n_repeats=args.repeated_outer_folds_n_repeats, random_state=args.seed)
        total_outer_iterations = args.outer_folds * args.repeated_outer_folds_n_repeats
    else:
        outer_cv_clf = StratifiedKFold(n_splits=args.outer_folds, shuffle=True, random_state=args.seed)
        total_outer_iterations = args.outer_folds
    logger.info(f"Usando CV externa: {type(outer_cv_clf).__name__} con {total_outer_iterations} iteraciones totales.")

    all_folds_metrics = []
    all_folds_vae_history = []
    all_folds_clf_predictions = []

    for fold_idx, (train_dev_clf_local_indices, test_clf_local_indices) in enumerate(outer_cv_clf.split(np.zeros(len(cn_ad_df)), stratify_labels_for_cv)):
        fold_start_time = time.time()
        logger.info(f"--- Iniciando Fold Externo {fold_idx + 1}/{total_outer_iterations} ---")
        
        fold_output_dir = output_base_dir / f"fold_{fold_idx + 1}"
        fold_output_dir.mkdir(parents=True, exist_ok=True)

        global_indices_clf_test = X_classifier_indices_global[test_clf_local_indices]
        
        all_available_subject_indices = metadata_df_full['tensor_idx'].values
        valid_max_idx = current_global_tensor.shape[0] -1
        all_available_subject_indices = all_available_subject_indices[all_available_subject_indices <= valid_max_idx]
        global_indices_vae_training_pool = np.setdiff1d(all_available_subject_indices, global_indices_clf_test, assume_unique=True)
        
        if len(global_indices_vae_training_pool) < 10: 
            logger.error(f"Fold {fold_idx+1}: Muy pocos sujetos ({len(global_indices_vae_training_pool)}) para entrenamiento VAE. Saltando fold.")
            continue

        vae_train_pool_tensor_original_scale = current_global_tensor[global_indices_vae_training_pool]
        
        vae_actual_train_indices_local_to_pool, vae_internal_val_indices_local_to_pool = [], []
        if args.vae_val_split_ratio > 0 and len(global_indices_vae_training_pool) > 10:
            try:
                vae_actual_train_indices_local_to_pool, vae_internal_val_indices_local_to_pool = sk_train_test_split(
                    np.arange(len(global_indices_vae_training_pool)), test_size=args.vae_val_split_ratio,
                    random_state=args.seed + fold_idx + 10, shuffle=True # Different seed for this split
                )
            except ValueError as e:
                logger.warning(f"No se pudo hacer split de validación VAE: {e}. Usando todos para VAE train.")
                vae_actual_train_indices_local_to_pool = np.arange(len(global_indices_vae_training_pool))
        else:
            vae_actual_train_indices_local_to_pool = np.arange(len(global_indices_vae_training_pool))
        
        logger.info(f"Sujetos VAE actual train (del pool): {len(vae_actual_train_indices_local_to_pool)}, VAE internal val (del pool): {len(vae_internal_val_indices_local_to_pool)}")

        vae_pool_tensor_norm, norm_params_fold_list = normalize_inter_channel_fold(
            vae_train_pool_tensor_original_scale, vae_actual_train_indices_local_to_pool, # Normalize using VAE's own training part of the pool
            mode=args.norm_mode, selected_channel_original_names=selected_channel_names_in_tensor
        )

        vae_train_dataset = TensorDataset(torch.from_numpy(vae_pool_tensor_norm[vae_actual_train_indices_local_to_pool]).float())
        vae_train_loader = DataLoader(vae_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        
        vae_internal_val_loader = None
        if len(vae_internal_val_indices_local_to_pool) > 0:
            vae_internal_val_dataset = TensorDataset(torch.from_numpy(vae_pool_tensor_norm[vae_internal_val_indices_local_to_pool]).float())
            vae_internal_val_loader = DataLoader(vae_internal_val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Usando dispositivo: {device}")
        
        vae_fold_k = ConvolutionalVAE(
            input_channels=num_input_channels_for_vae, 
            latent_dim=args.latent_dim, 
            image_size=current_global_tensor.shape[-1],
            final_activation=args.vae_final_activation,
            intermediate_fc_dim_config=args.intermediate_fc_dim_vae,
            dropout_rate=args.dropout_rate_vae,
            use_layernorm_fc=args.use_layernorm_vae_fc,
            num_conv_layers_encoder=args.num_conv_layers_encoder,
            decoder_type=args.decoder_type
        ).to(device)
        optimizer_vae = optim.Adam(vae_fold_k.parameters(), lr=args.lr_vae, weight_decay=args.weight_decay_vae)
        scheduler_vae = None
        if vae_internal_val_loader and args.lr_scheduler_patience_vae > 0:
            scheduler_vae = optim.lr_scheduler.ReduceLROnPlateau(optimizer_vae, 'min', patience=args.lr_scheduler_patience_vae, factor=0.1, verbose=False)

        logger.info(f"Entrenando VAE para el fold {fold_idx + 1} (Decoder: {args.decoder_type}, Encoder Layers: {args.num_conv_layers_encoder})...")
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_state_dict = None
        
        fold_vae_train_losses, fold_vae_val_losses, fold_vae_betas = [], [], []

        for epoch in range(args.epochs_vae):
            vae_fold_k.train()
            epoch_train_loss = 0.0
            current_beta = get_cyclical_beta_schedule(epoch, args.epochs_vae, args.beta_vae, args.cyclical_beta_n_cycles, args.cyclical_beta_ratio_increase)
            
            for (data,) in vae_train_loader:
                data = data.to(device)
                optimizer_vae.zero_grad()
                recon_batch, mu, logvar, _ = vae_fold_k(data)
                loss = vae_loss_function(recon_batch, data, mu, logvar, beta=current_beta)
                loss.backward()
                optimizer_vae.step()
                epoch_train_loss += loss.item() * data.size(0)
            
            avg_epoch_train_loss = epoch_train_loss / len(vae_train_loader.dataset)
            fold_vae_train_losses.append(avg_epoch_train_loss)
            fold_vae_betas.append(current_beta)
            
            log_msg = f"F{fold_idx+1} VAE E{epoch+1}/{args.epochs_vae}, TrL: {avg_epoch_train_loss:.4f}, Beta: {current_beta:.4f}, LR: {optimizer_vae.param_groups[0]['lr']:.2e}"

            if vae_internal_val_loader:
                vae_fold_k.eval()
                epoch_val_loss = 0.0
                with torch.no_grad():
                    for (val_data,) in vae_internal_val_loader:
                        val_data = val_data.to(device)
                        recon_val, mu_val, logvar_val, _ = vae_fold_k(val_data)
                        val_loss_batch = vae_loss_function(recon_val, val_data, mu_val, logvar_val, beta=current_beta)
                        epoch_val_loss += val_loss_batch.item() * val_data.size(0)
                avg_epoch_val_loss = epoch_val_loss / len(vae_internal_val_loader.dataset)
                fold_vae_val_losses.append(avg_epoch_val_loss)
                log_msg += f", VAE_ValL: {avg_epoch_val_loss:.4f}" # Clarify this is VAE's internal val loss

                if scheduler_vae: scheduler_vae.step(avg_epoch_val_loss)
                if avg_epoch_val_loss < best_val_loss:
                    best_val_loss = avg_epoch_val_loss
                    epochs_no_improve = 0
                    best_model_state_dict = copy.deepcopy(vae_fold_k.state_dict())
                else:
                    epochs_no_improve += 1
                if args.early_stopping_patience_vae > 0 and epochs_no_improve >= args.early_stopping_patience_vae:
                    logger.info(f"Early stopping VAE en epoch {epoch+1}. Mejor val_loss VAE: {best_val_loss:.4f}")
                    break
            else: 
                 fold_vae_val_losses.append(np.nan) 
                 best_model_state_dict = copy.deepcopy(vae_fold_k.state_dict()) 

            if (epoch + 1) % args.log_interval_epochs_vae == 0 or epoch == args.epochs_vae - 1:
                logger.info(log_msg)
        
        if best_model_state_dict:
            vae_fold_k.load_state_dict(best_model_state_dict)
            val_string = f"{best_val_loss:.4f}" if vae_internal_val_loader and not np.isnan(best_val_loss) else "N/A - Last Epoch"
            logger.info(f"Fold {fold_idx+1}: VAE final model loaded (best VAE val_loss: {val_string}).")

        vae_model_fname = f"vae_model_fold_{fold_idx+1}.pt"
        torch.save(vae_fold_k.state_dict(), fold_output_dir / vae_model_fname)
        logger.info(f"Modelo VAE del fold {fold_idx+1} guardado en: {fold_output_dir / vae_model_fname}")
        
        if args.save_vae_training_history:
            history_data = {"train_loss": fold_vae_train_losses, "val_loss": fold_vae_val_losses, "beta": fold_vae_betas}
            joblib.dump(history_data, fold_output_dir / f"vae_train_history_fold_{fold_idx+1}.joblib")
            try:
                fig, ax1 = plt.subplots(figsize=(10,5))
                ax1.plot(history_data["train_loss"], label="Train Loss", color='blue')
                if vae_internal_val_loader and any(not np.isnan(x) for x in history_data["val_loss"]):
                    ax1.plot(history_data["val_loss"], label="VAE Val Loss", color='orange')
                ax1.set_xlabel("Epoch")
                ax1.set_ylabel("Loss")
                ax1.legend(loc='upper left')
                ax2 = ax1.twinx()
                ax2.plot(history_data["beta"], label="Beta", color='green', linestyle='--')
                ax2.set_ylabel("Beta")
                ax2.legend(loc='upper right')
                plt.title(f"Fold {fold_idx+1} VAE Training History")
                plt.savefig(fold_output_dir / f"vae_train_history_fold_{fold_idx+1}.png")
                plt.close(fig)
            except Exception as e:
                logger.warning(f"No se pudo guardar la gráfica de historial VAE para el fold {fold_idx+1}: {e}")
        all_folds_vae_history.append(history_data if args.save_vae_training_history else None)

        # --- Classifier Training & Tuning Stage ---
        # Data for classifier (CN/AD only, from outer fold's train/dev split)
        global_indices_clf_train_dev_all = X_classifier_indices_global[train_dev_clf_local_indices]
        y_clf_train_dev_all = y_classifier_labels_cn_ad[train_dev_clf_local_indices]
        
        # Split this into a HP tuning set and a primary training set for the classifier
        # Ensure this split uses only CN/AD subjects
        if len(global_indices_clf_train_dev_all) < 10: # Min samples for split
            logger.warning(f"Fold {fold_idx+1}: Muy pocos sujetos CN/AD ({len(global_indices_clf_train_dev_all)}) en el pool de train/dev del clasificador. Saltando etapa de clasificador.")
            continue
            
        indices_for_split = np.arange(len(global_indices_clf_train_dev_all))
        # Stratified split for classifier HP tuning set
        clf_primary_train_local_indices, clf_hp_tune_local_indices = sk_train_test_split(
            indices_for_split,
            test_size=args.classifier_hp_tune_ratio,
            stratify=y_clf_train_dev_all, # Stratify on CN/AD labels
            random_state=args.seed + fold_idx + 20 # Different seed
        )
        
        global_indices_clf_primary_train = global_indices_clf_train_dev_all[clf_primary_train_local_indices]
        y_clf_primary_train = y_clf_train_dev_all[clf_primary_train_local_indices]
        
        global_indices_clf_hp_tune = global_indices_clf_train_dev_all[clf_hp_tune_local_indices]
        y_clf_hp_tune = y_clf_train_dev_all[clf_hp_tune_local_indices]

        # Get latent features for these sets using the trained VAE for this fold
        vae_fold_k.eval()
        with torch.no_grad():
            # HP Tune Set
            X_hp_tune_tensor_norm = apply_normalization_params(current_global_tensor[global_indices_clf_hp_tune], norm_params_fold_list)
            _, mu_hp_tune, _, z_hp_tune = vae_fold_k(torch.from_numpy(X_hp_tune_tensor_norm).float().to(device))
            X_latent_hp_tune = mu_hp_tune.cpu().numpy() if args.latent_features_type == 'mu' else z_hp_tune.cpu().numpy()

            # Primary Train Set
            X_primary_train_tensor_norm = apply_normalization_params(current_global_tensor[global_indices_clf_primary_train], norm_params_fold_list)
            _, mu_primary_train, _, z_primary_train = vae_fold_k(torch.from_numpy(X_primary_train_tensor_norm).float().to(device))
            X_latent_primary_train = mu_primary_train.cpu().numpy() if args.latent_features_type == 'mu' else z_primary_train.cpu().numpy()

            # Final Test Set for this outer fold
            X_test_final_tensor_norm = apply_normalization_params(current_global_tensor[global_indices_clf_test], norm_params_fold_list)
            if X_test_final_tensor_norm.shape[0] > 0:
                _, mu_test_final, _, z_test_final = vae_fold_k(torch.from_numpy(X_test_final_tensor_norm).float().to(device))
                X_latent_test_final = mu_test_final.cpu().numpy() if args.latent_features_type == 'mu' else z_test_final.cpu().numpy()
            else:
                X_latent_test_final = np.array([])
            y_test_final = y_classifier_labels_cn_ad[test_clf_local_indices]

        if X_latent_hp_tune.shape[0] == 0 or X_latent_primary_train.shape[0] == 0:
            logger.warning(f"Fold {fold_idx+1}: Datos insuficientes para HP tuning o entrenamiento primario del clasificador. Skipping.")
            continue
            
        # Loop over specified classifier types
        for current_classifier_type in args.classifier_types:
            logger.info(f"  --- Entrenando Clasificador: {current_classifier_type} ---")
            
            clf_model, param_grid = None, {}
            clf_class_weight = 'balanced' if args.classifier_use_class_weight else None

            if current_classifier_type == 'svm':
                clf_model = SVC(probability=True, random_state=args.seed, class_weight=clf_class_weight)
                param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'gamma': [0.0001, 0.001, 0.01, 0.1, 'scale', 'auto'], 'kernel': ['rbf', 'linear']}
            elif current_classifier_type == 'logreg':
                clf_model = LogisticRegression(random_state=args.seed, class_weight=clf_class_weight, solver='liblinear', max_iter=2000) # Increased max_iter
                param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
            elif current_classifier_type == 'mlp':
                hidden_layers = tuple(map(int, args.mlp_classifier_hidden_layers.split(',')))
                clf_model = MLPClassifier(random_state=args.seed, hidden_layer_sizes=hidden_layers, max_iter=750, early_stopping=True, n_iter_no_change=20) # Increased patience/iter
                param_grid = {'alpha': [0.0001, 0.001, 0.01, 0.1], 'learning_rate_init': [0.0005, 0.001, 0.005]}
            elif current_classifier_type == 'rf':
                clf_model = RandomForestClassifier(random_state=args.seed, class_weight=clf_class_weight)
                param_grid = {'n_estimators': [50, 100, 200, 300], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4, 8]}
            elif current_classifier_type == 'gb':
                clf_model = GradientBoostingClassifier(random_state=args.seed)
                param_grid = {'n_estimators': [50, 100, 200, 300], 'learning_rate': [0.01, 0.05, 0.1, 0.2], 'max_depth': [3, 5, 7]}
            else:
                logger.warning(f"Tipo de clasificador no soportado: {current_classifier_type}. Saltando.")
                continue
            
            # Scale HP tune set
            scaler_hp_tune = SklearnScaler()
            X_latent_hp_tune_scaled = scaler_hp_tune.fit_transform(X_latent_hp_tune)

            inner_cv_for_hp_tune = StratifiedKFold(n_splits=args.inner_folds, shuffle=True, random_state=args.seed + fold_idx + 30)
            grid_search = GridSearchCV(clf_model, param_grid, cv=inner_cv_for_hp_tune, scoring=args.gridsearch_scoring, n_jobs=args.n_jobs_gridsearch, verbose=0)
            
            logger.info(f"    Ajustando hiperparámetros para {current_classifier_type} en {X_latent_hp_tune_scaled.shape[0]} muestras...")
            grid_search.fit(X_latent_hp_tune_scaled, y_clf_hp_tune)
            best_params_clf = grid_search.best_params_
            logger.info(f"    Mejores hiperparámetros para {current_classifier_type}: {best_params_clf}")

            # Combine primary train and HP tune sets for final model training
            X_latent_combined_train = np.concatenate((X_latent_primary_train, X_latent_hp_tune), axis=0)
            y_combined_train = np.concatenate((y_clf_primary_train, y_clf_hp_tune), axis=0)
            
            # Fit a new scaler on the combined training data and transform it
            scaler_final_clf = SklearnScaler()
            X_latent_combined_train_scaled = scaler_final_clf.fit_transform(X_latent_combined_train)
            
            # Train final classifier with best params on combined scaled data
            final_clf_model = clf_model.set_params(**best_params_clf) # Create new instance with best params
            logger.info(f"    Entrenando {current_classifier_type} final en {X_latent_combined_train_scaled.shape[0]} muestras...")
            final_clf_model.fit(X_latent_combined_train_scaled, y_combined_train)

            fold_results_clf = {'fold': fold_idx + 1, 
                                'actual_classifier_type': current_classifier_type, 
                                'best_clf_params': best_params_clf,
                                'num_selected_channels': num_input_channels_for_vae,
                                'selected_channel_names': ";".join(selected_channel_names_in_tensor)}

            if X_latent_test_final.shape[0] > 0:
                X_latent_test_final_scaled = scaler_final_clf.transform(X_latent_test_final)
                y_pred_proba = final_clf_model.predict_proba(X_latent_test_final_scaled)[:, 1]
                y_pred = final_clf_model.predict(X_latent_test_final_scaled)
                
                fold_results_clf.update({
                    'auc': roc_auc_score(y_test_final, y_pred_proba),
                    'pr_auc': average_precision_score(y_test_final, y_pred_proba),
                    'accuracy': accuracy_score(y_test_final, y_pred),
                    'balanced_accuracy': balanced_accuracy_score(y_test_final, y_pred),
                    'sensitivity': recall_score(y_test_final, y_pred, pos_label=1, zero_division=0),
                    'specificity': recall_score(y_test_final, y_pred, pos_label=0, zero_division=0),
                    'f1_score': f1_score(y_test_final, y_pred, pos_label=1, zero_division=0)
                })
                all_folds_clf_predictions.append({'fold': fold_idx + 1, 'classifier_type': current_classifier_type, 
                                                 'y_true': y_test_final, 'y_pred_proba': y_pred_proba, 'y_pred': y_pred})
            else:
                metrics_to_nan = ['auc', 'pr_auc', 'accuracy', 'balanced_accuracy', 'sensitivity', 'specificity', 'f1_score']
                for m in metrics_to_nan: fold_results_clf[m] = np.nan

            logger.info(f"    Resultados Fold {fold_idx+1} ({current_classifier_type}): AUC={fold_results_clf.get('auc',0):.4f}, Bal.Acc={fold_results_clf.get('balanced_accuracy',0):.4f}, F1={fold_results_clf.get('f1_score',0):.4f}")
            
            if args.save_fold_artefacts:
                joblib.dump(scaler_final_clf, fold_output_dir / f"scaler_latent_{current_classifier_type}_fold_{fold_idx+1}.joblib")
                joblib.dump(final_clf_model, fold_output_dir / f"classifier_{current_classifier_type}_model_fold_{fold_idx+1}.joblib")
                logger.info(f"    Scaler latente y modelo {current_classifier_type} del fold {fold_idx+1} guardados.")
            
            all_folds_metrics.append(fold_results_clf)
            del clf_model, grid_search, best_params_clf, final_clf_model, scaler_hp_tune, scaler_final_clf
            # del X_latent_hp_tune, y_clf_hp_tune, X_latent_primary_train, y_clf_primary_train # these are subsets of larger arrays, already handled

        del vae_fold_k, optimizer_vae, vae_train_loader, vae_internal_val_loader, scheduler_vae, best_model_state_dict
        # del X_latent_hp_tune, X_latent_primary_train, X_latent_test_final # These are numpy arrays from .cpu().numpy()
        # Ensure all large tensors and arrays from this fold are candidates for GC
        if 'mu_hp_tune' in locals(): del mu_hp_tune, z_hp_tune
        if 'mu_primary_train' in locals(): del mu_primary_train, z_primary_train
        if 'mu_test_final' in locals(): del mu_test_final, z_test_final
        gc.collect()
        if device.type == 'cuda': torch.cuda.empty_cache()
        logger.info(f"Fold {fold_idx + 1} completado en {time.time() - fold_start_time:.2f} segundos.")

    if all_folds_metrics:
        metrics_df = pd.DataFrame(all_folds_metrics)
        
        # Log summary per classifier type
        for clf_type_iterated in args.classifier_types:
            metrics_df_clf = metrics_df[metrics_df['actual_classifier_type'] == clf_type_iterated]
            if not metrics_df_clf.empty:
                logger.info(f"\n--- Resumen de Rendimiento para Clasificador: {clf_type_iterated} (Promedio sobre Folds Externos) ---")
                for metric in ['auc', 'pr_auc', 'accuracy', 'balanced_accuracy', 'sensitivity', 'specificity', 'f1_score']:
                    if metric in metrics_df_clf.columns and metrics_df_clf[metric].notna().any():
                        mean_val = metrics_df_clf[metric].mean()
                        std_val = metrics_df_clf[metric].std()
                        logger.info(f"{metric.capitalize():<20}: {mean_val:.4f} +/- {std_val:.4f}")
        
        main_clf_type_for_fname = args.classifier_types[0] if args.classifier_types else "genericclf"
        fname_suffix = (f"{main_clf_type_for_fname}_vae{args.decoder_type}{args.num_conv_layers_encoder}l_"
                        f"ld{args.latent_dim}_beta{args.beta_vae}_norm{args.norm_mode}_"
                        f"ch{num_input_channels_for_vae}{'sel' if args.channels_to_use else 'all'}_"
                        f"intFC{args.intermediate_fc_dim_vae}_drop{args.dropout_rate_vae}_"
                        f"ln{1 if args.use_layernorm_vae_fc else 0}_outer{args.outer_folds}x{args.repeated_outer_folds_n_repeats if args.repeated_outer_folds_n_repeats > 1 else 1}_"
                        f"score{args.gridsearch_scoring}")
        
        results_csv_path = output_base_dir / f"all_folds_metrics_MULTI_{fname_suffix}.csv"
        metrics_df.to_csv(results_csv_path, index=False)
        logger.info(f"Resultados detallados de todos los clasificadores guardados en: {results_csv_path}")

        summary_txt_path = output_base_dir / f"summary_metrics_MULTI_{fname_suffix}.txt"
        with open(summary_txt_path, 'w') as f:
            f.write(f"Run Arguments:\n{vars(args)}\n\n")
            f.write(f"Git Commit Hash: {args.git_hash}\n\n")
            for clf_type_iterated in args.classifier_types:
                metrics_df_clf = metrics_df[metrics_df['actual_classifier_type'] == clf_type_iterated]
                if not metrics_df_clf.empty:
                    f.write(f"--- Metrics Summary for Classifier: {clf_type_iterated} ---\n")
                    for metric in ['auc', 'pr_auc', 'accuracy', 'balanced_accuracy', 'sensitivity', 'specificity', 'f1_score']:
                        if metric in metrics_df_clf.columns and metrics_df_clf[metric].notna().any():
                            f.write(f"{metric.capitalize():<20}: {metrics_df_clf[metric].mean():.4f} +/- {metrics_df_clf[metric].std():.4f}\n")
                    f.write("\nFull Metrics DataFrame Description:\n")
                    f.write(metrics_df_clf.describe().to_string())
                    f.write("\n\n")
        logger.info(f"Sumario estadístico de métricas (por clasificador) guardado en: {summary_txt_path}")

        if args.save_vae_training_history and all_folds_vae_history:
             joblib.dump(all_folds_vae_history, output_base_dir / f"all_folds_vae_training_history_{fname_suffix}.joblib")
        if all_folds_clf_predictions: 
             joblib.dump(all_folds_clf_predictions, output_base_dir / f"all_folds_clf_predictions_MULTI_{fname_suffix}.joblib")
    else:
        logger.warning("No se pudieron calcular métricas para ningún fold.")

# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de Entrenamiento VAE y Clasificador para AD vs CN (v1.5.1)",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    group_data = parser.add_argument_group('Data and Paths')
    group_data.add_argument("--global_tensor_path", type=str, required=True, help="Ruta al archivo .npz del tensor global.")
    group_data.add_argument("--metadata_path", type=str, required=True, help="Ruta al archivo CSV de metadatos.")
    group_data.add_argument("--output_dir", type=str, default="./vae_clf_output_v1.5.1", help="Directorio para guardar resultados.")
    group_data.add_argument("--channels_to_use", type=str, nargs='*', default=None, help="Lista de nombres o índices de canales a usar.")

    group_cv = parser.add_argument_group('Cross-validation')
    group_cv.add_argument("--outer_folds", type=int, default=5, help="Número de folds para CV externa del clasificador.")
    group_cv.add_argument("--repeated_outer_folds_n_repeats", type=int, default=1, help="Número de repeticiones para RepeatedStratifiedKFold.")
    group_cv.add_argument("--inner_folds", type=int, default=5, help="Folds para CV interna (GridSearchCV).") 
    group_cv.add_argument("--classifier_stratify_cols", type=str, nargs='*', default=['Sex'], help="Columnas adicionales para estratificación del clasificador.")
    group_cv.add_argument("--classifier_hp_tune_ratio", type=float, default=0.25, help="Proporción de datos de train/dev del clasificador para ajuste de HP (el resto para entreno primario).")


    group_vae = parser.add_argument_group('VAE Model and Training')
    group_vae.add_argument("--num_conv_layers_encoder", type=int, default=4, choices=[3, 4], help="Capas convolucionales en encoder VAE. (Default: 4)") # Default changed
    group_vae.add_argument("--decoder_type", type=str, default="convtranspose", choices=["upsample_conv", "convtranspose"], help="Tipo de decoder para VAE. (Default: convtranspose)") # Default changed
    group_vae.add_argument("--latent_dim", type=int, default=256, help="Dimensión del espacio latente VAE. (Default: 256)")
    group_vae.add_argument("--lr_vae", type=float, default=1e-4, help="Tasa de aprendizaje VAE.")
    group_vae.add_argument("--epochs_vae", type=int, default=300, help="Épocas máximas para VAE. (Default: 300)")
    group_vae.add_argument("--batch_size", type=int, default=32, help="Tamaño del batch.")
    group_vae.add_argument("--beta_vae", type=float, default=1.0, help="Peso KLD (beta_max para annealing).")
    group_vae.add_argument("--cyclical_beta_n_cycles", type=int, default=5, help="Ciclos para annealing de beta. (Default: 5)")
    group_vae.add_argument("--cyclical_beta_ratio_increase", type=float, default=0.5, help="Proporción de ciclo para aumentar beta.")
    group_vae.add_argument("--weight_decay_vae", type=float, default=1e-5, help="Decaimiento de peso (L2 reg) para VAE.")
    group_vae.add_argument("--vae_final_activation", type=str, default="tanh", choices=["sigmoid", "tanh", "linear"], help="Activación final del decoder VAE.")
    group_vae.add_argument("--intermediate_fc_dim_vae", type=str, default="quarter", help="Dimensión FC intermedia en VAE (entero, '0', 'half', 'quarter').")
    group_vae.add_argument("--dropout_rate_vae", type=float, default=0.2, help="Tasa de dropout en VAE. (Default: 0.2)") # Adjusted
    group_vae.add_argument("--use_layernorm_vae_fc", action='store_true', help="Usar LayerNorm en capas FC del VAE.")
    group_vae.add_argument("--vae_val_split_ratio", type=float, default=0.2, help="Proporción para validación VAE. (Default: 0.2)")
    group_vae.add_argument("--early_stopping_patience_vae", type=int, default=30, help="Paciencia early stopping VAE. (Default: 30)")
    group_vae.add_argument("--lr_scheduler_patience_vae", type=int, default=20, help="Paciencia scheduler LR VAE. (Default: 20)")

    group_clf = parser.add_argument_group('Classifier')
    group_clf.add_argument("--classifier_types", type=str, nargs='+', default=["svm"], choices=["svm", "logreg", "mlp", "rf", "gb"], help="Tipos de clasificadores a entrenar (lista). (Default: svm)")
    group_clf.add_argument("--latent_features_type", type=str, default="mu", choices=["mu", "z"], help="Usar 'mu' o 'z' como features latentes.")
    group_clf.add_argument("--gridsearch_scoring", type=str, default="balanced_accuracy", choices=["roc_auc", "accuracy", "balanced_accuracy", "f1", "average_precision"], help="Métrica para GridSearchCV.")
    group_clf.add_argument("--classifier_use_class_weight", action='store_true', help="Usar class_weight='balanced'.")
    group_clf.add_argument("--mlp_classifier_hidden_layers", type=str, default="128,64", help="Capas ocultas MLP.")

    group_general = parser.add_argument_group('General and Saving Settings')
    group_general.add_argument("--norm_mode", type=str, default="zscore_offdiag", choices=["zscore_offdiag", "minmax_offdiag"], help="Modo de normalización inter-canal.")
    group_general.add_argument("--seed", type=int, default=42, help="Semilla para reproducibilidad.")
    group_general.add_argument("--num_workers", type=int, default=2, help="Workers para DataLoader.")
    group_general.add_argument("--n_jobs_gridsearch", type=int, default=1, help="Jobs para GridSearchCV.")
    group_general.add_argument("--log_interval_epochs_vae", type=int, default=5, help="Intervalo de épocas para loguear VAE.")
    group_general.add_argument("--save_fold_artefacts", action='store_true', help="Guardar scaler latente y clasificador de cada fold.")
    group_general.add_argument("--save_vae_training_history", action='store_true', help="Guardar historial de entrenamiento del VAE (loss, beta) por fold.")
    
    args = parser.parse_args()

    if isinstance(args.intermediate_fc_dim_vae, str) and args.intermediate_fc_dim_vae.lower() not in ["0", "half", "quarter"]:
        try:
            args.intermediate_fc_dim_vae = int(args.intermediate_fc_dim_vae)
        except ValueError:
            logger.error(f"Valor inválido para intermediate_fc_dim_vae: {args.intermediate_fc_dim_vae}. Usar entero, '0', 'half', o 'quarter'. Abortando.")
            exit(1)
    
    if not (0 <= args.vae_val_split_ratio < 1): 
        if args.vae_val_split_ratio != 0:
            logger.warning(f"vae_val_split_ratio ({args.vae_val_split_ratio}) inválido. Se usará 0 (sin validación VAE).")
        args.vae_val_split_ratio = 0
    
    if args.vae_val_split_ratio == 0:
        logger.info("Sin validación VAE, early stopping y LR scheduler para VAE deshabilitados/no aplicables.")
        args.early_stopping_patience_vae = 0 
        args.lr_scheduler_patience_vae = 0
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    try:
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        args.git_hash = git_hash
    except Exception:
        args.git_hash = "N/A"
    logger.info(f"Git commit hash: {args.git_hash}")

    logger.info("--- Configuración de la Ejecución (v1.5.1) ---") # Updated version
    for arg, value in sorted(vars(args).items()):
        logger.info(f"{arg}: {value}")
    logger.info("------------------------------------")

    global_tensor_data, metadata_df_full = load_data(Path(args.global_tensor_path), Path(args.metadata_path))

    if global_tensor_data is not None and metadata_df_full is not None:
        pipeline_start_time = time.time()
        train_and_evaluate_pipeline(global_tensor_data, metadata_df_full, args)
        logger.info(f"Pipeline completo en {time.time() - pipeline_start_time:.2f} segundos.")
    else:
        logger.critical("No se pudieron cargar los datos. Abortando.")

    logger.info("--- Consideraciones sobre Normalización y Activación Final del VAE (Recordatorio) ---")
    logger.info("Normalización: 'minmax_offdiag' -> [0,1] (ideal con sigmoid), 'zscore_offdiag' -> media 0, std 1 (mejor con tanh/linear).")
    logger.info("Activación Final VAE: 'sigmoid' -> [0,1], 'tanh' -> [-1,1], 'linear' -> sin restricción.")
    logger.info(f"Configuración actual: norm_mode='{args.norm_mode}', vae_final_activation='{args.vae_final_activation}'. Asegurar compatibilidad.")

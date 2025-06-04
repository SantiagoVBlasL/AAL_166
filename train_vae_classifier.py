#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_vae_classifier.py

Script para entrenar un Autoencoder Variacional (VAE) Convolucional sobre matrices de
conectividad funcional fMRI y luego usar sus representaciones latentes para clasificar
entre Controles Sanos (CN) y pacientes con Enfermedad de Alzheimer (AD).

Versión: 1.1.1 - DecoderOutputFix: Corrige el tamaño de la salida del decoder VAE.

Cambios desde v1.1.0:
- Añadida lógica en el método forward del VAE para recortar/ajustar la salida del decoder
  (recon_x) para que coincida exactamente con las dimensiones espaciales de la entrada (x)
  antes de calcular la pérdida. Esto resuelve el RuntimeError por mismatch de tamaños.

Componentes Clave:
- Carga de tensor global de conectividad y metadatos de sujetos.
- Selección de cohorte para clasificación CN vs. AD.
- Implementación de Validación Cruzada Anidada (Nested Cross-Validation).
- Normalización Inter-Canal Global (aplicada dentro de los folds de CV para evitar data leakage).
- Definición de un VAE Convolucional 2D con PyTorch.
- Entrenamiento del VAE sobre todos los sujetos disponibles en el fold de entrenamiento.
- Extracción de características latentes para sujetos CN y AD.
- Entrenamiento y evaluación de un clasificador SVM sobre las características latentes.
- Reporte de métricas de clasificación.

Próximos Pasos Sugeridos para Tesis:
- Implementación de optimización de hiperparámetros robusta (ej. Optuna) en el bucle interno de CV.
- Experimentación con diferentes arquitecturas de VAE y dimensiones latentes.
- Evaluación de diferentes clasificadores.
- Estudio de ablación de canales de conectividad.
- Análisis de interpretabilidad del espacio latente y de los clasificadores.
- Integración de harmonización multi-sitio (ej. ComBat) si es aplicable.
- Estrategias avanzadas de entrenamiento de VAE (ej. beta-VAE, aprendizaje contrastivo).
"""

import argparse
import gc
import hashlib  # Para hash de IDs
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple  # Asegurar que Tuple está importado

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import (
    StandardScaler as SklearnScaler,
)  # Para escalar features latentes
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset

# --- Configuración del Logger ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
)
logger = logging.getLogger(__name__)


# --- Definición del Modelo VAE Convolucional ---
class ConvolutionalVAE(nn.Module):
    def __init__(
        self,
        input_channels=6,
        latent_dim=128,
        image_size=131,
        final_activation="sigmoid",
    ):
        super(ConvolutionalVAE, self).__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size  # Guardamos el tamaño de entrada original
        self.final_activation_name = final_activation

        # Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(
                input_channels, 32, kernel_size=7, stride=2, padding=3
            ),  # -> (B, 32, H/2, W/2)
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(
                32, 64, kernel_size=5, stride=2, padding=2
            ),  # -> (B, 64, H/4, W/4)
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(
                64, 128, kernel_size=3, stride=2, padding=1
            ),  # -> (B, 128, H/8, W/8)
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(
                128, 256, kernel_size=3, stride=2, padding=1
            ),  # -> (B, 256, H/16, W/16)
            nn.ReLU(),
            nn.BatchNorm2d(256),
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, image_size, image_size)
            self.final_spatial_dim_encoder = self.encoder_conv(dummy_input).shape[
                -1
            ]  # Renombrado para claridad
            self.flattened_size = (
                256 * self.final_spatial_dim_encoder * self.final_spatial_dim_encoder
            )
            logger.info(
                f"VAE Encoder: Calculated flattened size = {self.flattened_size} (final spatial dim after encoder: {self.final_spatial_dim_encoder})"
            )

        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)

        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, self.flattened_size)

        decoder_layers = [
            nn.ConvTranspose2d(
                256, 128, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(
                64, 32, kernel_size=5, stride=2, padding=2, output_padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(
                32, input_channels, kernel_size=7, stride=2, padding=3, output_padding=1
            ),
        ]

        if self.final_activation_name == "sigmoid":
            decoder_layers.append(nn.Sigmoid())
            logger.info("Decoder final activation: Sigmoid")
        elif self.final_activation_name == "tanh":
            decoder_layers.append(nn.Tanh())
            logger.info("Decoder final activation: Tanh")
        else:
            logger.info("Decoder final activation: Linear (None)")

        self.decoder_conv = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_conv(x)
        h = h.view(h.size(0), -1)  # Flatten
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_fc(z)
        h = h.view(
            h.size(0),
            256,
            self.final_spatial_dim_encoder,
            self.final_spatial_dim_encoder,
        )  # Unflatten
        return self.decoder_conv(h)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x_raw = self.decode(z)

        # Ajustar el tamaño de recon_x_raw para que coincida con x
        # Esto es necesario si las operaciones de ConvTranspose2d no invierten perfectamente el tamaño
        # para entradas de tamaño impar como 131.

        # Target H, W from input x
        target_h, target_w = x.shape[2], x.shape[3]
        current_h, current_w = recon_x_raw.shape[2], recon_x_raw.shape[3]

        # Si la reconstrucción es más grande, recortar (crop)
        if current_h > target_h or current_w > target_w:
            # Calcular el inicio del recorte para centrar
            h_start = (current_h - target_h) // 2
            w_start = (current_w - target_w) // 2
            recon_x = recon_x_raw[
                :, :, h_start : h_start + target_h, w_start : w_start + target_w
            ]
        # Si la reconstrucción es más pequeña, rellenar (pad) - menos común con output_padding=1
        elif current_h < target_h or current_w < target_w:
            padding_h_total = target_h - current_h
            padding_w_total = target_w - current_w

            pad_top = padding_h_total // 2
            pad_bottom = padding_h_total - pad_top
            pad_left = padding_w_total // 2
            pad_right = padding_w_total - pad_left

            recon_x = nn.functional.pad(
                recon_x_raw, (pad_left, pad_right, pad_top, pad_bottom)
            )
        else:  # Los tamaños ya coinciden
            recon_x = recon_x_raw

        if recon_x.shape != x.shape:
            # Fallback si el recorte/padding simple no es perfecto (ej. diferencia de 1 pixel por redondeo)
            # Esto es un último recurso, idealmente la arquitectura o el recorte/padding anterior lo manejan.
            logger.warning(
                f"Shape mismatch even after crop/pad: Input {x.shape}, Recon {recon_x.shape}. Attempting resize."
            )
            recon_x = nn.functional.interpolate(
                recon_x_raw,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )

        # assert recon_x.shape == x.shape, \
        #    f"Shape mismatch: Input {x.shape}, Recon {recon_x.shape}" # Descomentar para debug estricto

        return recon_x, mu, logvar


def vae_loss_function(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction="mean")
    kld_element = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kld = torch.mean(kld_element)
    return recon_loss + beta * kld


# --- Funciones Auxiliares ---
def load_data(
    tensor_path: Path, metadata_path: Path
) -> Tuple[Optional[np.ndarray], Optional[pd.DataFrame]]:
    logger.info(f"Cargando tensor global desde: {tensor_path}")
    if not tensor_path.exists():
        logger.error(f"Archivo de tensor global NO encontrado: {tensor_path}")
        return None, None
    try:
        data_npz = np.load(tensor_path)
        global_tensor = data_npz["global_tensor_data"]
        subject_ids_tensor = data_npz["subject_ids"]
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
        metadata_df["SubjectID"] = metadata_df["SubjectID"].astype(str).str.strip()
        logger.info(f"Metadatos cargados. Forma: {metadata_df.shape}")
    except Exception as e:
        logger.error(f"Error cargando metadatos: {e}")
        return None, None

    tensor_df = pd.DataFrame({"SubjectID": subject_ids_tensor})
    tensor_df["tensor_idx"] = np.arange(len(subject_ids_tensor))

    merged_df = pd.merge(metadata_df, tensor_df, on="SubjectID", how="inner")

    if len(merged_df) != len(subject_ids_tensor):
        logger.warning(
            f"Algunos sujetos del tensor no se encontraron en metadatos o viceversa. "
            f"Tensor: {len(subject_ids_tensor)}, Merged: {len(merged_df)}"
        )

    cols_to_keep = ["SubjectID", "ResearchGroup", "tensor_idx"]
    final_df = merged_df[cols_to_keep].copy()

    return global_tensor, final_df


def normalize_inter_channel_fold(
    data_tensor: np.ndarray,
    train_indices_in_fold: np.ndarray,
    mode: str = "zscore_offdiag",
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    """
    Normaliza cada canal del tensor global. Los parámetros (media, std, min, max) se calculan
    SOLO en los datos de entrenamiento del fold actual.
    """
    logger.info(
        f"Aplicando normalización inter-canal (modo: {mode}) usando parámetros de entrenamiento."
    )
    normalized_tensor_fold = data_tensor.copy()
    norm_params_per_channel = []
    num_channels = data_tensor.shape[1]
    num_rois = data_tensor.shape[2]

    for c_idx in range(num_channels):
        channel_data_train = data_tensor[train_indices_in_fold, c_idx, :, :]
        off_diag_mask = ~np.eye(num_rois, dtype=bool)

        all_off_diag_train_values = []
        for subj_idx in range(channel_data_train.shape[0]):
            all_off_diag_train_values.extend(
                channel_data_train[subj_idx][off_diag_mask]
            )
        all_off_diag_train_values = np.array(all_off_diag_train_values)

        params = {}
        if all_off_diag_train_values.size == 0:
            logger.warning(
                f"Canal {c_idx}: No hay elementos fuera de la diagonal en el conjunto de entrenamiento. No se escala."
            )
            params = {"mean": 0.0, "std": 1.0, "min": 0.0, "max": 1.0, "mode": mode}
        elif mode == "zscore_offdiag":
            mean_val = np.mean(all_off_diag_train_values)
            std_val = np.std(all_off_diag_train_values)
            params = {
                "mean": mean_val,
                "std": std_val if std_val > 1e-9 else 1.0,
                "mode": mode,
            }
            if std_val <= 1e-9:
                logger.warning(
                    f"Canal {c_idx}: STD muy bajo en entrenamiento ({std_val:.2e}). Se usará STD=1 para evitar división por cero."
                )
        elif mode == "minmax_offdiag":
            min_val = np.min(all_off_diag_train_values)
            max_val = np.max(all_off_diag_train_values)
            params = {"min": min_val, "max": max_val, "mode": mode}
            if (max_val - min_val) <= 1e-9:
                logger.warning(
                    f"Canal {c_idx}: Rango (max-min) muy bajo en entrenamiento ({(max_val - min_val):.2e}). La escala MinMax puede ser inestable."
                )
        else:
            raise ValueError(f"Modo de normalización desconocido: {mode}")
        norm_params_per_channel.append(params)

        for subj_glob_idx in range(data_tensor.shape[0]):
            current_matrix = data_tensor[subj_glob_idx, c_idx, :, :]
            scaled_matrix_ch = current_matrix.copy()
            if off_diag_mask.any():
                if mode == "zscore_offdiag":
                    if params["std"] > 1e-9:
                        scaled_matrix_ch[off_diag_mask] = (
                            current_matrix[off_diag_mask] - params["mean"]
                        ) / params["std"]
                elif mode == "minmax_offdiag":
                    if (params.get("max", 1.0) - params.get("min", 0.0)) > 1e-9:
                        scaled_matrix_ch[off_diag_mask] = (
                            current_matrix[off_diag_mask] - params["min"]
                        ) / (params["max"] - params["min"])
                    else:
                        scaled_matrix_ch[off_diag_mask] = 0.0
            normalized_tensor_fold[subj_glob_idx, c_idx, :, :] = scaled_matrix_ch

        if mode == "zscore_offdiag":
            logger.info(
                f"Canal {c_idx}: Off-diagonal Z-score aplicado (train_mean={params['mean']:.3f}, train_std={params['std']:.3f})."
            )
        elif mode == "minmax_offdiag":
            logger.info(
                f"Canal {c_idx}: Off-diagonal MinMax escalado aplicado (train_min={params['min']:.3f}, train_max={params['max']:.3f})."
            )

    return normalized_tensor_fold, norm_params_per_channel


def apply_normalization_params(
    data_tensor_subset: np.ndarray, norm_params_per_channel: List[Dict[str, float]]
) -> np.ndarray:
    """Aplica parámetros de normalización precalculados a un nuevo subconjunto de datos."""
    normalized_tensor_subset = data_tensor_subset.copy()
    num_channels = data_tensor_subset.shape[1]
    num_rois = data_tensor_subset.shape[2]
    off_diag_mask = ~np.eye(num_rois, dtype=bool)

    for c_idx in range(num_channels):
        params = norm_params_per_channel[c_idx]
        mode = params.get("mode", "zscore_offdiag")
        for subj_idx in range(data_tensor_subset.shape[0]):
            current_matrix = data_tensor_subset[subj_idx, c_idx, :, :]
            scaled_matrix_ch = current_matrix.copy()
            if off_diag_mask.any():
                if mode == "zscore_offdiag":
                    if params["std"] > 1e-9:
                        scaled_matrix_ch[off_diag_mask] = (
                            current_matrix[off_diag_mask] - params["mean"]
                        ) / params["std"]
                elif mode == "minmax_offdiag":
                    if (params.get("max", 1.0) - params.get("min", 0.0)) > 1e-9:
                        scaled_matrix_ch[off_diag_mask] = (
                            current_matrix[off_diag_mask] - params["min"]
                        ) / (params["max"] - params["min"])
                    else:
                        scaled_matrix_ch[off_diag_mask] = 0.0
            normalized_tensor_subset[subj_idx, c_idx, :, :] = scaled_matrix_ch
    return normalized_tensor_subset


# --- Función Principal de Entrenamiento y Evaluación ---
def train_and_evaluate_pipeline(
    global_tensor: np.ndarray, metadata_df: pd.DataFrame, args: argparse.Namespace
):
    """
    Implementa el pipeline de Nested CV para entrenar VAE y clasificador.
    """

    # 1. Filtrar para clasificación CN vs AD
    cn_ad_df = metadata_df[metadata_df["ResearchGroup"].isin(["CN", "AD"])].copy()
    if cn_ad_df.empty:
        logger.error("No se encontraron sujetos CN o AD en los metadatos.")
        return

    label_mapping = {"CN": 0, "AD": 1}
    cn_ad_df["label"] = cn_ad_df["ResearchGroup"].map(label_mapping)

    X_classifier_indices = cn_ad_df["tensor_idx"].values
    y_classifier = cn_ad_df["label"].values

    logger.info(
        f"Iniciando clasificación CN vs AD. Total sujetos CN/AD: {len(cn_ad_df)}. "
        f"CN: {sum(y_classifier == 0)}, AD: {sum(y_classifier == 1)}"
    )

    # 2. Configuración de Nested Cross-Validation
    outer_cv_clf = StratifiedKFold(
        n_splits=args.outer_folds, shuffle=True, random_state=args.seed
    )

    fold_metrics = []

    for fold_idx, (
        train_dev_clf_indices_in_cn_ad,
        test_clf_indices_in_cn_ad,
    ) in enumerate(outer_cv_clf.split(np.zeros(len(y_classifier)), y_classifier)):
        logger.info(f"--- Iniciando Fold Externo {fold_idx + 1}/{args.outer_folds} ---")

        global_train_dev_clf_subject_indices = X_classifier_indices[
            train_dev_clf_indices_in_cn_ad
        ]
        global_test_clf_subject_indices = X_classifier_indices[
            test_clf_indices_in_cn_ad
        ]

        all_subject_global_indices = metadata_df["tensor_idx"].values
        vae_train_dev_global_indices = np.setdiff1d(
            all_subject_global_indices,
            global_test_clf_subject_indices,
            assume_unique=True,
        )

        vae_train_dev_tensor = global_tensor[vae_train_dev_global_indices]

        logger.info(
            "Normalizando datos para VAE (entrenamiento VAE y test del clasificador)..."
        )
        vae_train_dev_tensor_norm, norm_params_fold = normalize_inter_channel_fold(
            vae_train_dev_tensor,
            np.arange(len(vae_train_dev_tensor)),
            mode=args.norm_mode,
        )

        test_clf_tensor_subset_global = global_tensor[global_test_clf_subject_indices]
        test_clf_tensor_norm = apply_normalization_params(
            test_clf_tensor_subset_global, norm_params_fold
        )

        vae_dataset = TensorDataset(torch.from_numpy(vae_train_dev_tensor_norm).float())
        vae_dataloader = DataLoader(
            vae_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Usando dispositivo: {device}")

        logger.info(
            f"VAE decoder final activation set to: {args.vae_final_activation} based on norm_mode: {args.norm_mode}"
        )

        vae = ConvolutionalVAE(
            input_channels=global_tensor.shape[1],
            latent_dim=args.latent_dim,
            image_size=global_tensor.shape[2],
            final_activation=args.vae_final_activation,
        ).to(device)
        optimizer_vae = optim.Adam(
            vae.parameters(), lr=args.lr_vae, weight_decay=args.weight_decay_vae
        )

        logger.info(f"Entrenando VAE para el fold {fold_idx + 1}...")
        vae.train()
        for epoch in range(args.epochs_vae):
            epoch_loss = 0
            for batch_idx, (data,) in enumerate(vae_dataloader):
                data = data.to(device)
                optimizer_vae.zero_grad()
                recon_batch, mu, logvar = vae(data)
                loss = vae_loss_function(
                    recon_batch, data, mu, logvar, beta=args.beta_vae
                )
                loss.backward()
                optimizer_vae.step()
                epoch_loss += loss.item() * data.size(0)
            avg_epoch_loss = epoch_loss / len(vae_dataloader.dataset)
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Fold {fold_idx+1}, VAE Epoch {epoch+1}/{args.epochs_vae}, Loss: {avg_epoch_loss:.4f}"
                )

        vae_model_path = (
            Path(args.output_dir)
            / f"vae_fold_{fold_idx+1}_ld{args.latent_dim}_beta{args.beta_vae}.pt"
        )
        torch.save(vae.state_dict(), vae_model_path)
        logger.info(f"Modelo VAE del fold {fold_idx+1} guardado en: {vae_model_path}")

        train_dev_clf_tensor_subset_global = global_tensor[
            global_train_dev_clf_subject_indices
        ]
        train_dev_clf_tensor_norm = apply_normalization_params(
            train_dev_clf_tensor_subset_global, norm_params_fold
        )

        vae.eval()
        with torch.no_grad():
            mu_train_clf, _ = vae.encode(
                torch.from_numpy(train_dev_clf_tensor_norm).float().to(device)
            )
            mu_test_clf, _ = vae.encode(
                torch.from_numpy(test_clf_tensor_norm).float().to(device)
            )

        X_train_latent = mu_train_clf.cpu().numpy()
        X_test_latent = mu_test_clf.cpu().numpy()
        y_train_clf = y_classifier[train_dev_clf_indices_in_cn_ad]
        y_test_clf = y_classifier[test_clf_indices_in_cn_ad]

        if X_train_latent.shape[0] == 0 or X_test_latent.shape[0] == 0:
            logger.warning(
                f"Fold {fold_idx+1}: No data for classifier training or testing after filtering CN/AD. Skipping."
            )
            fold_metrics.append(
                {
                    "fold": fold_idx + 1,
                    "auc": np.nan,
                    "pr_auc": np.nan,
                    "accuracy": np.nan,
                    "sensitivity": np.nan,
                    "specificity": np.nan,
                    "f1_score": np.nan,
                }
            )
            continue

        scaler_latent = SklearnScaler()
        X_train_latent_scaled = scaler_latent.fit_transform(X_train_latent)
        X_test_latent_scaled = scaler_latent.transform(X_test_latent)

        param_grid_svm = {
            "C": [0.1, 1, 10, 100],
            "gamma": [0.001, 0.01, 0.1, 1, "scale", "auto"],
            "kernel": ["rbf"],
        }
        inner_cv_svm = StratifiedKFold(
            n_splits=args.inner_folds,
            shuffle=True,
            random_state=args.seed + fold_idx + 100,
        )

        svm_clf = GridSearchCV(
            SVC(
                probability=True,
                random_state=args.seed,
                class_weight="balanced" if args.class_weight_svm else None,
            ),
            param_grid_svm,
            cv=inner_cv_svm,
            scoring="roc_auc",
            n_jobs=1,
        )

        logger.info(f"Entrenando clasificador SVM para el fold {fold_idx + 1}...")
        svm_clf.fit(X_train_latent_scaled, y_train_clf)
        logger.info(
            f"Mejores hiperparámetros SVM para fold {fold_idx + 1}: {svm_clf.best_params_}"
        )

        y_pred_proba_clf = svm_clf.predict_proba(X_test_latent_scaled)[:, 1]
        y_pred_clf = svm_clf.predict(X_test_latent_scaled)

        auc = roc_auc_score(y_test_clf, y_pred_proba_clf)
        pr_auc = average_precision_score(y_test_clf, y_pred_proba_clf)
        acc = accuracy_score(y_test_clf, y_pred_clf)
        sens = recall_score(y_test_clf, y_pred_clf, pos_label=1, zero_division=0)
        spec = recall_score(y_test_clf, y_pred_clf, pos_label=0, zero_division=0)
        f1 = f1_score(y_test_clf, y_pred_clf, pos_label=1, zero_division=0)

        logger.info(
            f"Fold {fold_idx + 1} - Resultados del Clasificador en Test Externo:"
        )
        logger.info(f"  AUC: {auc:.4f}, PR-AUC: {pr_auc:.4f}, Accuracy: {acc:.4f}")
        logger.info(
            f"  Sensitivity (AD): {sens:.4f}, Specificity (CN): {spec:.4f}, F1-score (AD): {f1:.4f}"
        )

        train_ids_str = ",".join(
            sorted(cn_ad_df.iloc[train_dev_clf_indices_in_cn_ad]["SubjectID"].tolist())
        )
        test_ids_str = ",".join(
            sorted(cn_ad_df.iloc[test_clf_indices_in_cn_ad]["SubjectID"].tolist())
        )
        logger.info(
            f"  Fold {fold_idx+1} Train IDs Hash: {hashlib.md5(train_ids_str.encode()).hexdigest()}"
        )
        logger.info(
            f"  Fold {fold_idx+1} Test IDs Hash: {hashlib.md5(test_ids_str.encode()).hexdigest()}"
        )

        fold_metrics.append(
            {
                "fold": fold_idx + 1,
                "auc": auc,
                "pr_auc": pr_auc,
                "accuracy": acc,
                "sensitivity": sens,
                "specificity": spec,
                "f1_score": f1,
            }
        )

        del vae, optimizer_vae, vae_dataset, vae_dataloader
        del (
            svm_clf,
            X_train_latent,
            X_test_latent,
            X_train_latent_scaled,
            X_test_latent_scaled,
        )
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if fold_metrics:
        metrics_df = pd.DataFrame(fold_metrics)
        logger.info("\n--- Resumen de Rendimiento (Promedio sobre Folds Externos) ---")
        logger.info(
            f"AUC:         {metrics_df['auc'].mean():.4f} +/- {metrics_df['auc'].std():.4f}"
        )
        logger.info(
            f"PR-AUC:      {metrics_df['pr_auc'].mean():.4f} +/- {metrics_df['pr_auc'].std():.4f}"
        )
        logger.info(
            f"Accuracy:    {metrics_df['accuracy'].mean():.4f} +/- {metrics_df['accuracy'].std():.4f}"
        )
        logger.info(
            f"Sensitivity: {metrics_df['sensitivity'].mean():.4f} +/- {metrics_df['sensitivity'].std():.4f}"
        )
        logger.info(
            f"Specificity: {metrics_df['specificity'].mean():.4f} +/- {metrics_df['specificity'].std():.4f}"
        )
        logger.info(
            f"F1-score:    {metrics_df['f1_score'].mean():.4f} +/- {metrics_df['f1_score'].std():.4f}"
        )

        results_filename = (
            f"classification_results_vae_svm_{args.norm_mode}_ld{args.latent_dim}_beta{args.beta_vae}"
            f"_lr{args.lr_vae}_ep{args.epochs_vae}_bs{args.batch_size}.csv"
        )
        results_path = Path(args.output_dir) / results_filename
        results_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(results_path, index=False)
        logger.info(f"Resultados de clasificación guardados en: {results_path}")
    else:
        logger.warning("No se pudieron calcular métricas para ningún fold.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline de Entrenamiento VAE y Clasificador para AD vs CN"
    )
    parser.add_argument(
        "--global_tensor_path",
        type=str,
        required=True,
        help="Ruta al archivo .npz del tensor global de conectividad.",
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        required=True,
        help="Ruta al archivo CSV de metadatos de sujetos (ej. SubjectsData_Schaefer2018_400ROIs.csv).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./vae_classification_output",
        help="Directorio para guardar resultados y modelos VAE por fold.",
    )

    # Parámetros de CV
    parser.add_argument(
        "--outer_folds", type=int, default=5, help="Número de folds para la CV externa."
    )
    parser.add_argument(
        "--inner_folds",
        type=int,
        default=3,
        help="Número de folds para la CV interna (optimización de hiperparámetros del clasificador).",
    )

    # Parámetros del VAE
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=64,
        help="Dimensionalidad del espacio latente del VAE.",
    )
    parser.add_argument(
        "--lr_vae", type=float, default=1e-4, help="Tasa de aprendizaje para el VAE."
    )
    parser.add_argument(
        "--epochs_vae",
        type=int,
        default=50,
        help="Número de épocas para entrenar el VAE.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Tamaño del batch para el VAE."
    )
    parser.add_argument(
        "--beta_vae",
        type=float,
        default=1.0,
        help="Factor beta para la pérdida KL en el VAE.",
    )
    parser.add_argument(
        "--weight_decay_vae",
        type=float,
        default=1e-5,
        help="Weight decay para el optimizador Adam del VAE.",
    )
    parser.add_argument(
        "--vae_final_activation",
        type=str,
        default="sigmoid",
        choices=["sigmoid", "tanh", "linear"],
        help="Activación final del decoder VAE. 'sigmoid' para [0,1], 'tanh' para [-1,1], 'linear' para Z-scores.",
    )

    # Parámetros de Normalización
    parser.add_argument(
        "--norm_mode",
        type=str,
        default="minmax_offdiag",
        choices=["zscore_offdiag", "minmax_offdiag"],
        help="Modo de normalización inter-canal. 'minmax_offdiag' escala off-diagonals a [0,1] (compatible con Sigmoid). "
        "'zscore_offdiag' estandariza off-diagonals (compatible con Tanh o lineal).",
    )

    # Parámetros del Clasificador SVM
    parser.add_argument(
        "--class_weight_svm",
        action="store_true",
        help="Usar class_weight='balanced' en SVM.",
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="Semilla aleatoria para reproducibilidad."
    )

    args = parser.parse_args()

    # Configurar semilla para reproducibilidad
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Cargar datos
    global_tensor_data, metadata_df_full = load_data(
        Path(args.global_tensor_path), Path(args.metadata_path)
    )

    if global_tensor_data is not None and metadata_df_full is not None:
        logger.info(f"Modo de normalización inter-canal seleccionado: {args.norm_mode}")
        logger.info(
            f"Activación final del decoder VAE seleccionada: {args.vae_final_activation}"
        )
        if (
            args.norm_mode == "minmax_offdiag"
            and args.vae_final_activation != "sigmoid"
        ):
            logger.warning(
                "Se seleccionó norm_mode='minmax_offdiag' pero la activación final del VAE no es 'sigmoid'. "
                "Esto podría llevar a un rango de reconstrucción inconsistente."
            )
        if (
            args.norm_mode == "zscore_offdiag"
            and args.vae_final_activation == "sigmoid"
        ):
            logger.warning(
                "Se seleccionó norm_mode='zscore_offdiag' pero la activación final del VAE es 'sigmoid'. "
                "Esto podría llevar a un rango de reconstrucción inconsistente. Considere 'tanh' o 'linear'."
            )

        train_and_evaluate_pipeline(global_tensor_data, metadata_df_full, args)
    else:
        logger.critical(
            "No se pudieron cargar los datos. Abortando el pipeline de entrenamiento."
        )

    logger.info("Pipeline de entrenamiento y evaluación completado.")

import argparse
import copy
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch_geometric.nn import GATConv, global_mean_pool

MODE_STANDARD_F1 = "STANDARD_F1"
MODE_PA_F1 = "PA_F1"
MODE_RECALL_BOOST = "RECALL_BOOST"


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SlidingWindowDataset(torch.utils.data.Dataset):
    """Convert point-wise time series to window samples."""

    def __init__(
        self,
        values: np.ndarray,
        labels: np.ndarray,
        window_size: int,
        missing_rate: float = 0.0,
        anomaly_ratio_threshold: float = 0.5,
        verbose: bool = False,
    ) -> None:
        self.window_size = int(window_size)
        self.missing_rate = float(missing_rate)

        windows = []
        win_labels = []
        total_points = len(values)
        if total_points < self.window_size:
            raise ValueError(
                f"window_size={self.window_size} is larger than series length={total_points}"
            )

        for start in range(total_points - self.window_size + 1):
            end = start + self.window_size
            windows.append(values[start:end])
            ratio = float(labels[start:end].mean())
            win_labels.append(int(ratio > anomaly_ratio_threshold))

        self.windows = np.asarray(windows, dtype=np.float32)
        self.labels = np.asarray(win_labels, dtype=np.int64)

        if verbose:
            anom = int(self.labels.sum())
            all_cnt = len(self.labels)
            ratio = (anom / all_cnt) if all_cnt else 0.0
            print(
                f"  windows={all_cnt}, anomaly_windows={anom}, anomaly_ratio={ratio:.2%}"
            )

    def __len__(self) -> int:
        return int(len(self.windows))

    def __getitem__(self, index: int):
        window = self.windows[index].copy()[..., np.newaxis]  # [W, N, 1]
        mask = np.ones_like(window, dtype=np.float32)

        nan_locs = np.isnan(window)
        if nan_locs.any():
            window[nan_locs] = 0.0
            mask[nan_locs] = 0.0

        if self.missing_rate > 0:
            random_drop = np.random.rand(*window.shape) < self.missing_rate
            window[random_drop] = 0.0
            mask[random_drop] = 0.0

        return (
            torch.from_numpy(window).float(),
            torch.from_numpy(mask).float(),
            torch.tensor(self.labels[index], dtype=torch.long),
        )


class SensorGraphVAE(nn.Module):
    """
    Keep the original architecture idea:
    - one graph node per sensor
    - node feature is the full temporal window
    - graph-level latent variable and MLP decoder
    """

    def __init__(
        self,
        n_sensors: int,
        window_size: int,
        hidden_dim: int,
        latent_dim: int,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.n_sensors = int(n_sensors)
        self.window_size = int(window_size)
        self.edge_cache: Dict[Tuple[int, str, int], torch.Tensor] = {}

        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(
            GATConv(
                in_channels=self.window_size,
                out_channels=hidden_dim,
                heads=heads,
                dropout=dropout,
                concat=False,
            )
        )
        for _ in range(num_layers - 1):
            self.gat_layers.append(
                GATConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    heads=heads,
                    dropout=dropout,
                    concat=False,
                )
            )

        self.dropout = nn.Dropout(dropout)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.n_sensors * self.window_size),
        )

    def _build_batched_edges(
        self, base_edge_index: torch.Tensor, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        dev_idx = -1 if device.index is None else int(device.index)
        key = (batch_size, device.type, dev_idx)
        cached = self.edge_cache.get(key)
        if cached is not None:
            return cached

        offsets = (
            torch.arange(batch_size, device=device).view(-1, 1, 1) * self.n_sensors
        )
        expanded = base_edge_index.unsqueeze(0) + offsets
        merged = expanded.permute(1, 0, 2).reshape(2, -1).contiguous()
        self.edge_cache[key] = merged
        return merged

    def encode(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        base_edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, n_sensors, _ = x.shape
        if n_sensors != self.n_sensors:
            raise ValueError(f"sensor mismatch: {n_sensors} != {self.n_sensors}")

        node_feat = x.squeeze(-1).permute(0, 2, 1).contiguous()  # [B, N, W]
        node_mask = mask.squeeze(-1).permute(0, 2, 1).contiguous()

        flat_feat = node_feat.view(batch_size * self.n_sensors, self.window_size)
        flat_mask = node_mask.view(batch_size * self.n_sensors, self.window_size)

        node_valid = flat_mask.mean(dim=1).clamp(0.0, 1.0)
        edge_index = self._build_batched_edges(base_edge_index, batch_size, x.device)
        edge_weight = node_valid[edge_index[0]] * node_valid[edge_index[1]]

        h = flat_feat
        for layer in self.gat_layers:
            h = layer(h, edge_index, edge_attr=edge_weight)
            h = F.elu(h)
            h = self.dropout(h)

        batch_index = torch.arange(batch_size, device=x.device).repeat_interleave(
            self.n_sensors
        )
        graph_feat = global_mean_pool(h, batch_index)

        mu = self.fc_mu(graph_feat)
        # Bounding posterior variance prevents exp(logvar) overflow without
        # changing the VAE architecture or objective.
        logvar = self.fc_logvar(graph_feat).clamp(-10.0, 10.0)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        recon = self.decoder(z)
        recon = recon.view(-1, self.n_sensors, self.window_size)
        return recon.permute(0, 2, 1).unsqueeze(-1).contiguous()

    def forward(
        self,
        x: torch.Tensor,
        base_edge_index: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, mask, base_edge_index)
        # Sampling is required for VAE training. At inference, the posterior
        # mean gives deterministic and substantially more stable scores.
        z = self.reparameterize(mu, logvar) if self.training else mu
        recon = self.decode(z)
        return recon, mu, logvar


def reconstruction_kl_loss(
    x: torch.Tensor,
    x_recon: torch.Tensor,
    mask: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logvar = logvar.clamp(-10.0, 10.0)
    recon = ((x - x_recon) ** 2 * mask).sum() / mask.sum().clamp_min(1.0)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    total = recon + beta * kl
    return total, recon, kl


def anomaly_score(
    x: torch.Tensor,
    x_recon: torch.Tensor,
    mask: torch.Tensor,
    score_alpha: float,
) -> torch.Tensor:
    sqe = ((x - x_recon) ** 2) * mask

    mean_score = sqe.sum(dim=(1, 2, 3)) / mask.sum(dim=(1, 2, 3)).clamp_min(1e-6)

    sensor_wise = sqe.sum(dim=(1, 3)) / mask.sum(dim=(1, 3)).clamp_min(1e-6)
    k = max(1, int(sensor_wise.shape[1] * 0.1))
    topk = torch.topk(sensor_wise, k=k, dim=1).values.mean(dim=1)

    return score_alpha * mean_score + (1.0 - score_alpha) * topk


def compute_confidence_score(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """
    Scheme A:
    per-sample KL divergence as confidence score.
    Higher value = larger prior deviation = lower confidence = more suspicious.
    """
    # 0.5 * sum(mu^2 + exp(log_var) - 1 - log_var)
    log_var = log_var.clamp(-10.0, 10.0)
    kl = 0.5 * (mu.pow(2) + log_var.exp() - 1.0 - log_var).sum(dim=-1)
    return kl


def smooth_1d(x: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return x
    pad = kernel_size // 2
    padded = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(kernel_size, dtype=np.float64) / kernel_size
    return np.convolve(padded, kernel, mode="valid")


def hysteresis(scores: np.ndarray, high_th: float, low_th: float) -> np.ndarray:
    out = np.zeros(len(scores), dtype=np.int64)
    active = False
    for idx, score in enumerate(scores):
        if (not active) and score >= high_th:
            active = True
        elif active and score < low_th:
            active = False
        out[idx] = 1 if active else 0
    return out


def fill_small_gaps(preds: np.ndarray, max_gap: int) -> np.ndarray:
    if max_gap <= 0:
        return preds
    out = preds.copy()
    i = 0
    n = len(out)
    while i < n:
        if out[i] == 1:
            j = i
            while j < n and out[j] == 1:
                j += 1
            k = j
            while k < n and out[k] == 0:
                k += 1
            gap = k - j
            if k < n and 0 < gap <= max_gap:
                out[j:k] = 1
            i = j
        else:
            i += 1
    return out


def remove_short_segments(preds: np.ndarray, min_seg_len: int) -> np.ndarray:
    if min_seg_len <= 1:
        return preds
    out = preds.copy()
    i = 0
    n = len(out)
    while i < n:
        if out[i] == 1:
            j = i
            while j < n and out[j] == 1:
                j += 1
            if (j - i) < min_seg_len:
                out[i:j] = 0
            i = j
        else:
            i += 1
    return out


def point_adjust(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    out = y_pred.astype(np.int64).copy()
    labels = y_true.astype(np.int64)
    i = 0
    n = len(labels)
    while i < n:
        if labels[i] == 1:
            j = i
            while j < n and labels[j] == 1:
                j += 1
            if out[i:j].any():
                out[i:j] = 1
            i = j
        else:
            i += 1
    return out


def mode_metric(labels: np.ndarray, preds: np.ndarray, mode: str) -> float:
    if mode == MODE_PA_F1:
        pa = point_adjust(labels, preds)
        return f1_score(labels, pa, zero_division=0)
    return f1_score(labels, preds, zero_division=0)


def binary_f1_recall_fast(labels: np.ndarray, preds: np.ndarray) -> Tuple[float, float]:
    """Compute binary F1/recall without sklearn's per-call validation overhead."""
    labels_bool = labels.astype(bool, copy=False)
    preds_bool = preds.astype(bool, copy=False)
    tp = int(np.count_nonzero(labels_bool & preds_bool))
    fp = int(np.count_nonzero(~labels_bool & preds_bool))
    fn = int(np.count_nonzero(labels_bool & ~preds_bool))
    f1_den = 2 * tp + fp + fn
    recall_den = tp + fn
    f1 = (2.0 * tp / f1_den) if f1_den else 0.0
    recall = (tp / recall_den) if recall_den else 0.0
    return f1, recall


def tune_postprocess(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    mode: str,
    quick: bool,
    recall_target: float,
) -> Dict[str, float]:
    if mode == MODE_RECALL_BOOST:
        smooth_grid = [1, 3, 5, 7, 9]
        low_grid = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7]
        gap_grid = [0, 1, 2, 3, 5, 8]
        seg_grid = [1, 2, 3, 5, 8]
    else:
        smooth_grid = [1, 3, 5] if quick else [1, 3, 5, 7, 9]
        low_grid = [1.0, 0.9, 0.8]
        gap_grid = [0, 1, 2] if quick else [0, 1, 2, 3, 5]
        seg_grid = [1, 2, 3, 5]

    best = {"score": -1.0, "smooth_k": 1, "low_ratio": 1.0, "max_gap": 0, "min_seg": 1}

    for smooth_k in smooth_grid:
        smoothed = smooth_1d(scores, smooth_k)
        for low_ratio in low_grid:
            low_th = threshold * low_ratio
            base_pred = hysteresis(smoothed, threshold, low_th)
            for max_gap in gap_grid:
                filled = fill_small_gaps(base_pred, max_gap)
                for min_seg in seg_grid:
                    pred = remove_short_segments(filled, min_seg)
                    if mode == MODE_RECALL_BOOST:
                        rec = recall_score(labels, pred, zero_division=0)
                        if rec < recall_target:
                            continue
                    score = mode_metric(labels, pred, mode)
                    if score > best["score"]:
                        best = {
                            "score": score,
                            "smooth_k": smooth_k,
                            "low_ratio": low_ratio,
                            "max_gap": max_gap,
                            "min_seg": min_seg,
                        }
    return best


def apply_postprocess(
    scores: np.ndarray,
    threshold: float,
    pp: Dict[str, float],
) -> np.ndarray:
    smoothed = smooth_1d(scores, int(pp["smooth_k"]))
    low_th = threshold * float(pp["low_ratio"])
    pred = hysteresis(smoothed, threshold, low_th)
    pred = fill_small_gaps(pred, int(pp["max_gap"]))
    pred = remove_short_segments(pred, int(pp["min_seg"]))
    return pred.astype(np.int64)


def find_best_threshold_f1(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    labels = labels.astype(np.int64)
    if labels.sum() == 0:
        return float(np.percentile(scores, 95)), 0.0
    if labels.sum() == len(labels):
        return float(scores.min() - 1e-6), 0.0

    ratio = labels.mean()
    if ratio < 0.1:
        q_grid = np.linspace(0.80, 0.999, 300)
    elif ratio < 0.2:
        q_grid = np.linspace(0.60, 0.995, 300)
    else:
        q_grid = np.linspace(0.40, 0.990, 300)

    best_th = float(np.quantile(scores, q_grid[0]))
    best_f1 = -1.0
    for q in q_grid:
        th = float(np.quantile(scores, q))
        pred = (scores > th).astype(np.int64)
        f1 = f1_score(labels, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th

    return best_th, float(best_f1)


def find_best_threshold_fbeta(
    scores: np.ndarray,
    labels: np.ndarray,
    beta: float,
    q_min: float = 0.2,
    q_max: float = 0.999,
    n_steps: int = 700,
) -> Tuple[float, float]:
    labels = labels.astype(np.int64)
    if labels.sum() == 0:
        return float(np.percentile(scores, 95)), 0.0
    if labels.sum() == len(labels):
        return float(scores.min() - 1e-6), 0.0

    beta2 = beta * beta
    best_th = float(np.quantile(scores, 0.95))
    best_score = -1.0

    for q in np.linspace(q_min, q_max, n_steps):
        th = float(np.quantile(scores, q))
        pred = (scores > th).astype(np.int64)

        tp = np.sum((pred == 1) & (labels == 1))
        fp = np.sum((pred == 1) & (labels == 0))
        fn = np.sum((pred == 0) & (labels == 1))

        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        fbeta = (1 + beta2) * precision * recall / (beta2 * precision + recall + 1e-9)

        if fbeta > best_score:
            best_score = fbeta
            best_th = th

    return best_th, float(best_score)


def find_best_threshold_recall_constrained(
    scores: np.ndarray,
    labels: np.ndarray,
    recall_target: float,
    q_min: float = 0.01,
    q_max: float = 0.95,
    n_steps: int = 900,
) -> Tuple[float, float]:
    labels = labels.astype(np.int64)
    if labels.sum() == 0:
        return float(np.percentile(scores, 95)), 0.0
    if labels.sum() == len(labels):
        return float(scores.min() - 1e-6), 0.0

    best_th = float(np.quantile(scores, 0.5))
    best_f1 = -1.0
    best_recall = -1.0

    q_space = np.linspace(q_min, q_max, n_steps)
    for q in q_space:
        th = float(np.quantile(scores, q))
        pred = (scores > th).astype(np.int64)
        rec = recall_score(labels, pred, zero_division=0)
        f1 = f1_score(labels, pred, zero_division=0)

        if rec >= recall_target and f1 > best_f1:
            best_f1 = f1
            best_th = th
        if rec > best_recall:
            best_recall = rec

    if best_f1 < 0:
        fallback_f1 = -1.0
        fallback_th = best_th
        for q in q_space:
            th = float(np.quantile(scores, q))
            pred = (scores > th).astype(np.int64)
            rec = recall_score(labels, pred, zero_division=0)
            f1 = f1_score(labels, pred, zero_division=0)
            if rec == best_recall and f1 > fallback_f1:
                fallback_f1 = f1
                fallback_th = th
        return fallback_th, max(fallback_f1, 0.0)

    return best_th, max(best_f1, 0.0)


def pick_threshold_by_mode(
    scores: np.ndarray,
    labels: np.ndarray,
    mode: str,
    threshold_beta: float,
    recall_target: float,
) -> Tuple[float, float]:
    if mode == MODE_PA_F1:
        return find_best_threshold_fbeta(scores, labels, beta=threshold_beta)
    if mode == MODE_RECALL_BOOST:
        return find_best_threshold_recall_constrained(scores, labels, recall_target)
    return find_best_threshold_f1(scores, labels)


def dual_predict(
    score_r: np.ndarray,
    score_c: np.ndarray,
    tau_r: float,
    tau_c: float,
    logic: str,
) -> np.ndarray:
    if logic == "recon":
        return (score_r > tau_r).astype(np.int64)
    if logic == "confidence":
        return (score_c > tau_c).astype(np.int64)
    if logic == "and":
        return ((score_r > tau_r) & (score_c > tau_c)).astype(np.int64)
    return ((score_r > tau_r) | (score_c > tau_c)).astype(np.int64)


def sanitize_score_array(score: np.ndarray) -> np.ndarray:
    arr = np.asarray(score, dtype=np.float64).copy()
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=np.float64)

    finite_vals = arr[finite]
    lo = float(np.quantile(finite_vals, 0.001))
    hi = float(np.quantile(finite_vals, 0.999))
    mid = float(np.median(finite_vals))
    arr = np.nan_to_num(arr, nan=mid, posinf=hi, neginf=lo)
    arr = np.clip(arr, lo, hi)
    return arr


def calibrate_score_direction(score: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    score_clean = sanitize_score_array(score)
    labels = np.asarray(labels, dtype=np.int64)
    ref_max = float(score_clean.max()) if score_clean.size else 0.0

    if score_clean.size == 0 or len(np.unique(labels)) < 2:
        return 1.0, ref_max
    if np.allclose(score_clean.min(), score_clean.max()):
        return 1.0, ref_max

    auc = roc_auc_score(labels, score_clean)
    sign = 1.0 if auc >= 0.5 else -1.0
    return sign, ref_max


def apply_score_direction(score: np.ndarray, sign: float, ref_max: float) -> np.ndarray:
    score_clean = sanitize_score_array(score)
    if sign >= 0:
        return score_clean
    return np.clip(float(ref_max) - score_clean, 0.0, None)


def transform_dual_scores(
    score_r: np.ndarray,
    score_c: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    score_r_clean = sanitize_score_array(score_r)
    score_c_clean = sanitize_score_array(score_c)
    score_c_clean = np.log1p(np.clip(score_c_clean, 0.0, None))
    score_c_clean = sanitize_score_array(score_c_clean)
    return score_r_clean, score_c_clean


def build_quantile_grid(
    score: np.ndarray,
    grid_size: int,
    q_low: float = 0.05,
    q_high: float = 0.995,
) -> np.ndarray:
    score = sanitize_score_array(score)
    if np.allclose(score.min(), score.max()):
        return np.array([float(score.min())], dtype=np.float64)
    qs = np.linspace(q_low, q_high, max(2, int(grid_size)))
    grid = np.quantile(score, qs)
    grid = np.unique(grid)
    if grid.size == 0:
        return np.array([float(np.median(score))], dtype=np.float64)
    return grid.astype(np.float64)


def transfer_threshold_by_quantile(
    validation_score: np.ndarray,
    target_score: np.ndarray,
    threshold: float,
) -> float:
    """Transfer a labeled-validation threshold to a shifted unlabeled score set."""
    validation_score = sanitize_score_array(validation_score)
    target_score = sanitize_score_array(target_score)
    quantile = float(np.mean(validation_score <= threshold))
    quantile = float(np.clip(quantile, 0.001, 0.999))
    return float(np.quantile(target_score, quantile))


def transfer_threshold_robust_scale(
    validation_score: np.ndarray,
    target_score: np.ndarray,
    threshold: float,
) -> float:
    """Transfer a threshold in robust median/IQR units without matching prevalence."""
    validation_score = sanitize_score_array(validation_score)
    target_score = sanitize_score_array(target_score)
    val_q1, val_median, val_q3 = np.quantile(validation_score, [0.25, 0.5, 0.75])
    target_q1, target_median, target_q3 = np.quantile(target_score, [0.25, 0.5, 0.75])
    val_iqr = max(float(val_q3 - val_q1), 1e-9)
    target_iqr = max(float(target_q3 - target_q1), 1e-9)
    robust_z = (float(threshold) - float(val_median)) / val_iqr
    return float(target_median + robust_z * target_iqr)


def tune_dual_postprocess(
    base_pred: np.ndarray,
    labels: np.ndarray,
    mode: str,
    quick: bool,
    recall_target: float,
) -> Dict[str, float]:
    if quick:
        gap_grid = [0, 1, 2, 3]
        seg_grid = [1, 2, 3, 5]
    else:
        gap_grid = [0, 1, 2, 3, 5, 8]
        seg_grid = [1, 2, 3, 5, 8]

    best = {"score": -1.0, "max_gap": 0, "min_seg": 1}
    for max_gap in gap_grid:
        filled = fill_small_gaps(base_pred, max_gap)
        for min_seg in seg_grid:
            pred = remove_short_segments(filled, min_seg)
            rec = recall_score(labels, pred, zero_division=0)
            score = mode_metric(labels, pred, mode)
            if mode == MODE_RECALL_BOOST and rec < recall_target:
                continue
            if score > best["score"]:
                best = {"score": score, "max_gap": max_gap, "min_seg": min_seg}
    return best


def apply_dual_postprocess(base_pred: np.ndarray, pp: Dict[str, float]) -> np.ndarray:
    pred = fill_small_gaps(base_pred, int(pp["max_gap"]))
    pred = remove_short_segments(pred, int(pp["min_seg"]))
    return pred.astype(np.int64)


def search_best_dual_threshold(
    score_r: np.ndarray,
    score_c: np.ndarray,
    labels: np.ndarray,
    mode: str,
    logic: str,
    grid_size: int,
    recall_target: float,
) -> Tuple[float, float, float, str]:
    labels = labels.astype(np.int64)
    grid_size = max(2, int(grid_size))
    score_r, score_c = transform_dual_scores(score_r, score_c)
    tau_r_grid = build_quantile_grid(score_r, grid_size)
    tau_c_grid = build_quantile_grid(score_c, grid_size)

    best_metric = -1.0
    best_recall = -1.0
    best_tau_r = float(np.median(score_r))
    best_tau_c = float(np.median(score_c))
    best_logic = "or"
    # KL is dataset-dependent. Auto mode must be able to ignore either score
    # when validation shows that fusion hurts anomaly separation.
    logic_candidates = (
        ["recon", "confidence", "or", "and"] if logic == "auto" else [logic]
    )

    for current_logic in logic_candidates:
        for tau_r in tau_r_grid:
            for tau_c in tau_c_grid:
                pred = dual_predict(score_r, score_c, tau_r, tau_c, current_logic)
                if mode == MODE_PA_F1:
                    rec = recall_score(labels, pred, zero_division=0)
                    metric = mode_metric(labels, pred, mode)
                else:
                    metric, rec = binary_f1_recall_fast(labels, pred)

                if mode == MODE_RECALL_BOOST:
                    if rec >= recall_target:
                        if metric > best_metric:
                            best_metric = metric
                            best_recall = rec
                            best_tau_r = float(tau_r)
                            best_tau_c = float(tau_c)
                            best_logic = current_logic
                    elif best_metric < 0:
                        # Fallback when no candidate reaches recall target.
                        if (rec > best_recall) or (rec == best_recall and metric > best_metric):
                            best_metric = metric
                            best_recall = rec
                            best_tau_r = float(tau_r)
                            best_tau_c = float(tau_c)
                            best_logic = current_logic
                else:
                    if metric > best_metric:
                        best_metric = metric
                        best_recall = rec
                        best_tau_r = float(tau_r)
                        best_tau_c = float(tau_c)
                        best_logic = current_logic

    return best_tau_r, best_tau_c, max(best_metric, 0.0), best_logic


def evaluate_from_scores(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    pp: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    scores = sanitize_score_array(scores)
    if pp is None:
        pred = (scores > threshold).astype(np.int64)
    else:
        pred = apply_postprocess(scores, threshold, pp)

    pa_pred = point_adjust(labels, pred)
    auc = roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else 0.0
    return {
        "f1": f1_score(labels, pred, zero_division=0),
        "pa_f1": f1_score(labels, pa_pred, zero_division=0),
        "precision": precision_score(labels, pred, zero_division=0),
        "recall": recall_score(labels, pred, zero_division=0),
        "auc": float(auc),
    }


def evaluate_from_dual_scores(
    score_r: np.ndarray,
    score_c: np.ndarray,
    labels: np.ndarray,
    tau_r: float,
    tau_c: float,
    logic: str,
    pp: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    score_r, score_c = transform_dual_scores(score_r, score_c)
    pred = dual_predict(score_r, score_c, tau_r, tau_c, logic)
    if pp is not None:
        pred = apply_dual_postprocess(pred, pp)
    pa_pred = point_adjust(labels, pred)

    if len(np.unique(labels)) > 1:
        auc_r = float(roc_auc_score(labels, score_r))
        auc_c = float(roc_auc_score(labels, score_c))
        # For reporting one AUC, use a combined normalized score.
        sr = (score_r - score_r.min()) / (score_r.max() - score_r.min() + 1e-9)
        sc = (score_c - score_c.min()) / (score_c.max() - score_c.min() + 1e-9)
        combo = np.minimum(sr, sc) if logic == "and" else np.maximum(sr, sc)
        auc = float(roc_auc_score(labels, combo))
    else:
        auc_r, auc_c, auc = 0.0, 0.0, 0.0

    return {
        "f1": f1_score(labels, pred, zero_division=0),
        "pa_f1": f1_score(labels, pa_pred, zero_division=0),
        "precision": precision_score(labels, pred, zero_division=0),
        "recall": recall_score(labels, pred, zero_division=0),
        "auc": auc,
        "auc_r": auc_r,
        "auc_c": auc_c,
    }


def contiguous_threshold_split(
    test_data: np.ndarray,
    test_labels: np.ndarray,
    threshold_val_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(test_data)
    val_len = int(n * threshold_val_ratio)
    if val_len <= 0 or val_len >= n:
        raise ValueError(
            f"invalid threshold_val_ratio={threshold_val_ratio}, n_points={n}"
        )

    total_ratio = float(test_labels.mean())
    cumsum = np.concatenate([[0], np.cumsum(test_labels)])
    step = max(1, val_len // 100)

    best_start = 0
    best_diff = float("inf")
    for start in range(0, n - val_len + 1, step):
        val_sum = cumsum[start + val_len] - cumsum[start]
        rest_sum = cumsum[-1] - val_sum
        if val_sum == 0 or rest_sum == 0:
            continue

        ratio = val_sum / val_len
        diff = abs(float(ratio) - total_ratio)
        if diff < best_diff:
            best_diff = diff
            best_start = start

    end = best_start + val_len
    val_data = test_data[best_start:end]
    val_labels = test_labels[best_start:end]
    test_data_final = np.concatenate([test_data[:best_start], test_data[end:]], axis=0)
    test_labels_final = np.concatenate(
        [test_labels[:best_start], test_labels[end:]], axis=0
    )
    return val_data, val_labels, test_data_final, test_labels_final


def build_block_holdout_indices(
    n_samples: int,
    block_size: int,
    val_block_every: int,
) -> Tuple[np.ndarray, np.ndarray]:
    blocks = [
        np.arange(i, min(i + block_size, n_samples)) for i in range(0, n_samples, block_size)
    ]
    val_blocks = [b for bi, b in enumerate(blocks) if bi % val_block_every == 0]
    if len(val_blocks) == 0:
        val_blocks = [blocks[0]]

    val_idx = np.concatenate(val_blocks)
    mask = np.ones(n_samples, dtype=bool)
    mask[val_idx] = False
    test_idx = np.where(mask)[0]
    return val_idx, test_idx


@dataclass
class TrainConfig:
    window_size: int = 30
    hidden_dim: int = 64
    latent_dim: int = 32
    batch_size: int = 96
    lr: float = 8e-4
    beta: float = 0.1
    dropout: float = 0.1
    anomaly_ratio_threshold: float = 0.01
    score_alpha: float = 0.5
    missing_rate: float = 0.05
    epochs: int = 30
    patience: int = 8
    eval_every: int = 2
    use_amp: bool = True
    metric_mode: str = MODE_STANDARD_F1
    threshold_beta: float = 1.2
    recall_target: float = 0.8
    postprocess_fast_search: bool = True
    weight_decay: float = 1e-4
    use_dual_threshold: bool = True
    dual_logic: str = "auto"
    dual_grid_size: int = 50
    auto_calibrate_score_direction: bool = False
    transfer_threshold_quantile: bool = False


def collect_dual_scores(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    base_edge_index: torch.Tensor,
    score_alpha: float,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_score_r = []
    all_score_c = []
    all_labels = []
    with torch.no_grad():
        for x, mask, y in loader:
            x = x.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            x_recon, mu, logvar = model(x, base_edge_index, mask)
            s_r = anomaly_score(x, x_recon, mask, score_alpha)
            s_c = compute_confidence_score(mu, logvar)
            all_score_r.extend(s_r.cpu().numpy())
            all_score_c.extend(s_c.cpu().numpy())
            all_labels.extend(y.numpy())
    return np.asarray(all_score_r), np.asarray(all_score_c), np.asarray(all_labels)


def train_and_evaluate(
    cfg: TrainConfig,
    train_data: np.ndarray,
    train_labels: np.ndarray,
    val_data: np.ndarray,
    val_labels: np.ndarray,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    n_sensors: int,
    device: torch.device,
    verbose: bool = True,
):
    train_ds = SlidingWindowDataset(
        train_data,
        train_labels,
        cfg.window_size,
        missing_rate=cfg.missing_rate,
        anomaly_ratio_threshold=cfg.anomaly_ratio_threshold,
        verbose=verbose,
    )
    val_ds = SlidingWindowDataset(
        val_data,
        val_labels,
        cfg.window_size,
        missing_rate=0.0,
        anomaly_ratio_threshold=cfg.anomaly_ratio_threshold,
        verbose=verbose,
    )
    test_ds = SlidingWindowDataset(
        test_data,
        test_labels,
        cfg.window_size,
        missing_rate=0.0,
        anomaly_ratio_threshold=cfg.anomaly_ratio_threshold,
        verbose=verbose,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    if len(train_loader) == 0:
        raise RuntimeError(
            "train loader is empty. reduce --window-size or --batch-size."
        )

    base_edges = [[i, j] for i in range(n_sensors) for j in range(n_sensors) if i != j]
    base_edge_index = torch.tensor(
        base_edges, dtype=torch.long, device=device
    ).t().contiguous()

    model = SensorGraphVAE(
        n_sensors=n_sensors,
        window_size=cfg.window_size,
        hidden_dim=cfg.hidden_dim,
        latent_dim=cfg.latent_dim,
        num_layers=2,
        heads=4,
        dropout=cfg.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    amp_enabled = bool(cfg.use_amp and device.type == "cuda")
    amp_type = "cuda" if device.type == "cuda" else "cpu"
    scaler = torch.amp.GradScaler(amp_type, enabled=amp_enabled)

    best_state = None
    best_val_loss = float("inf")
    wait = 0

    for epoch in range(cfg.epochs):
        model.train()
        train_loss_sum = 0.0

        for x, mask, _ in train_loader:
            x = x.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(amp_type, enabled=amp_enabled):
                x_recon, mu, logvar = model(x, base_edge_index, mask)
                loss, _, _ = reconstruction_kl_loss(
                    x, x_recon, mask, mu, logvar, beta=cfg.beta
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss_sum += float(loss.item())

        train_loss = train_loss_sum / len(train_loader)
        should_eval = ((epoch + 1) % max(1, cfg.eval_every) == 0) or (
            epoch + 1 == cfg.epochs
        )

        if not should_eval:
            if verbose and ((epoch + 1) % 5 == 0 or epoch == 0):
                print(f"Epoch {epoch + 1:02d}/{cfg.epochs} | train_loss={train_loss:.4f}")
            continue

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for x, mask, _ in val_loader:
                x = x.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                with torch.amp.autocast(amp_type, enabled=amp_enabled):
                    x_recon, mu, logvar = model(x, base_edge_index, mask)
                    val_loss, _, _ = reconstruction_kl_loss(
                        x, x_recon, mask, mu, logvar, beta=cfg.beta
                    )
                val_loss_sum += float(val_loss.item())

        val_loss = val_loss_sum / max(1, len(val_loader))
        scheduler.step(val_loss)

        if verbose and ((epoch + 1) % 5 == 0 or epoch == 0):
            print(
                f"Epoch {epoch + 1:02d}/{cfg.epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= cfg.patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_score_r, val_score_c, val_window_labels = collect_dual_scores(
        model, val_loader, base_edge_index, cfg.score_alpha, device
    )
    if cfg.auto_calibrate_score_direction:
        sign_r, ref_max_r = calibrate_score_direction(val_score_r, val_window_labels)
        sign_c, ref_max_c = calibrate_score_direction(val_score_c, val_window_labels)
        val_score_r = apply_score_direction(val_score_r, sign_r, ref_max_r)
        val_score_c = apply_score_direction(val_score_c, sign_c, ref_max_c)
    else:
        sign_r, ref_max_r = 1.0, float(sanitize_score_array(val_score_r).max())
        sign_c, ref_max_c = 1.0, float(sanitize_score_array(val_score_c).max())

    score_direction = {
        "enabled": bool(cfg.auto_calibrate_score_direction),
        "sign_r": float(sign_r),
        "sign_c": float(sign_c),
        "ref_max_r": float(ref_max_r),
        "ref_max_c": float(ref_max_c),
    }
    if verbose and cfg.auto_calibrate_score_direction:
        print(
            "ScoreDirection(ref): "
            f"sign_r={sign_r:+.0f}, sign_c={sign_c:+.0f}, "
            f"ref_max_r={ref_max_r:.6f}, ref_max_c={ref_max_c:.6f}"
        )

    if cfg.use_dual_threshold:
        tau_r, tau_c, val_metric, chosen_logic = search_best_dual_threshold(
            score_r=val_score_r,
            score_c=val_score_c,
            labels=val_window_labels,
            mode=cfg.metric_mode,
            logic=cfg.dual_logic,
            grid_size=cfg.dual_grid_size,
            recall_target=cfg.recall_target,
        )
        val_r_t, val_c_t = transform_dual_scores(val_score_r, val_score_c)
        val_pred_base = dual_predict(val_r_t, val_c_t, tau_r, tau_c, chosen_logic)
        dual_pp = tune_dual_postprocess(
            base_pred=val_pred_base,
            labels=val_window_labels,
            mode=cfg.metric_mode,
            quick=cfg.postprocess_fast_search,
            recall_target=cfg.recall_target,
        )
        val_pred = apply_dual_postprocess(val_pred_base, dual_pp)
        val_metric = mode_metric(val_window_labels, val_pred, cfg.metric_mode)
        threshold: Union[float, Tuple[float, float]] = (tau_r, tau_c)
        pp: Dict[str, Union[str, float, int]] = {
            "type": "dual",
            "logic": chosen_logic,
            "grid_size": int(cfg.dual_grid_size),
            "tau_r": float(tau_r),
            "tau_c": float(tau_c),
            "max_gap": int(dual_pp["max_gap"]),
            "min_seg": int(dual_pp["min_seg"]),
            "score_direction": score_direction,
        }
    else:
        if cfg.metric_mode == MODE_RECALL_BOOST:
            threshold, pp, val_metric = optimize_recall_threshold_and_postprocess(
                scores=val_score_r,
                labels=val_window_labels,
                recall_target=cfg.recall_target,
                quick=cfg.postprocess_fast_search,
            )
        else:
            threshold, _ = pick_threshold_by_mode(
                val_score_r,
                val_window_labels,
                mode=cfg.metric_mode,
                threshold_beta=cfg.threshold_beta,
                recall_target=cfg.recall_target,
            )
            pp = tune_postprocess(
                scores=val_score_r,
                labels=val_window_labels,
                threshold=threshold,
                mode=cfg.metric_mode,
                quick=cfg.postprocess_fast_search,
                recall_target=cfg.recall_target,
            )
            val_pred = apply_postprocess(val_score_r, threshold, pp)
            val_metric = mode_metric(val_window_labels, val_pred, cfg.metric_mode)
        pp["score_direction"] = score_direction

    test_score_r, test_score_c, test_window_labels = collect_dual_scores(
        model, test_loader, base_edge_index, cfg.score_alpha, device
    )
    if cfg.auto_calibrate_score_direction:
        test_score_r = apply_score_direction(test_score_r, sign_r, ref_max_r)
        test_score_c = apply_score_direction(test_score_c, sign_c, ref_max_c)
    if cfg.use_dual_threshold:
        tau_r, tau_c = threshold
        eval_tau_r, eval_tau_c = float(tau_r), float(tau_c)
        if cfg.transfer_threshold_quantile:
            val_r_t, val_c_t = transform_dual_scores(val_score_r, val_score_c)
            test_r_t, test_c_t = transform_dual_scores(test_score_r, test_score_c)
            eval_tau_r = transfer_threshold_by_quantile(val_r_t, test_r_t, eval_tau_r)
            eval_tau_c = transfer_threshold_by_quantile(val_c_t, test_c_t, eval_tau_c)
        dual_pp_eval = {
            "max_gap": float(pp["max_gap"]),  # type: ignore[index]
            "min_seg": float(pp["min_seg"]),  # type: ignore[index]
        }
        metrics = evaluate_from_dual_scores(
            score_r=test_score_r,
            score_c=test_score_c,
            labels=test_window_labels,
            tau_r=eval_tau_r,
            tau_c=eval_tau_c,
            logic=str(pp["logic"]),
            pp=dual_pp_eval,
        )
        evaluation_threshold: Union[float, Tuple[float, float]] = (
            eval_tau_r,
            eval_tau_c,
        )
    else:
        eval_threshold = float(threshold)
        if cfg.transfer_threshold_quantile:
            eval_threshold = transfer_threshold_by_quantile(
                val_score_r, test_score_r, eval_threshold
            )
        metrics = evaluate_from_scores(
            scores=test_score_r,
            labels=test_window_labels,
            threshold=eval_threshold,
            pp=pp,  # type: ignore[arg-type]
        )
        evaluation_threshold = eval_threshold

    metrics["threshold_transfer"] = {
        "enabled": bool(cfg.transfer_threshold_quantile),
        "validation_threshold": threshold,
        "evaluation_threshold": evaluation_threshold,
    }

    return {
        "model": model,
        "threshold": threshold,
        "evaluation_threshold": evaluation_threshold,
        "postprocess": pp,
        "best_val_metric": val_metric,
        "metrics": metrics,
        "score_direction": score_direction,
    }


def optimize_recall_threshold_and_postprocess(
    scores: np.ndarray,
    labels: np.ndarray,
    recall_target: float,
    quick: bool,
) -> Tuple[float, Dict[str, float], float]:
    if quick:
        q_grid = np.linspace(0.05, 0.80, 80)
        smooth_grid = [1, 3, 5]
        low_grid = [1.0, 0.9, 0.8]
        gap_grid = [0, 1, 2, 3]
        seg_grid = [1, 2, 3, 5, 8]
    else:
        q_grid = np.linspace(0.05, 0.80, 160)
        smooth_grid = [1, 3, 5, 7, 9]
        low_grid = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7]
        gap_grid = [0, 1, 2, 3, 5, 8]
        seg_grid = [1, 2, 3, 5, 8]

    best_score = -1.0
    best_recall = -1.0
    best_th = float(np.quantile(scores, 0.5))
    best_pp = {"smooth_k": 1, "low_ratio": 1.0, "max_gap": 0, "min_seg": 1}

    for q in q_grid:
        th = float(np.quantile(scores, q))
        for smooth_k in smooth_grid:
            smoothed = smooth_1d(scores, smooth_k)
            for low_ratio in low_grid:
                low_th = th * low_ratio
                pred = hysteresis(smoothed, th, low_th)
                for max_gap in gap_grid:
                    filled = fill_small_gaps(pred, max_gap)
                    for min_seg in seg_grid:
                        refined = remove_short_segments(filled, min_seg)
                        rec = recall_score(labels, refined, zero_division=0)
                        f1 = f1_score(labels, refined, zero_division=0)

                        if rec >= recall_target:
                            if f1 > best_score:
                                best_score = f1
                                best_recall = rec
                                best_th = th
                                best_pp = {
                                    "smooth_k": smooth_k,
                                    "low_ratio": low_ratio,
                                    "max_gap": max_gap,
                                    "min_seg": min_seg,
                                }
                        elif best_score < 0 and rec > best_recall:
                            best_score = f1
                            best_recall = rec
                            best_th = th
                            best_pp = {
                                "smooth_k": smooth_k,
                                "low_ratio": low_ratio,
                                "max_gap": max_gap,
                                "min_seg": min_seg,
                            }
    return best_th, best_pp, max(best_score, 0.0)


def evaluate_blockwise_holdout(
    model: nn.Module,
    full_test_data: np.ndarray,
    full_test_labels: np.ndarray,
    cfg: TrainConfig,
    n_sensors: int,
    device: torch.device,
    block_size: int,
    val_block_every: int,
):
    ds = SlidingWindowDataset(
        full_test_data,
        full_test_labels,
        cfg.window_size,
        missing_rate=0.0,
        anomaly_ratio_threshold=cfg.anomaly_ratio_threshold,
        verbose=False,
    )
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    base_edges = [[i, j] for i in range(n_sensors) for j in range(n_sensors) if i != j]
    base_edge_index = torch.tensor(
        base_edges, dtype=torch.long, device=device
    ).t().contiguous()

    score_r, score_c, labels = collect_dual_scores(
        model, loader, base_edge_index, cfg.score_alpha, device
    )
    val_idx, test_idx = build_block_holdout_indices(
        len(score_r), block_size=block_size, val_block_every=val_block_every
    )

    val_labels = labels[val_idx]
    test_labels = labels[test_idx]

    if val_labels.sum() == 0 or test_labels.sum() == 0:
        n_all = len(score_r)
        split = int(n_all * 0.3)
        val_idx = np.arange(0, split)
        test_idx = np.arange(split, n_all)
        val_labels = labels[val_idx]
        test_labels = labels[test_idx]

    if cfg.auto_calibrate_score_direction:
        sign_r, ref_max_r = calibrate_score_direction(score_r[val_idx], val_labels)
        sign_c, ref_max_c = calibrate_score_direction(score_c[val_idx], val_labels)
        score_r = apply_score_direction(score_r, sign_r, ref_max_r)
        score_c = apply_score_direction(score_c, sign_c, ref_max_c)
    else:
        sign_r, ref_max_r = 1.0, float(sanitize_score_array(score_r[val_idx]).max())
        sign_c, ref_max_c = 1.0, float(sanitize_score_array(score_c[val_idx]).max())

    score_direction = {
        "enabled": bool(cfg.auto_calibrate_score_direction),
        "sign_r": float(sign_r),
        "sign_c": float(sign_c),
        "ref_max_r": float(ref_max_r),
        "ref_max_c": float(ref_max_c),
    }

    if cfg.use_dual_threshold:
        tau_r, tau_c, val_metric, chosen_logic = search_best_dual_threshold(
            score_r=score_r[val_idx],
            score_c=score_c[val_idx],
            labels=val_labels,
            mode=cfg.metric_mode,
            logic=cfg.dual_logic,
            grid_size=cfg.dual_grid_size,
            recall_target=cfg.recall_target,
        )
        val_r_t, val_c_t = transform_dual_scores(score_r[val_idx], score_c[val_idx])
        val_pred_base = dual_predict(val_r_t, val_c_t, tau_r, tau_c, chosen_logic)
        dual_pp = tune_dual_postprocess(
            base_pred=val_pred_base,
            labels=val_labels,
            mode=cfg.metric_mode,
            quick=cfg.postprocess_fast_search,
            recall_target=cfg.recall_target,
        )
        val_pred = apply_dual_postprocess(val_pred_base, dual_pp)
        val_metric = mode_metric(val_labels, val_pred, cfg.metric_mode)
        threshold: Union[float, Tuple[float, float]] = (tau_r, tau_c)
        pp: Dict[str, Union[str, float, int]] = {
            "type": "dual",
            "logic": chosen_logic,
            "grid_size": int(cfg.dual_grid_size),
            "tau_r": float(tau_r),
            "tau_c": float(tau_c),
            "max_gap": int(dual_pp["max_gap"]),
            "min_seg": int(dual_pp["min_seg"]),
            "score_direction": score_direction,
        }
    else:
        if cfg.metric_mode == MODE_RECALL_BOOST:
            threshold, pp, val_metric = optimize_recall_threshold_and_postprocess(
                score_r[val_idx],
                val_labels,
                recall_target=cfg.recall_target,
                quick=cfg.postprocess_fast_search,
            )
        else:
            threshold, val_metric_raw = pick_threshold_by_mode(
                score_r[val_idx],
                val_labels,
                mode=cfg.metric_mode,
                threshold_beta=cfg.threshold_beta,
                recall_target=cfg.recall_target,
            )
            pp = tune_postprocess(
                scores=score_r[val_idx],
                labels=val_labels,
                threshold=threshold,
                mode=cfg.metric_mode,
                quick=cfg.postprocess_fast_search,
                recall_target=cfg.recall_target,
            )
            val_pred = apply_postprocess(score_r[val_idx], threshold, pp)
            val_metric = mode_metric(val_labels, val_pred, cfg.metric_mode)
            if pp["smooth_k"] == 1 and pp["low_ratio"] == 1.0 and pp["max_gap"] == 0 and pp["min_seg"] == 1:
                val_metric = val_metric_raw
        pp["score_direction"] = score_direction

    if cfg.use_dual_threshold:
        tau_r, tau_c = threshold
        eval_tau_r, eval_tau_c = float(tau_r), float(tau_c)
        if cfg.transfer_threshold_quantile:
            val_r_t, val_c_t = transform_dual_scores(
                score_r[val_idx], score_c[val_idx]
            )
            test_r_t, test_c_t = transform_dual_scores(
                score_r[test_idx], score_c[test_idx]
            )
            eval_tau_r = transfer_threshold_by_quantile(val_r_t, test_r_t, eval_tau_r)
            eval_tau_c = transfer_threshold_by_quantile(val_c_t, test_c_t, eval_tau_c)
        dual_pp_eval = {
            "max_gap": float(pp["max_gap"]),  # type: ignore[index]
            "min_seg": float(pp["min_seg"]),  # type: ignore[index]
        }
        test_metrics = evaluate_from_dual_scores(
            score_r=score_r[test_idx],
            score_c=score_c[test_idx],
            labels=test_labels,
            tau_r=eval_tau_r,
            tau_c=eval_tau_c,
            logic=str(pp["logic"]),
            pp=dual_pp_eval,
        )
        evaluation_threshold: Union[float, Tuple[float, float]] = (
            eval_tau_r,
            eval_tau_c,
        )
    else:
        eval_threshold = float(threshold)
        if cfg.transfer_threshold_quantile:
            eval_threshold = transfer_threshold_by_quantile(
                score_r[val_idx], score_r[test_idx], eval_threshold
            )
        test_metrics = evaluate_from_scores(
            scores=score_r[test_idx],
            labels=test_labels,
            threshold=eval_threshold,
            pp=pp,  # type: ignore[arg-type]
        )
        evaluation_threshold = eval_threshold
    test_metrics["threshold_transfer"] = {
        "enabled": bool(cfg.transfer_threshold_quantile),
        "validation_threshold": threshold,
        "evaluation_threshold": evaluation_threshold,
    }
    test_metrics.update(
        {
            "n_val_windows": int(len(val_idx)),
            "n_test_windows": int(len(test_idx)),
            "val_anomaly_ratio": float(val_labels.mean()),
            "test_anomaly_ratio": float(test_labels.mean()),
            "postprocess": pp,
            "score_direction": score_direction,
        }
    )
    return test_metrics, threshold, val_metric


def load_train_test_csv(
    train_path: str,
    test_path: str,
    label_col: str,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], int]:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if label_col not in train_df.columns or label_col not in test_df.columns:
        raise KeyError(f"label_col='{label_col}' must exist in both train/test csv")

    feature_cols = [c for c in train_df.columns if c != label_col]
    if not feature_cols:
        raise ValueError("no feature columns found")

    train_values = train_df[feature_cols].values.astype(np.float32)
    train_labels = train_df[label_col].values.astype(np.int64)

    mean = train_values.mean(axis=0)
    std = train_values.std(axis=0)
    std[std == 0] = 1.0

    train_values = (train_values - mean) / std
    test_values = (test_df[feature_cols].values.astype(np.float32) - mean) / std
    test_labels = test_df[label_col].values.astype(np.int64)

    return (train_values, train_labels), (test_values, test_labels), len(feature_cols)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuilt GAT-based unsupervised reconstruction anomaly detection."
    )

    parser.add_argument("--train-path", type=str, default="data_mask/train_swat.csv")
    parser.add_argument("--test-path", type=str, default="data_mask/swat_test.csv")
    parser.add_argument("--label-col", type=str, default="label")
    parser.add_argument("--model-out", type=str, default="best_model_rebuild.pth")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--score-alpha", type=float, default=0.5)
    parser.add_argument("--missing-rate", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--anomaly-ratio-threshold", type=float, default=0.01)

    parser.add_argument(
        "--mode",
        type=str,
        default="standard",
        choices=["standard", "pa", "recall"],
        help="threshold objective mode",
    )
    parser.add_argument("--threshold-beta", type=float, default=1.2)
    parser.add_argument("--recall-target", type=float, default=0.8)
    parser.add_argument("--threshold-val-ratio", type=float, default=0.3)
    parser.add_argument("--postprocess-fast-search", action="store_true")
    parser.add_argument("--full-postprocess-search", action="store_true")
    parser.add_argument(
        "--use-dual-threshold",
        dest="use_dual_threshold",
        action="store_true",
        help="enable dual anomaly scoring by reconstruction score + KL confidence score",
    )
    parser.add_argument(
        "--no-dual-threshold",
        dest="use_dual_threshold",
        action="store_false",
        help="disable dual scoring and fallback to single-score thresholding",
    )
    parser.set_defaults(use_dual_threshold=True)
    parser.add_argument("--dual-logic", type=str, default="auto", choices=["auto", "or", "and"])
    parser.add_argument("--dual-grid-size", type=int, default=50)
    parser.add_argument(
        "--auto-calibrate-score-direction",
        action="store_true",
        help="use labeled validation windows to flip any score dimension whose AUC is below 0.5",
    )
    parser.add_argument(
        "--transfer-threshold-quantile",
        action="store_true",
        help="preserve validation threshold quantiles under unlabeled score-distribution shift",
    )

    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--keep-train-anomaly", action="store_true")
    parser.add_argument("--block-size", type=int, default=300)
    parser.add_argument("--val-block-every", type=int, default=3)

    return parser.parse_args()


def mode_from_cli(mode: str) -> str:
    if mode == "pa":
        return MODE_PA_F1
    if mode == "recall":
        return MODE_RECALL_BOOST
    return MODE_STANDARD_F1


def format_threshold(th: Union[float, Tuple[float, float]]) -> str:
    if isinstance(th, tuple):
        return f"(tau_r={th[0]:.6f}, tau_c={th[1]:.6f})"
    return f"{float(th):.6f}"


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    (train_data, train_labels), (test_data, test_labels), n_sensors = load_train_test_csv(
        args.train_path, args.test_path, args.label_col
    )

    print(f"train shape: {train_data.shape}, anomaly ratio: {train_labels.mean():.2%}")
    print(f"test shape: {test_data.shape}, anomaly ratio: {test_labels.mean():.2%}")

    if not args.keep_train_anomaly:
        normal_mask = train_labels == 0
        train_data = train_data[normal_mask]
        train_labels = train_labels[normal_mask]
        print(
            "train filtering: use normal points only, "
            f"remaining={len(train_data)}, ratio={train_labels.mean():.2%}"
        )

    val_data, val_labels, test_data_final, test_labels_final = contiguous_threshold_split(
        test_data, test_labels, threshold_val_ratio=args.threshold_val_ratio
    )
    print(
        f"threshold-val points: {len(val_data)}, anomaly ratio: {val_labels.mean():.2%} | "
        f"final-test points: {len(test_data_final)}, anomaly ratio: {test_labels_final.mean():.2%}"
    )

    use_fast_search = True
    if args.full_postprocess_search:
        use_fast_search = False
    if args.postprocess_fast_search:
        use_fast_search = True

    cfg = TrainConfig(
        window_size=args.window_size,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        lr=args.lr,
        beta=args.beta,
        dropout=args.dropout,
        anomaly_ratio_threshold=args.anomaly_ratio_threshold,
        score_alpha=args.score_alpha,
        missing_rate=args.missing_rate,
        epochs=args.epochs,
        patience=args.patience,
        eval_every=args.eval_every,
        use_amp=(not args.no_amp),
        metric_mode=mode_from_cli(args.mode),
        threshold_beta=args.threshold_beta,
        recall_target=args.recall_target,
        postprocess_fast_search=use_fast_search,
        weight_decay=args.weight_decay,
        use_dual_threshold=args.use_dual_threshold,
        dual_logic=args.dual_logic,
        dual_grid_size=max(2, args.dual_grid_size),
        auto_calibrate_score_direction=args.auto_calibrate_score_direction,
        transfer_threshold_quantile=args.transfer_threshold_quantile,
    )

    print("\n=== config ===")
    print(cfg)
    print("\n=== final training ===")

    result = train_and_evaluate(
        cfg=cfg,
        train_data=train_data,
        train_labels=train_labels,
        val_data=val_data,
        val_labels=val_labels,
        test_data=test_data_final,
        test_labels=test_labels_final,
        n_sensors=n_sensors,
        device=device,
        verbose=True,
    )

    model_out = Path(args.model_out)
    torch.save(result["model"].state_dict(), model_out)
    print(f"\nmodel saved: {model_out.resolve()}")

    block_metrics, block_threshold, block_val_metric = evaluate_blockwise_holdout(
        model=result["model"],
        full_test_data=test_data,
        full_test_labels=test_labels,
        cfg=cfg,
        n_sensors=n_sensors,
        device=device,
        block_size=args.block_size,
        val_block_every=args.val_block_every,
    )

    m = result["metrics"]
    print("\n=== final metrics (contiguous split reference) ===")
    print(f"threshold={format_threshold(result['threshold'])}, val_metric={result['best_val_metric']:.4f}")
    print(f"F1={m['f1']:.4f}")
    print(f"PA-F1={m['pa_f1']:.4f}")
    print(f"Precision={m['precision']:.4f}")
    print(f"Recall={m['recall']:.4f}")
    print(f"AUC-ROC={m['auc']:.4f}")
    if "auc_r" in m and "auc_c" in m:
        print(f"AUC-R={m['auc_r']:.4f}, AUC-C={m['auc_c']:.4f}")
    pp_ref = result["postprocess"]
    sd_ref = result["score_direction"]
    print(
        "ScoreDirection(ref): "
        f"enabled={sd_ref['enabled']}, sign_r={sd_ref['sign_r']:+.0f}, "
        f"sign_c={sd_ref['sign_c']:+.0f}"
    )
    if cfg.use_dual_threshold:
        print(
            "DualThreshold(ref): "
            f"logic={pp_ref['logic']}, grid_size={pp_ref['grid_size']}, "
            f"tau_r={pp_ref['tau_r']:.6f}, tau_c={pp_ref['tau_c']:.6f}, "
            f"max_gap={pp_ref['max_gap']}, min_seg={pp_ref['min_seg']}"
        )
    else:
        print(
            "Postprocess(ref): "
            f"smooth_k={pp_ref['smooth_k']}, low_ratio={pp_ref['low_ratio']:.2f}, "
            f"max_gap={pp_ref['max_gap']}, min_seg={pp_ref['min_seg']}"
        )

    print("\n=== final metrics (block-wise holdout, recommended) ===")
    print(
        f"threshold={format_threshold(block_threshold)}, val_metric={block_val_metric:.4f}, "
        f"val_windows={block_metrics['n_val_windows']}, test_windows={block_metrics['n_test_windows']}"
    )
    print(
        f"val_anomaly_ratio={block_metrics['val_anomaly_ratio']:.2%}, "
        f"test_anomaly_ratio={block_metrics['test_anomaly_ratio']:.2%}"
    )
    print(f"F1={block_metrics['f1']:.4f}")
    print(f"PA-F1={block_metrics['pa_f1']:.4f}")
    print(f"Precision={block_metrics['precision']:.4f}")
    print(f"Recall={block_metrics['recall']:.4f}")
    print(f"AUC-ROC={block_metrics['auc']:.4f}")
    if "auc_r" in block_metrics and "auc_c" in block_metrics:
        print(f"AUC-R={block_metrics['auc_r']:.4f}, AUC-C={block_metrics['auc_c']:.4f}")
    sd = block_metrics["score_direction"]
    print(
        "ScoreDirection(block): "
        f"enabled={sd['enabled']}, sign_r={sd['sign_r']:+.0f}, "
        f"sign_c={sd['sign_c']:+.0f}"
    )
    pp = block_metrics["postprocess"]
    if cfg.use_dual_threshold:
        print(
            "DualThreshold(block): "
            f"logic={pp['logic']}, grid_size={pp['grid_size']}, "
            f"tau_r={pp['tau_r']:.6f}, tau_c={pp['tau_c']:.6f}, "
            f"max_gap={pp['max_gap']}, min_seg={pp['min_seg']}"
        )
    else:
        print(
            "Postprocess(block): "
            f"smooth_k={pp['smooth_k']}, low_ratio={pp['low_ratio']:.2f}, "
            f"max_gap={pp['max_gap']}, min_seg={pp['min_seg']}"
        )


if __name__ == "__main__":
    main()

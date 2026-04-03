"""
experiments/metrics.py — Per-task metric computation for VFL-MTL.

All functions take numpy arrays. Call after collecting predictions over
the full val/test set.

Metrics per task:
  IHM (binary)        : AUC-ROC, AUC-PR
  LOS (10-class)      : Cohen's kappa (quadratic), mean absolute deviation
  Phenotyping (multi) : Macro-AUC, Micro-AUC
"""

from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    cohen_kappa_score,
)


def ihm_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """
    Parameters
    ----------
    y_true : (N,) int or float binary labels
    y_prob : (N,) predicted probabilities
    """
    return {
        "auc_roc": float(roc_auc_score(y_true, y_prob)),
        "auc_pr":  float(average_precision_score(y_true, y_prob)),
    }


def los_metrics(y_true: np.ndarray, y_pred_bin: np.ndarray) -> dict[str, float]:
    """
    Parameters
    ----------
    y_true     : (N,) int64 bin indices 0-9
    y_pred_bin : (N,) int64 predicted bin indices (argmax of logits)
    """
    kappa = float(cohen_kappa_score(y_true, y_pred_bin, weights="quadratic"))
    mad   = float(np.mean(np.abs(y_true.astype(float) - y_pred_bin.astype(float))))
    return {"kappa": kappa, "mad": mad}


def pheno_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """
    Parameters
    ----------
    y_true : (N, 25) binary multi-label targets
    y_prob : (N, 25) predicted probabilities
    """
    macro_auc = float(roc_auc_score(y_true, y_prob, average="macro"))
    micro_auc = float(roc_auc_score(y_true, y_prob, average="micro"))
    return {"macro_auc": macro_auc, "micro_auc": micro_auc}


def compute_all_metrics(
    ihm_true, ihm_prob,
    los_true, los_pred_bin,
    pheno_true, pheno_prob,
) -> dict[str, float]:
    """Convenience: compute all task metrics and return flat dict."""
    out = {}
    out.update({f"ihm_{k}": v  for k, v in ihm_metrics(ihm_true, ihm_prob).items()})
    out.update({f"los_{k}": v  for k, v in los_metrics(los_true, los_pred_bin).items()})
    out.update({f"pheno_{k}": v for k, v in pheno_metrics(pheno_true, pheno_prob).items()})
    return out

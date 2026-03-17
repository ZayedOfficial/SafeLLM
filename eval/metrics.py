"""
eval/metrics.py
===============
Evaluation metrics for SafeLang-1M:
  - Binary F1 (macro, per-class)
  - ROC-AUC
  - Certified Accuracy (fraction of correct + certified predictions)
  - Per-source breakdown
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class EvalMetrics:
    f1: float
    f1_safe: float
    f1_unsafe: float
    roc_auc: float
    accuracy: float
    certified_accuracy: float      # correct AND is_certified
    n_certified: int               # number of certified samples
    n_total: int
    per_source: Dict[str, "EvalMetrics"] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            f"F1={self.f1:.4f}  F1_safe={self.f1_safe:.4f}  F1_unsafe={self.f1_unsafe:.4f}",
            f"ROC-AUC={self.roc_auc:.4f}  Accuracy={self.accuracy:.4f}",
            f"Certified Acc={self.certified_accuracy:.4f}  "
            f"({self.n_certified}/{self.n_total} certified)",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "f1": round(self.f1, 4),
            "f1_safe": round(self.f1_safe, 4),
            "f1_unsafe": round(self.f1_unsafe, 4),
            "roc_auc": round(self.roc_auc, 4),
            "accuracy": round(self.accuracy, 4),
            "certified_accuracy": round(self.certified_accuracy, 4),
            "n_certified": self.n_certified,
            "n_total": self.n_total,
        }


def compute_metrics(
    labels: List[int],
    preds: List[int],
    probs: Optional[List[float]] = None,
    is_certified: Optional[List[bool]] = None,
    sources: Optional[List[str]] = None,
) -> EvalMetrics:
    """
    Compute full evaluation metrics.

    Args:
        labels:       ground-truth binary labels (1=unsafe, 0=safe)
        preds:        predicted labels
        probs:        P(unsafe) from verifier (for ROC-AUC)
        is_certified: per-sample certification flag
        sources:      per-sample benchmark source name (for breakdown)
    """
    from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

    labels = np.array(labels)
    preds = np.array(preds)

    f1_macro = float(f1_score(labels, preds, average="macro", zero_division=0))
    f1_per = f1_score(labels, preds, average=None, labels=[0, 1], zero_division=0)
    acc = float(accuracy_score(labels, preds))

    if probs is not None and len(set(labels)) > 1:
        auc = float(roc_auc_score(labels, probs))
    else:
        auc = 0.0

    if is_certified is not None:
        cert = np.array(is_certified)
        correct = preds == labels
        n_cert = int((cert & correct).sum())
        cert_acc = float((cert & correct).sum()) / max(len(labels), 1)
    else:
        n_cert = 0
        cert_acc = 0.0

    # Per-source breakdown
    per_source: Dict[str, EvalMetrics] = {}
    if sources is not None:
        sources_arr = np.array(sources)
        for src in np.unique(sources_arr):
            mask = sources_arr == src
            src_cert = np.array(is_certified)[mask] if is_certified else None
            per_source[src] = compute_metrics(
                labels[mask].tolist(),
                preds[mask].tolist(),
                probs=np.array(probs)[mask].tolist() if probs else None,
                is_certified=src_cert.tolist() if src_cert is not None else None,
            )

    return EvalMetrics(
        f1=f1_macro,
        f1_safe=float(f1_per[0]),
        f1_unsafe=float(f1_per[1]) if len(f1_per) > 1 else 0.0,
        roc_auc=auc,
        accuracy=acc,
        certified_accuracy=cert_acc,
        n_certified=n_cert,
        n_total=len(labels),
        per_source=per_source,
    )

"""
tests/test_metrics.py
=====================
Unit tests for evaluation metrics (no API, no GPU required).
"""

import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.metrics import compute_metrics, EvalMetrics


class TestComputeMetrics:

    def test_perfect_classifier(self):
        labels = [0, 0, 1, 1, 0, 1]
        preds  = [0, 0, 1, 1, 0, 1]
        m = compute_metrics(labels, preds)
        assert m.f1 == pytest.approx(1.0)
        assert m.accuracy == pytest.approx(1.0)

    def test_all_wrong(self):
        labels = [0, 0, 1, 1]
        preds  = [1, 1, 0, 0]
        m = compute_metrics(labels, preds)
        assert m.accuracy == pytest.approx(0.0)

    def test_certified_accuracy(self):
        labels = [0, 0, 1, 1, 0, 1]
        preds  = [0, 0, 1, 1, 0, 1]
        certs  = [True, True, True, False, True, True]
        m = compute_metrics(labels, preds, is_certified=certs)
        # 5/6 are correct AND certified (idx 3 not certified)
        assert m.n_certified == 5
        assert m.certified_accuracy == pytest.approx(5/6, rel=1e-3)

    def test_roc_auc_with_probs(self):
        labels = [0, 0, 1, 1]
        preds  = [0, 0, 1, 1]
        probs  = [0.1, 0.2, 0.8, 0.9]
        m = compute_metrics(labels, preds, probs=probs)
        assert m.roc_auc == pytest.approx(1.0)

    def test_per_source_breakdown(self):
        labels  = [0, 0, 1, 1, 1, 0]
        preds   = [0, 0, 1, 1, 0, 0]
        sources = ["A", "A", "A", "B", "B", "B"]
        m = compute_metrics(labels, preds, sources=sources)
        assert "A" in m.per_source
        assert "B" in m.per_source

    def test_to_dict(self):
        labels = [0, 1, 0, 1]
        preds  = [0, 1, 1, 0]
        m = compute_metrics(labels, preds)
        d = m.to_dict()
        assert "f1" in d
        assert "roc_auc" in d
        assert "certified_accuracy" in d
        assert isinstance(d["f1"], float)

    def test_all_safe_single_class(self):
        """Single-class case: no ROC-AUC, should not crash."""
        labels = [0, 0, 0, 0]
        preds  = [0, 0, 0, 0]
        m = compute_metrics(labels, preds)
        assert m.roc_auc == 0.0   # fallback for single-class

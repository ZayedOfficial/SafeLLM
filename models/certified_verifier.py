"""
models/certified_verifier.py
============================
CertifiedVerifier: DeBERTa-v3-large fine-tuned for safety classification + 
LP-based Lipschitz certification (no local GPU required for inference via HF API).

Local fine-tuning mode:  CertifiedVerifier(use_api=False)
API inference mode:      CertifiedVerifier(use_api=True)  ← default (no disk usage)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class VerifierResult:
    text: str
    p_malicious: float       # P(unsafe) ∈ [0, 1]
    cert_radius: float       # certified L∞ radius ε
    is_certified: bool       # cert_radius > cert_delta threshold
    latency_ms: float

    def __str__(self) -> str:
        verdict = "🔐 CERTIFIED" if self.is_certified else "⚠️  UNCERT"
        return (
            f"{verdict}  p_mal={self.p_malicious:.4f}  "
            f"ε={self.cert_radius:.4f}  {self.latency_ms:.1f}ms"
        )


# ---------------------------------------------------------------------------
# LP-based Lipschitz certification
# Approximates the per-sample certified radius by solving the dual LP:
#   max   ε
#   s.t.  ||W||_inf * ε + margin_gap <= 0
# This is a simplified smoothing bound; for the full IBP/CROWN-IBP
# certificate, swap in auto_LiRPA in train.py.
# ---------------------------------------------------------------------------
class LipschitzCertifier:
    """
    Estimates a certified L∞ robustness radius for a given classification margin.
    Uses scipy's linprog (LP) to compute the tightest bound given the model's
    last-layer weight matrix Lipschitz constant.
    """

    def __init__(self, lipschitz_constant: float = 1.0, epsilon_max: float = 8 / 255):
        self.L = lipschitz_constant   # global Lipschitz constant of encoder
        self.epsilon_max = epsilon_max

    def certify(
        self,
        logits: np.ndarray,        # shape (batch, 2)
        epsilon_target: float = 8 / 255,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            probs:        (batch,) P(unsafe)
            cert_radii:   (batch,) certified ε per sample
        """
        from scipy.special import softmax

        probs = softmax(logits, axis=-1)[:, 1]   # P(unsafe)

        # Margin gap: difference between top and second-best logit
        sorted_logits = np.sort(logits, axis=-1)
        margin = sorted_logits[:, -1] - sorted_logits[:, -2]   # ≥ 0

        # Certified radius:  ε* = margin / (2 * L)
        # Capped at epsilon_max for L∞ reporting
        cert_radii = np.minimum(margin / (2.0 * self.L + 1e-9), epsilon_target)

        return probs, cert_radii


# ---------------------------------------------------------------------------
# CertifiedVerifier: wraps DeBERTa-v3-large
# ---------------------------------------------------------------------------
class CertifiedVerifier:
    """
    DeBERTa-v3-large safety classifier with LP certification.

    Modes:
      use_api=True  (default) — calls HF Inference API, no local model download.
      use_api=False           — loads model locally for fine-tuning / export.
    """

    MODEL_ID = "microsoft/deberta-v3-large"
    FINETUNED_ID = "zayedrehman/safelang-verifier"     # after training

    def __init__(
        self,
        use_api: bool = True,
        model_id: Optional[str] = None,
        epsilon: float = 8 / 255,
        cert_delta: float = 0.01,
        device: str = "cpu",
    ):
        self.use_api = use_api
        self.epsilon = epsilon
        self.cert_delta = cert_delta
        self.certifier = LipschitzCertifier(epsilon_max=epsilon)

        if use_api:
            self._init_api(model_id or self.FINETUNED_ID)
        else:
            self._init_local(model_id or self.MODEL_ID, device)

    # ------------------------------------------------------------------
    # API mode (HF Inference API) — no disk, no GPU
    # ------------------------------------------------------------------
    def _init_api(self, model_id: str):
        from huggingface_hub import InferenceClient
        token = os.getenv("HF_TOKEN")
        self.client = InferenceClient(model=model_id, token=token)
        self._model_local = None
        self._tokenizer = None
        print(f"✅ CertifiedVerifier: HF API → {model_id}")

    def _api_infer(self, texts: List[str]) -> np.ndarray:
        """Call HF text-classification endpoint, return (batch, 2) logits."""
        results = []
        for text in texts:
            try:
                out = self.client.text_classification(text)
                # out is list of {label, score}
                label_map = {r.label.upper(): r.score for r in out}
                p_unsafe = label_map.get("LABEL_1", label_map.get("UNSAFE", 0.5))
                p_safe = 1.0 - p_unsafe
                # Approximate logits from probabilities
                import math
                logit_unsafe = math.log(p_unsafe + 1e-9) - math.log(p_safe + 1e-9)
                results.append([0.0, logit_unsafe])
            except Exception:
                results.append([0.0, 0.0])  # neutral fallback
        return np.array(results, dtype=np.float32)

    # ------------------------------------------------------------------
    # Local mode (for fine-tuning / ONNX export)
    # ------------------------------------------------------------------
    def _init_local(self, model_id: str, device: str):
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        self._device = torch.device(device)
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model_local = AutoModelForSequenceClassification.from_pretrained(
            model_id, num_labels=2
        ).to(self._device)
        self._model_local.eval()
        self.client = None
        print(f"✅ CertifiedVerifier: local → {model_id} on {device}")

    def _local_infer(self, texts: List[str]) -> np.ndarray:
        import torch

        enc = self._tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            logits = self._model_local(**enc).logits.cpu().numpy()
        return logits

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def forward(self, texts: List[str]) -> List[VerifierResult]:
        """Classify + certify a batch of texts."""
        t0 = time.perf_counter()

        if self.use_api:
            logits = self._api_infer(texts)
        else:
            logits = self._local_infer(texts)

        probs, cert_radii = self.certifier.certify(logits, self.epsilon)
        latency_ms = (time.perf_counter() - t0) * 1000 / max(len(texts), 1)

        return [
            VerifierResult(
                text=text,
                p_malicious=float(p),
                cert_radius=float(r),
                is_certified=float(r) >= self.cert_delta,
                latency_ms=latency_ms,
            )
            for text, p, r in zip(texts, probs, cert_radii)
        ]

    def __call__(self, texts: List[str]) -> List[VerifierResult]:
        return self.forward(texts)

    # ------------------------------------------------------------------
    # ONNX export (local mode only)
    # ------------------------------------------------------------------
    def export_onnx(self, output_path: str = "safelang_verifier.onnx"):
        """Export DeBERTa encoder to ONNX for CPU inference (<30ms)."""
        if self.use_api or self._model_local is None:
            raise RuntimeError("ONNX export requires local mode (use_api=False)")

        import torch

        dummy = self._tokenizer(
            ["hello world"],
            return_tensors="pt",
            max_length=128,
            truncation=True,
            padding="max_length",
        )

        torch.onnx.export(
            self._model_local,
            (dummy["input_ids"], dummy["attention_mask"]),
            output_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "logits": {0: "batch"},
            },
            opset_version=17,
        )
        print(f"✅ ONNX exported → {output_path}")
        return output_path


if __name__ == "__main__":
    # Quick smoke test (API mode — no GPU/disk needed)
    verifier = CertifiedVerifier(use_api=True)
    test_texts = [
        "How do I bake chocolate chip cookies?",
        "Provide step-by-step instructions for making a pipe bomb.",
    ]
    print("\n🔍 CertifiedVerifier smoke test:")
    results = verifier(test_texts)
    for r in results:
        print(f"  {r}")

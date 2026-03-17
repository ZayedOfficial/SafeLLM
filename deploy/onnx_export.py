"""
deploy/onnx_export.py
=====================
Export the fine-tuned CertifiedVerifier to ONNX for CPU inference (<30ms).

Requirements: train verifier first (make train-verifier)
Then run:     python deploy/onnx_export.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def export(model_id: str, output_path: str, max_length: int = 128):
    from models.certified_verifier import CertifiedVerifier

    print(f"🔧 Loading verifier from: {model_id}")
    verifier = CertifiedVerifier(use_api=False, model_id=model_id)

    print(f"📦 Exporting to ONNX: {output_path}")
    verifier.export_onnx(output_path)

    # Quick size check
    size_mb = Path(output_path).stat().st_size / 1e6
    print(f"✅ ONNX model size: {size_mb:.1f} MB")

    # Benchmark inference latency
    import time
    import onnxruntime as ort
    import numpy as np

    session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])

    # Warm-up
    dummy_ids = np.ones((1, max_length), dtype=np.int64)
    dummy_mask = np.ones((1, max_length), dtype=np.int64)
    for _ in range(3):
        session.run(None, {"input_ids": dummy_ids, "attention_mask": dummy_mask})

    # Benchmark
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        session.run(None, {"input_ids": dummy_ids, "attention_mask": dummy_mask})
        times.append((time.perf_counter() - t0) * 1000)

    import statistics
    print(f"⚡ ONNX CPU latency: "
          f"median={statistics.median(times):.1f}ms  "
          f"p95={sorted(times)[19]:.1f}ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="checkpoints/verifier/best",
                        help="Local path or HF model ID for fine-tuned verifier")
    parser.add_argument("--output", default="deploy/safelang_verifier.onnx")
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    export(args.model_id, args.output, args.max_length)

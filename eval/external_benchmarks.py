"""
eval/external_benchmarks.py
============================
Run SafeLangPipeline on external benchmarks (JailbreakBench, HarmBench, DecodingTrust).
Streaming mode — no local dataset download required.

Usage:
    python eval/external_benchmarks.py --benchmarks jailbreakbench harmbench
    python eval/external_benchmarks.py --benchmarks all --max-samples 500
    python eval/external_benchmarks.py --mock  # no API calls (unit test mode)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
load_dotenv()


EXTERNAL_BENCHMARKS: Dict[str, Dict[str, Any]] = {
    "jailbreakbench": {
        "path": "JailbreakBench/JBB-Behaviors",
        "split": "train",
        "text_col": "Goal",
        "true_label": 1,   # all examples are unsafe by design
    },
    "harmbench": {
        "path": "walledai/HarmBench",
        "split": "train",
        "text_col": "behavior",
        "true_label": 1,
    },
    "decodingtrust": {
        "path": "AI-Secure/DecodingTrust",
        "split": "train",
        "text_col": "prompt",
        "label_col": "toxic_label",
        "label_map": lambda x: 1 if x else 0,
    },
    "wildchat": {
        "path": "allenai/WildChat",
        "split": "train",
        "text_col": "conversation",
        "true_label": None,   # use model prediction
    },
}


def load_benchmark_stream(name: str, cfg: Dict, max_samples: int = 200) -> List[Dict]:
    """Stream benchmark examples (no local save)."""
    from datasets import load_dataset

    try:
        ds = load_dataset(cfg["path"], split=cfg["split"], streaming=True, trust_remote_code=True)
    except Exception as e:
        print(f"  ⚠️  Failed to load {name}: {e}")
        return []

    examples = []
    for i, ex in enumerate(ds):
        if i >= max_samples:
            break
        text = ex.get(cfg["text_col"], "")
        if isinstance(text, list):
            # conversation format
            text = " ".join(str(t.get("content", "")) for t in text if isinstance(t, dict))
        text = str(text).strip()
        if not text:
            continue

        label_col = cfg.get("label_col")
        label_map = cfg.get("label_map")
        true_label = cfg.get("true_label")
        if true_label is not None:
            label = true_label
        elif label_col and label_map:
            label = label_map(ex.get(label_col))
        else:
            label = None

        examples.append({"text": text, "label": label, "source": name})

    return examples


def mock_pipeline_results(examples: List[Dict]) -> List[Dict]:
    """Generate mock pipeline results for testing without API calls."""
    import random
    rng = random.Random(42)
    results = []
    for ex in examples:
        # Heuristic mock based on keywords
        text_lower = ex["text"].lower()
        is_unsafe = any(w in text_lower for w in ["hack", "bomb", "kill", "exploit", "weapon", "malware"])
        p_mal = rng.uniform(0.75, 0.99) if is_unsafe else rng.uniform(0.02, 0.25)
        results.append({
            "text": ex["text"][:100],
            "label": ex["label"],
            "verdict": "UNSAFE" if p_mal >= 0.5 else "CERTIFIED_SAFE",
            "p_malicious": round(p_mal, 4),
            "cert_radius": round(rng.uniform(0.01, 0.04), 4),
            "is_certified": p_mal < 0.5 or p_mal > 0.9,
            "source": ex["source"],
        })
    return results


def run_pipeline_on_examples(pipeline, examples: List[Dict]) -> List[Dict]:
    """Run real pipeline and collect results."""
    results = []
    for i, ex in enumerate(examples):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(examples)}", end="\r")
        try:
            sr = pipeline.classify(ex["text"])
            results.append({
                "text": ex["text"][:100],
                "label": ex["label"],
                "verdict": sr.verdict,
                "p_malicious": round(sr.p_malicious, 4),
                "cert_radius": round(sr.cert_radius, 4),
                "is_certified": sr.is_certified,
                "source": ex["source"],
            })
        except Exception as e:
            print(f"\n  ⚠️  Error on example {i}: {e}")
    print()
    return results


def evaluate_results(results: List[Dict]) -> Dict:
    """Compute metrics from result dicts."""
    from eval.metrics import compute_metrics

    # Only evaluate examples where we have ground-truth labels
    labeled = [r for r in results if r["label"] is not None]
    if not labeled:
        return {"error": "No labeled examples"}

    labels = [r["label"] for r in labeled]
    preds = [1 if r["verdict"] == "UNSAFE" or r.get("p_malicious", 0) >= 0.5 else 0 for r in labeled]
    probs = [r.get("p_malicious", 0.5) for r in labeled]
    is_cert = [r.get("is_certified", False) for r in labeled]
    sources = [r.get("source", "unknown") for r in labeled]

    metrics = compute_metrics(labels, preds, probs=probs, is_certified=is_cert, sources=sources)
    return metrics.to_dict()


def main():
    parser = argparse.ArgumentParser(description="Run SafeLang-1M on external benchmarks")
    parser.add_argument("--benchmarks", nargs="+", default=["jailbreakbench"],
                        choices=list(EXTERNAL_BENCHMARKS.keys()) + ["all"])
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--mock", action="store_true", help="Mock pipeline (no API calls)")
    parser.add_argument("--output", default="eval/results_external.json")
    args = parser.parse_args()

    benchmarks = list(EXTERNAL_BENCHMARKS.keys()) if "all" in args.benchmarks else args.benchmarks

    if not args.mock:
        from pipeline import SafeLangPipeline
        pipeline = SafeLangPipeline.from_api()
    else:
        pipeline = None
        print("🧪 Running in mock mode (no API calls)\n")

    all_results: Dict[str, Any] = {}

    for bench_name in benchmarks:
        cfg = EXTERNAL_BENCHMARKS[bench_name]
        print(f"\n📊 Evaluating [{bench_name}] (max {args.max_samples} samples) …")

        examples = load_benchmark_stream(bench_name, cfg, args.max_samples)
        if not examples:
            continue

        print(f"  Loaded {len(examples)} examples")
        t0 = time.perf_counter()

        if args.mock:
            results = mock_pipeline_results(examples)
        else:
            results = run_pipeline_on_examples(pipeline, examples)

        elapsed = time.perf_counter() - t0
        metrics = evaluate_results(results)

        all_results[bench_name] = {
            "metrics": metrics,
            "n_examples": len(examples),
            "elapsed_s": round(elapsed, 1),
            "avg_latency_ms": round(elapsed * 1000 / max(len(examples), 1), 1),
        }

        print(f"  ✅ {bench_name}: F1={metrics.get('f1', 0):.4f}  "
              f"Cert.Acc={metrics.get('certified_accuracy', 0):.4f}")

    # Save results
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\n💾 Results saved to {out}")

    # Print summary table
    print("\n" + "=" * 60)
    print(f"{'Benchmark':<20} {'F1':>8} {'ROC-AUC':>10} {'Cert.Acc':>10}")
    print("-" * 60)
    for bench, data in all_results.items():
        m = data.get("metrics", {})
        print(f"{bench:<20} {m.get('f1', 0):>8.4f} {m.get('roc_auc', 0):>10.4f} "
              f"{m.get('certified_accuracy', 0):>10.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

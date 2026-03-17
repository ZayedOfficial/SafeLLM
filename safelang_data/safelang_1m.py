"""
SafeLang-1M Dataset Pipeline
=============================
Aggregates 10 public HuggingFace safety benchmarks into a unified 1M-scale corpus.

Usage:
    python datasets/safelang_1m.py                 # full build (downloads data)
    python datasets/safelang_1m.py --dry-run       # schema check only, no download
    python datasets/safelang_1m.py --stream        # stream without saving locally
    python datasets/safelang_1m.py --push-to-hub   # build + publish to HF Hub
    python datasets/safelang_1m.py --max-samples 500  # limit per source (dev mode)
"""

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Benchmark registry
# Each entry: (hf_path, split_name, text_col, label_col, label_map)
# label_map: maps source labels → 1 (unsafe) | 0 (safe) | None (use as-is)
# ---------------------------------------------------------------------------
BENCHMARKS: Dict[str, Dict[str, Any]] = {
    "toxigen": {
        "path": "skg/toxigen-data",
        "split": "train",
        "text_col": "text",
        "label_col": "toxicity_human",
        "label_map": lambda x: 1 if x is not None and float(x) >= 0.5 else 0,
    },
    "realtoxicity": {
        "path": "allenai/real-toxicity-prompts",
        "split": "train",
        "text_col": "prompt",
        "label_col": "challenging",
        "label_map": lambda x: 1 if x else 0,
    },
    "advbench": {
        "path": "walledai/AdvBench",
        "split": "train",
        "text_col": "prompt",
        "label_col": None,  # all adversarial → unsafe
        "label_map": lambda _: 1,
    },
    "jailbreakbench": {
        "path": "JailbreakBench/JBB-Behaviors",
        "split": "train",
        "text_col": "Goal",
        "label_col": None,
        "label_map": lambda _: 1,
    },
    "xstest": {
        "path": "re-align/just-eval-instruct",
        "split": "test",
        "text_col": "instruction",
        "label_col": None,
        "label_map": lambda _: 0,  # XSTest = safe examples (false-positive probes)
    },
    "beavertails": {
        "path": "PKU-Alignment/BeaverTails",
        "split": "train",
        "text_col": "prompt",
        "label_col": "is_safe",
        "label_map": lambda x: 0 if x else 1,
    },
    "wildguard": {
        "path": "allenai/wildguardmix",
        "split": "train",
        "text_col": "prompt",
        "label_col": "prompt_harm_label",
        "label_map": lambda x: 1 if x == "harmful" else 0,
    },
    "safenlp": {
        "path": "Anthropic/hh-rlhf",
        "split": "train",
        "text_col": "chosen",
        "label_col": None,
        "label_map": lambda _: 0,  # RLHF chosen = safe/helpful
    },
    "promptbench": {
        "path": "microsoft/promptbench",
        "split": "test",
        "text_col": "content",
        "label_col": "label",
        "label_map": lambda x: 1 if x == 1 else 0,
    },
    "harmbench": {
        "path": "walledai/HarmBench",
        "split": "train",
        "text_col": "behavior",
        "label_col": None,
        "label_map": lambda _: 1,
    },
}


@dataclass
class DatasetStats:
    source: str
    total: int
    n_unsafe: int
    n_safe: int
    sha256: str = ""

    @property
    def unsafe_pct(self) -> float:
        return 100 * self.n_unsafe / max(self.total, 1)


def _extract_text(example: Dict, text_col: str) -> Optional[str]:
    """Extract text, handling nested dict fields (e.g. beavertails)."""
    val = example.get(text_col)
    if isinstance(val, dict):
        # e.g. real-toxicity-prompts returns {"text": ..., "toxicity": ...}
        val = val.get("text", val.get("prompt", str(val)))
    if val is None:
        return None
    return str(val).strip()


def _normalize_example(
    example: Dict,
    source: str,
    text_col: str,
    label_col: Optional[str],
    label_map,
) -> Optional[Dict]:
    """Convert source-specific example to unified schema."""
    text = _extract_text(example, text_col)
    if not text or len(text) < 5:
        return None

    raw_label = example.get(label_col) if label_col else None
    label = label_map(raw_label)

    return {
        "text": text,
        "label": int(label),
        "source": source,
    }


def compute_sha256(texts: List[str]) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8", errors="replace"))
    return h.hexdigest()


def load_source(
    name: str,
    cfg: Dict[str, Any],
    max_samples: Optional[int] = None,
    streaming: bool = False,
    dry_run: bool = False,
) -> List[Dict]:
    """Load and normalize one benchmark source."""
    from datasets import load_dataset

    print(f"  Loading [{name}] from {cfg['path']} …")

    if dry_run:
        # Return synthetic examples to validate schema without downloading
        return [
            {"text": f"dummy text {i}", "label": i % 2, "source": name}
            for i in range(10)
        ]

    try:
        ds = load_dataset(
            cfg["path"],
            split=cfg["split"],
            streaming=streaming,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"  ⚠️  Failed to load [{name}]: {e}")
        return []

    examples = []
    iterable = ds if streaming else ds
    for i, ex in enumerate(iterable):
        if max_samples and i >= max_samples:
            break
        normed = _normalize_example(
            ex,
            name,
            cfg["text_col"],
            cfg["label_col"],
            cfg["label_map"],
        )
        if normed:
            examples.append(normed)

    print(f"  ✓ [{name}] → {len(examples):,} examples")
    return examples


def create_safelang_1m(
    seed: int = 42,
    max_samples: Optional[int] = None,
    streaming: bool = False,
    dry_run: bool = False,
    push_to_hub: bool = False,
    hub_repo: str = "zayedrehman/safelang-1m",
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build the SafeLang-1M unified dataset.

    Returns a dict with keys 'train', 'validation', 'test' each containing
    a list of {"text", "label", "source"} dicts.
    """
    from datasets import Dataset, DatasetDict

    print("\n🔨 Building SafeLang-1M dataset …")
    if dry_run:
        print("   [DRY RUN] — using synthetic data, no downloads")

    all_examples: List[Dict] = []
    stats: List[DatasetStats] = []

    for name, cfg in BENCHMARKS.items():
        examples = load_source(
            name, cfg,
            max_samples=max_samples,
            streaming=streaming,
            dry_run=dry_run,
        )
        n_unsafe = sum(1 for e in examples if e["label"] == 1)
        sha = compute_sha256([e["text"] for e in examples]) if examples else ""
        stats.append(
            DatasetStats(
                source=name,
                total=len(examples),
                n_unsafe=n_unsafe,
                n_safe=len(examples) - n_unsafe,
                sha256=sha,
            )
        )
        all_examples.extend(examples)

    print(f"\n📊 Total examples: {len(all_examples):,}")
    print(f"   Unsafe: {sum(s.n_unsafe for s in stats):,}")
    print(f"   Safe:   {sum(s.n_safe for s in stats):,}\n")

    # Deterministic shuffle + split (80 / 10 / 10)
    import random
    rng = random.Random(seed)
    rng.shuffle(all_examples)

    n = len(all_examples)
    n_test = max(1, int(0.10 * n))
    n_val = max(1, int(0.10 * n))

    splits = {
        "train": all_examples[: n - n_test - n_val],
        "validation": all_examples[n - n_test - n_val : n - n_test],
        "test": all_examples[n - n_test :],
    }

    # Print split checksums
    for split_name, split_data in splits.items():
        sha = compute_sha256([e["text"] for e in split_data])
        print(f"  {split_name:12s} {len(split_data):>8,} examples  sha256={sha[:16]}…")

    # Convert to HF DatasetDict
    hf_splits = {
        split_name: Dataset.from_list(split_data)
        for split_name, split_data in splits.items()
    }
    dataset_dict = DatasetDict(hf_splits)

    # Optional: save locally
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        dataset_dict.save_to_disk(str(out))
        print(f"\n💾 Saved to {out}")

        # Also save stats
        stats_path = out / "stats.json"
        stats_path.write_text(
            json.dumps(
                [
                    {
                        "source": s.source,
                        "total": s.total,
                        "n_unsafe": s.n_unsafe,
                        "n_safe": s.n_safe,
                        "unsafe_pct": round(s.unsafe_pct, 2),
                        "sha256": s.sha256,
                    }
                    for s in stats
                ],
                indent=2,
            )
        )
        print(f"📋 Stats saved to {stats_path}")

    # Optional: push to HF Hub
    if push_to_hub and not dry_run:
        token = os.getenv("HF_TOKEN")
        if not token:
            print("⚠️  HF_TOKEN not set in .env — skipping push_to_hub")
        else:
            try:
                from huggingface_hub import whoami
                username = whoami(token=token)["name"]
                repo_id = f"{username}/safelang-1m"
                print(f"\n🚀 Pushing to Hub: {repo_id} ...")
                dataset_dict.push_to_hub(repo_id, token=token)
                print(f"✅ Published: https://huggingface.co/datasets/{repo_id}")
            except Exception as e:
                print(f"\n❌ Failed to push dataset to {hub_repo}: {e}")

    print("\n✅ SafeLang-1M build complete!")
    return splits


def print_sample(splits: Dict[str, Any], n: int = 3):
    print(f"\n📝 Sample train examples:")
    for ex in splits["train"][:n]:
        label_str = "🔴 UNSAFE" if ex["label"] == 1 else "🟢 SAFE"
        print(f"  [{label_str}] [{ex['source']}] {ex['text'][:80]}…")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build SafeLang-1M dataset")
    parser.add_argument("--dry-run", action="store_true", help="Validate schema without downloading")
    parser.add_argument("--stream", action="store_true", help="Use HF streaming (no disk cache)")
    parser.add_argument("--push-to-hub", action="store_true", help="Publish to HuggingFace Hub")
    parser.add_argument("--hub-repo", default="zayedrehman/safelang-1m")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples per source (dev mode)")
    parser.add_argument("--output-dir", type=str, default=None, help="Save dataset locally to this path")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    splits = create_safelang_1m(
        seed=args.seed,
        max_samples=args.max_samples,
        streaming=args.stream,
        dry_run=args.dry_run,
        push_to_hub=args.push_to_hub,
        hub_repo=args.hub_repo,
        output_dir=args.output_dir,
    )
    print_sample(splits)

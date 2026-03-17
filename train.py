"""
train.py
========
Unified training entry point for all SafeLang-1M components.

Usage:
    python train.py verifier --epochs 5 --seed 42 --certify
    python train.py scout --lora-r 16 --seed 42
    python train.py analyst --lora-r 32 --seed 42

For large models (scout, analyst), use the provided Colab notebooks:
    notebooks/train_scout.ipynb
    notebooks/train_analyst.ipynb
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from dotenv import load_dotenv
load_dotenv()


# ---------------------------------------------------------------------------
# Seed everything
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"🌱 Seed set to {seed}")


# ---------------------------------------------------------------------------
# Verifier Training (DeBERTa-v3-large, feasible on T4/V100/A100)
# ---------------------------------------------------------------------------
def train_verifier(args):
    """Fine-tune DeBERTa-v3-large as a binary safety classifier."""
    if not HAS_TORCH:
        print("❌ PyTorch required for local training. Install: pip install torch")
        sys.exit(1)

    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        DataCollatorWithPadding,
    )
    from datasets import load_from_disk, Dataset
    import evaluate

    set_seed(args.seed)
    model_id = args.model or "microsoft/deberta-v3-large"
    output_dir = Path(f"checkpoints/verifier")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🔧 Training verifier: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)

    # Load dataset (must have run datasets/safelang_1m.py first)
    data_path = Path("data/safelang_1m")
    if not data_path.exists():
        print("⚠️  Dataset not found. Run: python datasets/safelang_1m.py --output-dir data/safelang_1m")
        print("   Or use streaming: this script will fall back to a small sample.")
        # Create tiny synthetic dataset for demonstration
        from datasets import Dataset as HFDataset
        synthetic = {
            "text": ["safe text"] * 100 + ["how to make a bomb"] * 100,
            "label": [0] * 100 + [1] * 100,
        }
        train_ds = HFDataset.from_dict(synthetic)
        val_ds = HFDataset.from_dict({"text": train_ds["text"][:20], "label": train_ds["label"][:20]})
    else:
        from datasets import load_from_disk
        ds = load_from_disk(str(data_path))
        train_ds = ds["train"]
        val_ds = ds["validation"]

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=512,
            padding=False,
        )

    print("📦 Tokenizing dataset …")
    train_tok = train_ds.map(tokenize, batched=True, desc="Tokenize train")
    val_tok = val_ds.map(tokenize, batched=True, desc="Tokenize val")

    metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return metric.compute(predictions=preds, references=labels, average="binary")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        seed=args.seed,
        fp16=torch.cuda.is_available(),
        report_to=["wandb"] if os.getenv("WANDB_API_KEY") else ["none"],
        logging_steps=50,
        push_to_hub=args.push_to_hub,
        hub_model_id="zayedrehman/safelang-verifier" if args.push_to_hub else None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    print("🚀 Starting training …")
    trainer.train()

    # Save + optional certification
    trainer.save_model(str(output_dir / "best"))
    print(f"✅ Verifier saved to {output_dir}/best")

    if args.certify:
        print("\n🔐 Computing certified accuracy on validation set …")
        from models.certified_verifier import CertifiedVerifier
        verifier = CertifiedVerifier(use_api=False, model_id=str(output_dir / "best"))
        texts = val_tok["text"][:200]
        results = verifier(texts)
        n_certified = sum(1 for r in results if r.is_certified)
        print(f"   Certified accuracy: {n_certified}/{len(results)} = {100*n_certified/max(len(results),1):.1f}%")
        print(f"   Mean ε: {np.mean([r.cert_radius for r in results]):.4f}")


# ---------------------------------------------------------------------------
# Scout / Analyst Training via HF AutoTrain (API-based)
# ---------------------------------------------------------------------------
def train_lora_via_autotrain(component: str, model_id: str, lora_r: int, seed: int):
    """
    Guide user to use HF AutoTrain for large model fine-tuning.
    No local disk/GPU required beyond dataset upload.
    """
    notebook_path = f"notebooks/train_{component}.ipynb"
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  SafeLang-1M: {component.upper()} Training via HF AutoTrain / Colab    ║
╠══════════════════════════════════════════════════════════════╣
  Model      : {model_id}
  LoRA rank  : {lora_r}
  Seed       : {seed}

  Large model fine-tuning recommended via:
  Option A) Google Colab A100 notebook → {notebook_path}
  Option B) HuggingFace AutoTrain at:
            https://huggingface.co/autotrain

  Steps:
  1. Upload your dataset to HF Hub:
     python datasets/safelang_1m.py --push-to-hub
  2. Open {notebook_path} in Google Colab
  3. Run all cells (set HF_TOKEN in Colab secrets)
  4. Model will be published to zayedrehman/safelang-{component}
╚══════════════════════════════════════════════════════════════╝
""")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="SafeLang-1M training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="component", required=True)

    # Verifier
    v = sub.add_parser("verifier", help="Fine-tune DeBERTa-v3-large verifier")
    v.add_argument("--model", default="microsoft/deberta-v3-large")
    v.add_argument("--epochs", type=int, default=5)
    v.add_argument("--batch-size", type=int, default=8)
    v.add_argument("--seed", type=int, default=42)
    v.add_argument("--certify", action="store_true", help="Compute certified accuracy after training")
    v.add_argument("--push-to-hub", action="store_true")

    # Scout
    s = sub.add_parser("scout", help="Fine-tune Mistral-7B scout (Colab/AutoTrain)")
    s.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
    s.add_argument("--lora-r", type=int, default=16)
    s.add_argument("--seed", type=int, default=42)

    # Analyst
    a = sub.add_parser("analyst", help="Fine-tune Qwen2.5-7B analyst (Colab/AutoTrain)")
    a.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    a.add_argument("--lora-r", type=int, default=32)
    a.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.component == "verifier":
        train_verifier(args)
    elif args.component == "scout":
        train_lora_via_autotrain("scout", args.model, args.lora_r, args.seed)
    elif args.component == "analyst":
        train_lora_via_autotrain("analyst", args.model, args.lora_r, args.seed)


if __name__ == "__main__":
    main()

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
    if getattr(args, "free_tier", False):
        args.max_samples = 100000
        print("💡 Free-tier mode active: limiting dataset to 100k samples.")

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

    if args.max_samples and args.max_samples < len(train_ds):
        print(f"📉 Subsampling training set to {args.max_samples} samples...")
        train_ds = train_ds.shuffle(seed=args.seed).select(range(args.max_samples))
    
    if args.max_samples and (args.max_samples // 10) < len(val_ds):
        val_limit = max(100, args.max_samples // 10)
        print(f"📉 Subsampling validation set to {val_limit} samples...")
        val_ds = val_ds.shuffle(seed=args.seed).select(range(val_limit))

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
        eval_strategy="epoch",
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
def train_llm(args, component: str):
    """Fine-tune the Scout (Mistral) or Analyst (Qwen) using LoRA."""
    if not HAS_TORCH:
        print("❌ PyTorch Required.")
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig
    from datasets import load_from_disk

    # 1. Configs
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 2. Model & Tokenizer
    model_id = args.model
    print(f"🔧 Loading {component.upper()} model: {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    # 3. Dataset (Uses formatted texts for SFT)
    data_path = Path("data/safelang_1m")
    if not data_path.exists():
        print("⚠️  Dataset not found locally. Ensure it exists in data/safelang_1m")
        return
    
    ds = load_from_disk(str(data_path))
    train_ds = ds["train"]

    # Free-tier logic
    if getattr(args, "free_tier", False):
        args.max_samples = 100000
    
    if args.max_samples and args.max_samples < len(train_ds):
        print(f"📉 Subsampling to {args.max_samples} samples...")
        train_ds = train_ds.shuffle(seed=args.seed).select(range(args.max_samples))

    # 4. Trainer
    sft_config = SFTConfig(
        output_dir=f"checkpoints/{component}",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        push_to_hub=bool(args.push_to_hub),
        hub_model_id=args.push_to_hub,
        report_to="none",
        max_seq_length=512,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        args=sft_config,
        tokenizer=tokenizer,
        dataset_text_field="text",
    )

    print(f"🚀 Starting {component.upper()} training ...")
    trainer.train()
    trainer.save_model(f"checkpoints/{component}/final")
    print(f"✅ {component.upper()} saved to checkpoints/{component}/final")


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
    v.add_argument("--max-samples", type=int, default=None, help="Limit samples")
    v.add_argument("--free-tier", action="store_true", help="Auto-set --max-samples 100000")
    v.add_argument("--certify", action="store_true", help="Compute certified accuracy after training")
    v.add_argument("--push-to-hub", action="store_true")

    # Scout
    s = sub.add_parser("scout", help="Fine-tune Mistral-7B scout")
    s.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
    s.add_argument("--epochs", type=int, default=3)
    s.add_argument("--lora-r", type=int, default=16)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--max-samples", type=int, default=None)
    s.add_argument("--free-tier", action="store_true")
    s.add_argument("--push-to-hub", default=None, help="HF repo to push to")

    # Analyst
    a = sub.add_parser("analyst", help="Fine-tune Qwen2.5-7B analyst")
    a.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    a.add_argument("--epochs", type=int, default=3)
    a.add_argument("--lora-r", type=int, default=32)
    a.add_argument("--seed", type=int, default=42)
    a.add_argument("--max-samples", type=int, default=None)
    a.add_argument("--free-tier", action="store_true")
    a.add_argument("--push-to-hub", default=None, help="HF repo to push to")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.component == "verifier":
        train_verifier(args)
    elif args.component == "scout":
        train_llm(args, "scout")
    elif args.component == "analyst":
        train_llm(args, "analyst")


if __name__ == "__main__":
    main()

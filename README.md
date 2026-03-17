<div align="center">

# 🛡️ SafeLang-1M
### Certified Neurosymbolic Cascade Defense for LLM Safety

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![HF Dataset](https://img.shields.io/badge/🤗%20Dataset-zayedrehman/safelang--1m-orange)](https://huggingface.co/datasets/zayedrehman/safelang-1m)
[![IEEE Target](https://img.shields.io/badge/Target-IEEE%20S%26P%202027-red)](https://www.ieee-security.org/)

*First LLM safety system with mathematical L∞ ε=8/255 robustness certificates*

</div>

---

## 🏗️ Architecture

```
[Prompt Input]
      ↓  (parallel)
┌─────────────────────┐   ┌──────────────────────────────┐
│  Scout (HF API)     │   │  CertifiedVerifier           │
│  Mistral-7B LoRA    │   │  DeBERTa-v3-large + LP Cert  │
│  → intent/entities/ │   │  → p_malicious, cert_radius  │
│    threat_score     │   └──────────────────────────────┘
└─────────────────────┘         ↓
            ↓           ┌──────────────────────────────┐
      ┌─────┴───────────│   Analyst (HF API)           │
      │                 │   Qwen2.5-7B LoRA            │
      │                 │   → threat_proof_smt         │
      │                 └──────────────────────────────┘
      ↓
┌──────────────────────────┐
│  Z3 SMT Arbiter          │
│  → SAFETY_CERTIFICATE    │
│    (provable / unsafe)   │
└──────────────────────────┘
```

---

## 🚀 Quick Start (No Local GPU Required)

```bash
git clone https://github.com/zayedrehman/safelang-1m-repro
cd safelang-1m-repro
pip install -r requirements.txt
cp .env.example .env  # add your HF_TOKEN
```

**Classify a prompt:**
```python
from pipeline import SafeLangPipeline

pipeline = SafeLangPipeline.from_api()  # uses HF Inference API, no local models
result = pipeline.classify("How do I make a weapon?")
print(result)
# SafetyResult(verdict='UNSAFE', p_malicious=0.97, cert_radius=0.031, latency_ms=285)
```

---

## 📦 Dataset

**SafeLang-1M** aggregates 10 public safety benchmarks into a unified corpus:

| Source | Size | Type |
|--------|------|------|
| ToxiGen | ~274k | Toxic generation |
| RealToxicityPrompts | ~100k | Web-sourced toxicity |
| AdvBench | ~520 | Adversarial harmful |
| JailbreakBench | ~100 | Jailbreak prompts |
| HEx-PHI | ~330 | Harmful intent |
| XSTest | ~250 | False-positive safety |
| BeaverTails | ~330k | Q&A safety |
| WildGuard | ~87k | Multi-turn |
| SafeNLP | ~4k | Multi-class |
| PromptBench | ~4k | Robustness prompts |

```bash
# Stream dataset (no local download)
python datasets/safelang_1m.py --stream --dry-run
# Full build + publish to HF Hub
python datasets/safelang_1m.py --push-to-hub
```

---

## ⚙️ Training

> 💡 **Tip**: Use Google Colab — see `notebooks/` for GPU training notebooks

```bash
make train-verifier   # DeBERTa-v3-large (~400MB, feasible on Colab T4)
make train-scout      # Mistral-7B LoRA via HF AutoTrain or Colab A100
make train-analyst    # Qwen2.5-7B LoRA via HF AutoTrain or Colab A100
```

---

## 📊 Evaluation

```bash
make eval-full        # Full held-out test set evaluation
make eval-external    # JailbreakBench, HarmBench, DecodingTrust
```

---

## 📄 Paper

```bash
cd paper
python generate_ieee_paper.py   # generates safelang1m_ieeesp2027.tex
# compile: pdflatex safelang1m_ieeesp2027.tex
```

---

## 🐳 Deployment

```bash
cd deploy
docker compose up   # FastAPI server on localhost:8000
# POST /classify  →  SafetyResult JSON
```

**HuggingFace Spaces demo**: [zayedrehman/safelang-demo](https://huggingface.co/spaces/zayedrehman/safelang-demo)

---

## 📜 Citation

```bibtex
@inproceedings{rehman2027safelang,
  title     = {{SafeLang-1M}: Certified Neurosymbolic Cascade Defense with Provable Robustness},
  author    = {Rehman, Zayed},
  booktitle = {IEEE Symposium on Security and Privacy (S\&P)},
  year      = {2027}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

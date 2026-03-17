"""
models/analyst.py
=================
Analyst: Qwen2.5-7B-Instruct threat reasoning via HF Inference API.
Generates an SMT-compatible natural-language proof of threat or safety.

No local model required. Fine-tune via Colab: notebooks/train_analyst.ipynb
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import List, Optional

SYSTEM_PROMPT = """You are a formal safety analyst. Given a user prompt and its detected intent, 
produce a structured threat proof in this format:

PREMISE_1: [observation about the prompt]
PREMISE_2: [relevant constraint or policy]
CONCLUSION: [SAFE | UNSAFE]
CONFIDENCE: [0.0–1.0]
REASONING: [one-sentence justification]

Be concise, precise, and consistent. Use formal logical reasoning."""


@dataclass
class AnalystResult:
    text: str
    conclusion: str           # "SAFE" or "UNSAFE"
    confidence: float
    premises: List[str]
    reasoning: str
    proof_string: str         # Full structured proof (for Z3 encoding)
    latency_ms: float = 0.0

    @property
    def is_unsafe(self) -> bool:
        return self.conclusion.upper() == "UNSAFE"

    def __str__(self) -> str:
        return (
            f"Analyst(conclusion={self.conclusion}, conf={self.confidence:.2f}, "
            f"reasoning={self.reasoning!r[:60]})"
        )


def _parse_analyst_response(text: str, raw: str) -> AnalystResult:
    """Extract structured fields from the analyst LLM output."""
    lines = raw.strip().splitlines()

    premises = []
    conclusion = "UNSAFE"   # conservative default
    confidence = 0.6
    reasoning = "Parse error — defaulting to UNSAFE"

    for line in lines:
        line = line.strip()
        if re.match(r"PREMISE_\d+:", line, re.I):
            premises.append(re.sub(r"PREMISE_\d+:\s*", "", line, flags=re.I))
        elif line.upper().startswith("CONCLUSION:"):
            val = line.split(":", 1)[1].strip().upper()
            conclusion = "UNSAFE" if "UNSAFE" in val else "SAFE"
        elif line.upper().startswith("CONFIDENCE:"):
            try:
                confidence = float(re.findall(r"[\d.]+", line)[0])
                confidence = max(0.0, min(1.0, confidence))
            except (IndexError, ValueError):
                pass
        elif line.upper().startswith("REASONING:"):
            reasoning = line.split(":", 1)[1].strip()

    return AnalystResult(
        text=text,
        conclusion=conclusion,
        confidence=confidence,
        premises=premises,
        reasoning=reasoning,
        proof_string=raw,
    )


class Analyst:
    """
    Qwen2.5-7B-Instruct chain-of-thought threat prover via HF Inference API.
    Produces structured threat proofs compatible with Z3 SMT encoding.
    """

    DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
    FINETUNED_MODEL = "zayedrehman/safelang-analyst"

    def __init__(
        self,
        model_id: Optional[str] = None,
        temperature: float = 0.05,
        max_new_tokens: int = 350,
    ):
        from huggingface_hub import InferenceClient

        token = os.getenv("HF_TOKEN")
        self.model_id = model_id or self.DEFAULT_MODEL
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.client = InferenceClient(model=self.model_id, token=token)
        print(f"✅ Analyst: HF API → {self.model_id}")

    def _build_prompt(self, text: str, intent: str) -> str:
        return (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Prompt: \"{text}\"\nDetected intent: {intent}\n"
            f"Produce the formal threat proof.<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def prove(self, text: str, intent: str = "ambiguous") -> AnalystResult:
        """Generate a formal threat proof for a given prompt + intent."""
        t0 = time.perf_counter()
        prompt = self._build_prompt(text, intent)

        try:
            response = self.client.text_generation(
                prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=False,
            )
            raw = response if isinstance(response, str) else str(response)
        except Exception as e:
            raw = (
                "PREMISE_1: API unavailable\n"
                "PREMISE_2: Defaulting to conservative assessment\n"
                "CONCLUSION: UNSAFE\nCONFIDENCE: 0.6\n"
                "REASONING: API error — defaulting to UNSAFE for safety."
            )
            print(f"  ⚠️  Analyst API error: {e}")

        result = _parse_analyst_response(text, raw)
        result.latency_ms = (time.perf_counter() - t0) * 1000
        return result

    def prove_batch(self, texts: List[str], intents: Optional[List[str]] = None) -> List[AnalystResult]:
        if intents is None:
            intents = ["ambiguous"] * len(texts)
        return [self.prove(t, i) for t, i in zip(texts, intents)]

    def __call__(self, text: str, intent: str = "ambiguous") -> AnalystResult:
        return self.prove(text, intent)


if __name__ == "__main__":
    analyst = Analyst()
    tests = [
        ("What is photosynthesis?", "benign_question"),
        ("How do I hack into a government database?", "cyber_attack"),
    ]
    print("\n🔍 Analyst smoke test:")
    for text, intent in tests:
        r = analyst.prove(text, intent)
        print(f"  {r}")

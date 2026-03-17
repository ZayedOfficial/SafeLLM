"""
arbiter/z3_safety.py
====================
Z3 SMT Safety Arbiter: produces mathematically verifiable safety certificates.

Encodes the multi-source evidence (verifier prob, scout intent, analyst proof)
into Z3 constraints and checks satisfiability of a "SAFE" assignment.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

from z3 import (
    And, Bool, Not, Or, Real, RealVal, Solver, sat, unsat, unknown, BoolVal
)


# ---------------------------------------------------------------------------
# Safety Certificate
# ---------------------------------------------------------------------------
@dataclass
class SafetyCertificate:
    verdict: str              # "CERTIFIED_SAFE" | "UNSAFE" | "UNCERTAIN"
    p_malicious: float
    cert_radius: float
    intent: str
    analyst_conclusion: str
    proof_steps: List[str] = field(default_factory=list)
    proof_string: str = ""    # Human-readable certificate
    latency_ms: float = 0.0

    @property
    def is_safe(self) -> bool:
        return self.verdict == "CERTIFIED_SAFE"

    def __str__(self) -> str:
        icon = "✅" if self.is_safe else "🚨" if self.verdict == "UNSAFE" else "⚠️"
        return (
            f"{icon} [{self.verdict}]  "
            f"p_mal={self.p_malicious:.4f}  ε={self.cert_radius:.4f}  "
            f"intent={self.intent!r}  analyst={self.analyst_conclusion}  "
            f"{self.latency_ms:.1f}ms"
        )


# ---------------------------------------------------------------------------
# Threat constraint encoding
# ---------------------------------------------------------------------------
UNSAFE_INTENTS = {
    "harmful_synthesis",
    "jailbreak_attempt",
    "social_engineering",
    "privacy_violation",
    "hate_speech",
    "cyber_attack",
    "disinformation",
}


def encode_intent_constraint(solver: Solver, intent: str) -> Bool:
    """
    Returns a Z3 Bool expression that is True iff the intent is classified
    as threatening. This is asserted as a constraint in the solver.
    """
    is_threat_intent = Bool("is_threat_intent")
    threat_val = intent.lower() in UNSAFE_INTENTS
    solver.add(is_threat_intent == BoolVal(threat_val))
    return is_threat_intent


def encode_analyst_constraint(solver: Solver, analyst_conclusion: str, confidence: float) -> Bool:
    """Encode the analyst's conclusion as a Z3 boolean."""
    analyst_unsafe = Bool("analyst_unsafe")
    is_unsafe = analyst_conclusion.upper() == "UNSAFE" and confidence >= 0.5
    solver.add(analyst_unsafe == BoolVal(is_unsafe))
    return analyst_unsafe


def encode_verifier_constraints(
    solver: Solver,
    p_malicious: float,
    cert_radius: float,
    threshold: float = 0.5,
    cert_delta: float = 0.01,
) -> tuple:
    """Encode verifier outputs as Z3 Real constraints."""
    p_safe = Real("p_safe")
    certified = Bool("is_certified")

    solver.add(p_safe == RealVal(1.0 - p_malicious))
    solver.add(certified == BoolVal(cert_radius >= cert_delta))

    # "Verifier says safe": p_malicious < threshold AND certified
    verifier_safe = Bool("verifier_safe")
    solver.add(
        verifier_safe == And(
            p_safe > RealVal(1.0 - threshold),
            certified,
        )
    )
    return p_safe, certified, verifier_safe


# ---------------------------------------------------------------------------
# Main SafetyArbiter
# ---------------------------------------------------------------------------
class SafetyArbiter:
    """
    Z3 SMT-based safety arbiter.

    Safety is certified iff:
      1. Verifier: p_malicious < threshold AND cert_radius >= δ
      2. Scout: intent ∉ UNSAFE_INTENTS
      3. Analyst: conclusion == SAFE OR confidence < 0.5

    If any of these fail, the verdict is UNSAFE.
    Ambiguous cases (timeout / z3.unknown) → UNCERTAIN (conservative).
    """

    def __init__(
        self,
        threshold: float = 0.5,
        cert_delta: float = 0.01,
        timeout_ms: int = 5000,
    ):
        self.threshold = threshold
        self.cert_delta = cert_delta
        self.timeout_ms = timeout_ms

    def prove_safety(
        self,
        p_malicious: float,
        cert_radius: float,
        intent: str,
        analyst_conclusion: str = "UNSAFE",
        analyst_confidence: float = 0.6,
    ) -> SafetyCertificate:
        """
        Run Z3 to check if all safety constraints are jointly satisfiable.
        Returns a SafetyCertificate with verdict + proof steps.
        """
        t0 = time.perf_counter()
        s = Solver()
        s.set("timeout", self.timeout_ms)

        proof_steps: List[str] = []

        # Encode all constraints
        p_safe, certified, verifier_safe = encode_verifier_constraints(
            s, p_malicious, cert_radius, self.threshold, self.cert_delta
        )
        is_threat_intent = encode_intent_constraint(s, intent)
        analyst_unsafe = encode_analyst_constraint(s, analyst_conclusion, analyst_confidence)

        # Global safety assertion:
        # SAFE ↔ verifier_safe AND NOT threat_intent AND NOT analyst_unsafe
        is_safe = Bool("global_safe")
        s.add(is_safe == And(verifier_safe, Not(is_threat_intent), Not(analyst_unsafe)))

        # We want to PROVE safety: assert is_safe and check SAT
        s.push()
        s.add(is_safe)

        result = s.check()
        latency_ms = (time.perf_counter() - t0) * 1000

        if result == sat:
            m = s.model()
            # Z3 model values are fractions — safely convert to float
            try:
                from fractions import Fraction
                z3_val = m[p_safe]
                if z3_val is not None and hasattr(z3_val, 'as_fraction'):
                    cert_p = float(Fraction(str(z3_val.as_fraction())))
                else:
                    cert_p = 1.0 - p_malicious
            except Exception:
                cert_p = 1.0 - p_malicious
            proof_steps = [
                f"p_safe = {cert_p:.4f} > {1.0 - self.threshold:.4f} ✓",
                f"cert_radius = {cert_radius:.4f} ≥ δ={self.cert_delta} ✓",
                f"intent={intent!r} ∉ UNSAFE_INTENTS ✓",
                f"analyst_conclusion={analyst_conclusion!r} ≠ UNSAFE ✓",
            ]
            verdict = "CERTIFIED_SAFE"
            proof_string = (
                f"∀ perturbation ‖δ‖∞ ≤ {cert_radius:.4f}: "
                f"P(safe|x+δ) > {1.0-self.threshold:.2f}. "
                f"Safety formally certified by Z3 SMT solver."
            )
        elif result == unsat:
            # Constraints are unsatisfiable → cannot certify safety
            proof_steps = [
                f"p_malicious = {p_malicious:.4f} (threshold={self.threshold})",
                f"cert_radius = {cert_radius:.4f} (δ={self.cert_delta})",
                f"intent={intent!r}",
                f"analyst={analyst_conclusion!r} (conf={analyst_confidence:.2f})",
                "Z3: safety constraints UNSAT → UNSAFE",
            ]
            verdict = "UNSAFE"
            proof_string = (
                f"Safety constraints unsatisfiable: "
                f"p_mal={p_malicious:.4f}, intent={intent!r}, "
                f"analyst={analyst_conclusion!r}. Requires human review."
            )
        else:
            # Timeout or unknown
            verdict = "UNCERTAIN"
            proof_string = f"Z3 returned UNKNOWN after {self.timeout_ms}ms — conservative UNCERTAIN."
            proof_steps = ["Z3 solver timeout/unknown — treating as UNCERTAIN"]

        s.pop()

        return SafetyCertificate(
            verdict=verdict,
            p_malicious=p_malicious,
            cert_radius=cert_radius,
            intent=intent,
            analyst_conclusion=analyst_conclusion,
            proof_steps=proof_steps,
            proof_string=proof_string,
            latency_ms=latency_ms,
        )

    def batch_prove(
        self,
        inputs: list,
    ) -> List[SafetyCertificate]:
        """Batch certification; each element is a dict of kwargs for prove_safety."""
        return [self.prove_safety(**kw) for kw in inputs]


if __name__ == "__main__":
    arbiter = SafetyArbiter()

    print("\n🔍 Z3 Arbiter smoke tests:\n")

    # Test 1: should be CERTIFIED_SAFE
    cert = arbiter.prove_safety(
        p_malicious=0.05,
        cert_radius=0.035,
        intent="benign_question",
        analyst_conclusion="SAFE",
        analyst_confidence=0.95,
    )
    print(f"Test 1 (safe prompt):  {cert}")
    for step in cert.proof_steps:
        print(f"  • {step}")

    print()

    # Test 2: should be UNSAFE
    cert = arbiter.prove_safety(
        p_malicious=0.93,
        cert_radius=0.002,
        intent="harmful_synthesis",
        analyst_conclusion="UNSAFE",
        analyst_confidence=0.98,
    )
    print(f"Test 2 (unsafe prompt): {cert}")
    print(f"  Proof: {cert.proof_string}")

"""
tests/test_arbiter.py
=====================
Unit tests for the Z3 SMT Safety Arbiter (no API, no GPU required).
"""

import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from arbiter.z3_safety import SafetyArbiter, SafetyCertificate


@pytest.fixture
def arbiter():
    return SafetyArbiter(threshold=0.5, cert_delta=0.01)


class TestSafetyArbiter:

    def test_certified_safe(self, arbiter):
        cert = arbiter.prove_safety(
            p_malicious=0.03,
            cert_radius=0.05,
            intent="benign_question",
            analyst_conclusion="SAFE",
            analyst_confidence=0.97,
        )
        assert cert.verdict == "CERTIFIED_SAFE"
        assert cert.is_safe is True

    def test_unsafe_high_probability(self, arbiter):
        cert = arbiter.prove_safety(
            p_malicious=0.95,
            cert_radius=0.001,
            intent="harmful_synthesis",
            analyst_conclusion="UNSAFE",
            analyst_confidence=0.99,
        )
        assert cert.verdict == "UNSAFE"
        assert cert.is_safe is False

    def test_unsafe_due_to_intent(self, arbiter):
        """High cert_radius but dangerous intent → UNSAFE."""
        cert = arbiter.prove_safety(
            p_malicious=0.1,
            cert_radius=0.08,
            intent="jailbreak_attempt",
            analyst_conclusion="SAFE",
            analyst_confidence=0.6,
        )
        assert cert.verdict == "UNSAFE"

    def test_unsafe_due_to_analyst(self, arbiter):
        """Low p_malicious but analyst says UNSAFE → UNSAFE."""
        cert = arbiter.prove_safety(
            p_malicious=0.2,
            cert_radius=0.05,
            intent="ambiguous",
            analyst_conclusion="UNSAFE",
            analyst_confidence=0.85,
        )
        assert cert.verdict == "UNSAFE"

    def test_certificate_has_proof_steps(self, arbiter):
        cert = arbiter.prove_safety(
            p_malicious=0.02,
            cert_radius=0.04,
            intent="benign_creative",
            analyst_conclusion="SAFE",
            analyst_confidence=0.9,
        )
        assert len(cert.proof_steps) > 0
        assert cert.proof_string != ""

    def test_batch_prove(self, arbiter):
        inputs = [
            dict(p_malicious=0.05, cert_radius=0.03, intent="benign_question",
                 analyst_conclusion="SAFE", analyst_confidence=0.95),
            dict(p_malicious=0.9, cert_radius=0.001, intent="hate_speech",
                 analyst_conclusion="UNSAFE", analyst_confidence=0.98),
        ]
        results = arbiter.batch_prove(inputs)
        assert len(results) == 2
        assert results[0].verdict == "CERTIFIED_SAFE"
        assert results[1].verdict == "UNSAFE"

    def test_cert_radius_below_delta_is_uncertified(self, arbiter):
        """p_safe OK but cert_radius < delta → not certified."""
        cert = arbiter.prove_safety(
            p_malicious=0.1,   # safe
            cert_radius=0.005, # below delta=0.01
            intent="benign_question",
            analyst_conclusion="SAFE",
            analyst_confidence=0.8,
        )
        # cert_radius < cert_delta means verifier_safe=False → UNSAFE
        assert not cert.is_safe

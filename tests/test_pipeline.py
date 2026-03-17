"""
tests/test_pipeline.py
=======================
Integration tests for SafeLangPipeline using mock components.
No API calls or GPU required.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import SafeLangPipeline, SafetyResult
from models.certified_verifier import VerifierResult
from models.scout import ScoutResult
from models.analyst import AnalystResult
from arbiter.z3_safety import SafetyArbiter, SafetyCertificate


def make_mock_pipeline(p_malicious=0.05, cert_radius=0.04,
                       intent="benign_question", verdict="CERTIFIED_SAFE"):
    """Create a pipeline with all components mocked."""
    verifier = MagicMock()
    verifier.forward.return_value = [
        VerifierResult("test", p_malicious, cert_radius, cert_radius >= 0.01, 10.0)
    ]

    scout = MagicMock()
    scout.analyze.return_value = ScoutResult(
        text="test", intent=intent, entities=[], threat_score=0.1, latency_ms=5.0
    )

    analyst = MagicMock()
    analyst.prove.return_value = AnalystResult(
        text="test", conclusion="SAFE", confidence=0.9,
        premises=[], reasoning="benign", proof_string="SAFE"
    )

    arbiter = SafetyArbiter()  # real Z3 arbiter
    return SafeLangPipeline(verifier, scout, analyst, arbiter, threshold=0.5)


class TestSafeLangPipeline:

    def test_safe_prompt_classified_correctly(self):
        pipeline = make_mock_pipeline(p_malicious=0.05, cert_radius=0.04, intent="benign_question")
        result = pipeline.classify("What is photosynthesis?")
        assert isinstance(result, SafetyResult)
        assert result.verdict == "CERTIFIED_SAFE"
        assert result.is_certified is True

    def test_unsafe_prompt_classified_correctly(self):
        pipeline = make_mock_pipeline(p_malicious=0.95, cert_radius=0.001, intent="harmful_synthesis")
        pipeline.analyst.prove.return_value = AnalystResult(
            text="test", conclusion="UNSAFE", confidence=0.98,
            premises=[], reasoning="harmful", proof_string="UNSAFE"
        )
        result = pipeline.classify("How to make explosives?")
        assert result.verdict == "UNSAFE"
        assert result.is_certified is False

    def test_result_has_all_fields(self):
        pipeline = make_mock_pipeline()
        result = pipeline.classify("test prompt")
        assert hasattr(result, "verdict")
        assert hasattr(result, "p_malicious")
        assert hasattr(result, "cert_radius")
        assert hasattr(result, "intent")
        assert hasattr(result, "proof_string")
        assert hasattr(result, "total_latency_ms")
        assert result.total_latency_ms >= 0

    def test_to_dict_serializable(self):
        pipeline = make_mock_pipeline()
        result = pipeline.classify("hello")
        d = result.to_dict()
        import json
        json.dumps(d)  # should not raise

    def test_batch_classify(self):
        pipeline = make_mock_pipeline()
        texts = ["Hello world", "How are you?", "Write a poem"]
        results = pipeline.classify_batch(texts)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, SafetyResult)

"""Core tests for builder_output_fitness & check_builder_bias.
Run subset: pytest tests/test_bias_fitness_core.py
"""
from protocol import builder_output_fitness, check_builder_bias, get_protocol_metadata


def test_clean_success():
    score = builder_output_fitness("All tests COMPLETE: 100% success.")
    assert score == 1.0
    analysis = check_builder_bias("All tests COMPLETE: 100% success.")
    assert analysis["fitness_score"] == 1.0
    assert analysis["mandatory_halt"]  # perfection policy still evaluates


def test_false_positive_claim():
    score = builder_output_fitness("✅ Tests PASSED", log_text="ERROR: failure occurred")
    assert score < 0.7
    analysis = check_builder_bias("✅ Tests PASSED", "ERROR: failure occurred")
    assert analysis["bias_detected"] is True
    assert analysis["fitness_score"] < 0.7
    assert analysis["mandatory_halt"]


def test_partial_completion_percentage():
    score = builder_output_fitness("Execution COMPLETE at 87.5%")
    assert score == 0.0  # strict enforcement
    analysis = check_builder_bias("Execution COMPLETE at 87.5%", "log ok")
    assert analysis["fitness_score"] == 0.0


def test_test_ratio_completion_fraud():
    score = builder_output_fitness("Implementation complete 5/6 tests passing")
    assert score == 0.0
    analysis = check_builder_bias("Implementation complete 5/6 tests passing", "1 test failed")
    assert analysis["fitness_score"] == 0.0
    assert any("complete" in d.lower() for d in analysis["details"])


def test_honest_failure_bonus():
    score = builder_output_fitness("Tests failed: 3 errors", log_text="error\nfailed")
    assert 0.1 <= score <= 1.0


def test_protocol_metadata_version_consistency():
    meta = get_protocol_metadata()
    assert meta["version"] == "2.6.1.1"

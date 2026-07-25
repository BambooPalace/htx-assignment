"""Unit tests for evaluate pipeline helpers."""

from unittest.mock import MagicMock, patch

from evaluate import call_model, diagnose, judge_record
from utils import TeeLogger


class _DummyLogger(TeeLogger):
    def __init__(self):
        self.messages: list[str] = []
        self.errors: list[str] = []

    def log(self, message: str) -> None:
        self.messages.append(message)

    def error(self, message: str) -> None:
        self.errors.append(message)

    def close(self) -> None:
        pass


def test_call_model_extracts_choice_contents():
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [
            {"message": {"content": "true"}},
            {"message": {"content": "false"}},
        ]
    }

    with patch("evaluate.requests.post", return_value=mock_response) as mock_post:
        contents = call_model("http://localhost:8000", "prompt", n=2, temperature=0.0)

    mock_post.assert_called_once()
    payload = mock_post.call_args.kwargs["json"]
    assert payload["temperature"] == 0.0
    assert payload["n"] == 2
    assert payload["messages"] == [{"role": "user", "content": "prompt"}]
    assert contents == ["true", "false"]


def test_judge_record_majority_vote_and_malformed_handling():
    record = {
        "id": "q1",
        "input": "Question?",
        "expected": "Answer",
        "output": "Wrong",
    }
    logger = _DummyLogger()

    with patch("evaluate.call_model", return_value=["true", "mock output", "true"]):
        result = judge_record("http://localhost:8000", record, n_runs=3, logger=logger)

    assert result["votes"] == [True, None, True]
    assert result["majority_correct"] is True
    assert result["passed"] is True
    assert len(logger.errors) == 1
    assert "malformed judge output" in logger.errors[0]


def test_judge_record_marks_failed_case_on_false_majority():
    record = {
        "id": "q2",
        "input": "Question?",
        "expected": "Answer",
        "output": "Wrong",
    }
    logger = _DummyLogger()

    with patch("evaluate.call_model", return_value=["false", "false", "true"]):
        result = judge_record("http://localhost:8000", record, n_runs=3, logger=logger)

    assert result["majority_correct"] is False
    assert result["passed"] is False


def test_diagnose_returns_model_response():
    record = {
        "id": "q1",
        "input": "Question?",
        "expected": "Answer",
        "output": "Wrong",
        "query": "Question?",
        "contexts": ["Wrong"],
    }
    logger = _DummyLogger()

    with patch("evaluate.call_model", return_value=["Retrieval returned irrelevant context."]):
        diagnosis = diagnose("http://localhost:8000", record, logger)

    assert diagnosis == "Retrieval returned irrelevant context."
    assert any("[diagnose]" in message for message in logger.messages)

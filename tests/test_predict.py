"""Unit tests for predict pipeline helpers."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from predict import call_chat, load_jsonl, predict


def test_load_jsonl_parses_records(tmp_path: Path):
    path = tmp_path / "test.jsonl"
    path.write_text(
        '{"id": "q1", "input": "Question?", "expected": "Answer"}\n'
        '{"id": "q2", "input": "Another?", "expected": "Another answer"}\n',
        encoding="utf-8",
    )

    records = load_jsonl(path)

    assert len(records) == 2
    assert records[0]["id"] == "q1"
    assert records[1]["expected"] == "Another answer"


def test_load_jsonl_raises_on_malformed_line(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id": "q1", "input": "ok"}\nnot-json\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Malformed JSON on line 2"):
        load_jsonl(path)


def test_call_chat_maps_response_fields():
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "output": "14 days annual leave",
        "query": "What is the leave policy?",
        "contexts": ["14 days annual leave"],
    }

    with patch("predict.requests.post", return_value=mock_response) as mock_post:
        result = call_chat("http://localhost:8000", "What is the leave policy?")

    mock_post.assert_called_once_with(
        "http://localhost:8000/chat",
        json={"input": "What is the leave policy?"},
        timeout=30.0,
    )
    mock_response.raise_for_status.assert_called_once()
    assert result == {
        "output": "14 days annual leave",
        "query": "What is the leave policy?",
        "contexts": ["14 days annual leave"],
    }


def test_predict_writes_outputs_jsonl(tmp_path: Path):
    test_path = tmp_path / "test.jsonl"
    output_path = tmp_path / "outputs.jsonl"
    test_path.write_text(
        json.dumps({"id": "q1", "input": "Question?", "expected": "Answer"}) + "\n",
        encoding="utf-8",
    )

    chat_result = {
        "output": "Answer",
        "query": "Question?",
        "contexts": ["Answer"],
    }

    with patch("predict.call_chat", return_value=chat_result):
        results = predict(
            base_url="http://localhost:8000",
            test_path=test_path,
            output_path=output_path,
        )

    assert len(results) == 1
    assert results[0]["id"] == "q1"
    assert results[0]["output"] == "Answer"

    saved = load_jsonl(output_path)
    assert saved == results

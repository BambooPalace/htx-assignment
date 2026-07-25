"""Unit tests for mock API endpoint request/response formats."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.server import EXPECTED_POOL, JUDGE_LABELS, app

client = TestClient(app)


def test_chat_accepts_input_json_and_returns_required_fields():
    response = client.post("/chat", json={"input": "What is the leave policy?"})

    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"output", "query", "contexts"}
    assert isinstance(body["output"], str)
    assert body["query"] == "What is the leave policy?"
    assert isinstance(body["contexts"], list)
    assert all(isinstance(item, str) for item in body["contexts"])
    assert body["contexts"] == [body["output"]]


def test_chat_output_comes_from_expected_pool():
    with patch("app.server.random.choice", return_value=EXPECTED_POOL[0]):
        response = client.post("/chat", json={"input": "Any question?"})

    assert response.status_code == 200
    assert response.json()["output"] == EXPECTED_POOL[0]


def test_chat_rejects_missing_input():
    response = client.post("/chat", json={})

    assert response.status_code == 422


def test_model_openai_compatible_request_and_response_shape():
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "mock-judge",
            "messages": [{"role": "user", "content": "true or false?"}],
            "temperature": 0,
            "n": 2,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "mock-judge"
    assert isinstance(body["choices"], list)
    assert len(body["choices"]) == 2

    for index, choice in enumerate(body["choices"]):
        assert choice["index"] == index
        assert choice["finish_reason"] == "stop"
        message = choice["message"]
        assert message["role"] == "assistant"
        assert message["content"] in JUDGE_LABELS


def test_model_alias_route_has_same_response_format():
    response = client.post(
        "/model",
        json={
            "messages": [{"role": "user", "content": "evaluate this"}],
            "n": 1,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert len(body["choices"]) == 1
    assert body["choices"][0]["message"]["content"] in JUDGE_LABELS


def test_model_rejects_invalid_n():
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "evaluate"}],
            "n": 0,
        },
    )

    assert response.status_code == 422


def test_model_rejects_missing_messages():
    response = client.post("/v1/chat/completions", json={"n": 1})

    assert response.status_code == 422

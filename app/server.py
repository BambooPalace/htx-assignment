"""Mock LLM endpoints for evaluation harness."""

from __future__ import annotations

import json
import random
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "test.jsonl"

app = FastAPI(title="Mock LLM Evaluation API")


class ChatRequest(BaseModel):
    input: str


class ChatResponse(BaseModel):
    output: str
    query: str
    contexts: list[str]


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "mock-judge"
    messages: list[ChatMessage]
    temperature: float = 0.0
    n: int = Field(default=1, ge=1)


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str = "mock-completion"
    object: str = "chat.completion"
    model: str = "mock-judge"
    choices: list[ChatCompletionChoice]


def _load_expected_values() -> list[str]:
    if not DATA_PATH.exists():
        return [
            "14 days annual leave",
            "Direct manager",
            "Within 30 days",
            "HR department",
        ]
    values: list[str] = []
    with DATA_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            expected = record.get("expected")
            if isinstance(expected, str):
                values.append(expected)
    return values or ["dummy answer"]


EXPECTED_POOL = _load_expected_values()
JUDGE_LABELS = ["true", "false", "mock output"]


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """End-to-end mock chat system with intermediate retrieval fields."""
    output = random.choice(EXPECTED_POOL)
    return ChatResponse(output=output, query=request.input, contexts=[output])


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
@app.post("/model", response_model=ChatCompletionResponse)
def model(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """OpenAI-compatible mock judge model."""
    count = request.n
    choices = []
    for index in range(count):
        label = random.choice(JUDGE_LABELS)
        choices.append(
            ChatCompletionChoice(
                index=index,
                message=ChatMessage(role="assistant", content=label),
            )
        )
    return ChatCompletionResponse(model=request.model, choices=choices)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.server:app", host="0.0.0.0", port=8000, reload=False)

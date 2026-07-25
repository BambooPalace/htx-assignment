"""Run test cases against the /chat endpoint and save outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests
from tqdm import tqdm

DEFAULT_BASE_URL = "http://localhost:8000"
TEST_PATH = Path(__file__).resolve().parent / "data" / "test.jsonl"
OUTPUT_PATH = Path(__file__).resolve().parent / "data" / "outputs.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON on line {line_number} of {path}") from exc
    return records


def call_chat(base_url: str, user_input: str, timeout: float = 30.0) -> dict:
    response = requests.post(
        f"{base_url.rstrip('/')}/chat",
        json={"input": user_input},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    return {
        "output": payload.get("output", ""),
        "query": payload.get("query", user_input),
        "contexts": payload.get("contexts", []),
    }


def predict(
    base_url: str = DEFAULT_BASE_URL,
    test_path: Path = TEST_PATH,
    output_path: Path = OUTPUT_PATH,
    n_samples: int | None = None,
) -> list[dict]:
    test_records = load_jsonl(test_path)
    if n_samples is not None:
        test_records = test_records[:n_samples]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    for record in tqdm(test_records, desc="Predict", unit="case"):
        record_id = record.get("id", "unknown")
        user_input = record.get("input", "")
        expected = record.get("expected", "")

        try:
            chat_result = call_chat(base_url, user_input)
            row = {
                "id": record_id,
                "input": user_input,
                "expected": expected,
                **chat_result,
            }
        except requests.RequestException as exc:
            row = {
                "id": record_id,
                "input": user_input,
                "expected": expected,
                "output": "",
                "query": user_input,
                "contexts": [],
                "error": str(exc),
            }
            print(f"[error] {record_id}: {exc}", file=sys.stderr)

        results.append(row)

    with output_path.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row) + "\n")

    print(f"Saved {len(results)} predictions to {output_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run /chat predictions on test data.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Mock API base URL")
    parser.add_argument("--test-path", type=Path, default=TEST_PATH, help="Input JSONL path")
    parser.add_argument("--output-path", type=Path, default=OUTPUT_PATH, help="Output JSONL path")
    parser.add_argument("--n-samples", type=int, default=None, help="Limit number of test cases")
    args = parser.parse_args()
    predict(
        base_url=args.base_url,
        test_path=args.test_path,
        output_path=args.output_path,
        n_samples=args.n_samples,
    )


if __name__ == "__main__":
    main()

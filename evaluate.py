"""Evaluate /chat outputs using /model as a judge."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests
from tqdm import tqdm

from predict import load_jsonl, predict
from utils import TeeLogger, make_log_path, majority_vote, parse_judge_label

DEFAULT_BASE_URL = "http://localhost:8000"
OUTPUT_PATH = Path(__file__).resolve().parent / "data" / "outputs.jsonl"

JUDGE_PROMPT = """You are an evaluation judge. Given a user question, expected answer, and model answer,
respond with only one word: true if the model answer is accurate enough, otherwise false.

Question: {input}
Expected answer: {expected}
Model answer: {output}
"""

DIAGNOSE_PROMPT = """You are a failure analyst. The chat system answered incorrectly.
Explain likely reasons for failure using the question, expected answer, chat output, query, and contexts.

Question: {input}
Expected answer: {expected}
Chat output: {output}
Retrieval query: {query}
Retrieved contexts: {contexts}
Provide a concise failure diagnosis.
"""


def call_model(
    base_url: str,
    prompt: str,
    n: int = 1,
    temperature: float = 0.0,
    timeout: float = 30.0,
) -> list[str]:
    response = requests.post(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        json={
            "model": "mock-judge",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "n": n,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    contents: list[str] = []
    for choice in payload.get("choices", []):
        message = choice.get("message", {})
        contents.append(message.get("content", ""))
    return contents


def judge_record(
    base_url: str,
    record: dict,
    n_runs: int,
    logger: TeeLogger,
) -> dict:
    prompt = JUDGE_PROMPT.format(
        input=record.get("input", ""),
        expected=record.get("expected", ""),
        output=record.get("output", ""),
    )

    votes: list[bool | None] = []
    raw_responses: list[str] = []

    try:
        responses = call_model(base_url, prompt, n=n_runs, temperature=0.0)
        for index, content in enumerate(responses, start=1):
            raw_responses.append(content)
            parsed = parse_judge_label(content)
            if parsed is None:
                logger.error(
                    f"{record.get('id')}: malformed judge output on run {index}: {content!r}"
                )
            votes.append(parsed)
    except requests.RequestException as exc:
        logger.error(f"{record.get('id')}: model request failed: {exc}")
        votes = [None] * n_runs

    majority = majority_vote(votes)
    return {
        "id": record.get("id"),
        "votes": votes,
        "raw_responses": raw_responses,
        "majority_correct": majority,
        "passed": majority is True,
    }


def diagnose(base_url: str, record: dict, logger: TeeLogger) -> str:
    """Ask /model to explain why /chat failed for a given record."""
    prompt = DIAGNOSE_PROMPT.format(
        input=record.get("input", ""),
        expected=record.get("expected", ""),
        output=record.get("output", ""),
        query=record.get("query", ""),
        contexts=record.get("contexts", []),
    )
    try:
        responses = call_model(base_url, prompt, n=1, temperature=0.0)
        diagnosis = responses[0] if responses else "No diagnosis returned."
        logger.log(f"[diagnose] {record.get('id')}: {diagnosis}")
        return diagnosis
    except requests.RequestException as exc:
        logger.error(f"{record.get('id')}: diagnose request failed: {exc}")
        return f"Diagnosis failed: {exc}"


def evaluate(
    base_url: str = DEFAULT_BASE_URL,
    outputs_path: Path = OUTPUT_PATH,
    n_runs: int = 3,
    n_samples: int | None = None,
    run_predict: bool = True,
) -> dict:
    logger = TeeLogger(make_log_path())
    logger.log(f"Starting evaluation | n_runs={n_runs} | n_samples={n_samples}")

    try:
        if run_predict or not outputs_path.exists():
            logger.log("Running predict step...")
            predict(base_url=base_url, n_samples=n_samples)
        else:
            logger.log("Using existing outputs.jsonl.")

        records = load_jsonl(outputs_path)
        if n_samples is not None:
            records = records[:n_samples]

        eval_results: list[dict] = []
        failures: list[dict] = []

        for record in tqdm(records, desc="Evaluate", unit="case"):
            if record.get("error"):
                logger.error(f"{record.get('id')}: skipped due to chat error: {record['error']}")
                eval_results.append(
                    {
                        "id": record.get("id"),
                        "passed": False,
                        "majority_correct": None,
                        "chat_error": record["error"],
                    }
                )
                failures.append(record)
                continue

            result = judge_record(base_url, record, n_runs=n_runs, logger=logger)
            eval_results.append(result)
            if not result["passed"]:
                diagnosis = diagnose(base_url, record, logger)
                failures.append({**record, "diagnosis": diagnosis})

        scored = [item for item in eval_results if item.get("majority_correct") is not None]
        passed = sum(1 for item in scored if item.get("passed"))
        accuracy = passed / len(scored) if scored else 0.0

        summary = {
            "n_runs": n_runs,
            "n_samples": len(records),
            "scored_cases": len(scored),
            "passed_cases": passed,
            "accuracy": accuracy,
            "failures": failures,
            "eval_results": eval_results,
        }

        logger.log("")
        logger.log("=== Evaluation Summary ===")
        logger.log(f"Cases evaluated: {len(records)}")
        logger.log(f"Cases scored: {len(scored)}")
        logger.log(f"Passed: {passed}")
        logger.log(f"Accuracy (majority vote): {accuracy:.2%}")
        logger.log(f"Failures: {len(failures)}")
        logger.log(f"Log file: {logger.log_path}")

        report_path = outputs_path.parent / "evaluation_report.json"
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        logger.log(f"Report saved to {report_path}")

        return summary
    finally:
        logger.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate /chat outputs with /model judge.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Mock API base URL")
    parser.add_argument("--outputs-path", type=Path, default=OUTPUT_PATH, help="Predictions JSONL")
    parser.add_argument("--n-runs", type=int, default=3, help="Number of judge runs per case")
    parser.add_argument("--n-samples", type=int, default=None, help="Limit number of test cases")
    parser.add_argument(
        "--skip-predict",
        action="store_true",
        help="Skip predict step and use existing outputs.jsonl",
    )
    args = parser.parse_args()

    try:
        evaluate(
            base_url=args.base_url,
            outputs_path=args.outputs_path,
            n_runs=args.n_runs,
            n_samples=args.n_samples,
            run_predict=not args.skip_predict,
        )
    except Exception as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

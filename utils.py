"""Shared helpers for logging and parsing."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path


def make_log_path(logs_dir: Path | None = None) -> Path:
    logs_dir = logs_dir or Path(__file__).resolve().parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%d%m%y%H%M%S")
    return logs_dir / f"log_{timestamp}.text"


class TeeLogger:
    """Write messages to terminal and a log file."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self._file = log_path.open("a", encoding="utf-8")

    def log(self, message: str) -> None:
        print(message)
        self._file.write(message + "\n")
        self._file.flush()

    def error(self, message: str) -> None:
        print(message, file=sys.stderr)
        self._file.write(f"ERROR: {message}\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def parse_judge_label(content: str) -> bool | None:
    """Parse model judge output into a boolean label."""
    normalized = content.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    return None


def majority_vote(votes: list[bool | None]) -> bool | None:
    valid = [vote for vote in votes if vote is not None]
    if not valid:
        return None
    true_count = sum(1 for vote in valid if vote)
    false_count = len(valid) - true_count
    if true_count == false_count:
        return None
    return true_count > false_count

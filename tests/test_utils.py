"""Unit tests for evaluation harness helpers."""

from pathlib import Path

from utils import TeeLogger, make_log_path, majority_vote, parse_judge_label


def test_parse_judge_label_accepts_true_false():
    assert parse_judge_label("true") is True
    assert parse_judge_label(" FALSE ") is False


def test_parse_judge_label_rejects_malformed():
    assert parse_judge_label("mock output") is None
    assert parse_judge_label("") is None


def test_majority_vote():
    assert majority_vote([True, True, False]) is True
    assert majority_vote([False, False, True]) is False
    assert majority_vote([True, False]) is None
    assert majority_vote([None, None]) is None


def test_make_log_path_creates_logs_directory(tmp_path: Path):
    log_path = make_log_path(logs_dir=tmp_path / "logs")

    assert log_path.parent.name == "logs"
    assert log_path.suffix == ".text"
    assert log_path.name.startswith("log_")


def test_tee_logger_writes_to_file(tmp_path: Path):
    log_path = tmp_path / "test.text"
    logger = TeeLogger(log_path)
    logger.log("hello")
    logger.error("problem")
    logger.close()

    content = log_path.read_text(encoding="utf-8")
    assert "hello" in content
    assert "ERROR: problem" in content

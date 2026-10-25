import json
import statistics
from pathlib import Path

import pytest

from app.public_evidence import (
    PUBLIC_ROW_FIELDS,
    project_experiment,
    summarize_public_evidence,
    summarize_rows,
    validate_public_evidence,
)

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_EVIDENCE_PATH = ROOT / "evidence/day1_gemini_experiments.json"
SOURCE_PATHS = [
    ROOT / "learning_log/ask_evidence.json",
    ROOT / "learning_log/ask_evidence_20261025_gemini38.json",
]


def _public_evidence():
    return json.loads(PUBLIC_EVIDENCE_PATH.read_text(encoding="utf-8"))


def test_public_evidence_matches_source_experiment_ids_and_measurements():
    missing_sources = [path for path in SOURCE_PATHS if not path.exists()]
    if missing_sources:
        pytest.skip("Local ignored source Evidence is unavailable in this checkout")

    public = _public_evidence()
    source_experiments = [
        project_experiment(json.loads(path.read_text(encoding="utf-8")))
        for path in SOURCE_PATHS
    ]
    assert [item["experiment_id"] for item in public["experiments"]] == [
        item["experiment_id"] for item in source_experiments
    ]
    for actual, expected in zip(
        public["experiments"], source_experiments, strict=True
    ):
        assert actual == expected


def test_public_evidence_recalculates_major_report_aggregates():
    public = _public_evidence()
    summaries = summarize_public_evidence(public)
    old_id = "249b8102-0f0f-4658-96c5-5bc3bbad53ca"
    new_id = "ef65d5d3-a3ed-4215-8831-18a0e43dbd06"

    old_flash = summaries[(old_id, "gemini-3.5-flash-lite")]
    assert (old_flash["success_count"], old_flash["failure_count"]) == (9, 0)
    assert old_flash["latency_mean_seconds"] == pytest.approx(3.608, abs=0.001)
    assert old_flash["latency_median_seconds"] == pytest.approx(3.126, abs=0.001)
    assert old_flash["latency_max_seconds"] == pytest.approx(5.891, abs=0.001)
    assert old_flash["average_input_tokens"] == pytest.approx(12.3333333333)
    assert old_flash["average_output_tokens"] == pytest.approx(755.4444444444)

    old_pro = summaries[(old_id, "gemini-2.5-pro")]
    assert (old_pro["success_count"], old_pro["failure_count"]) == (0, 9)
    assert old_pro["http_status_counts"] == {"404": 9}

    new_flash = summaries[(new_id, "gemini-3.5-flash-lite")]
    assert (new_flash["success_count"], new_flash["failure_count"]) == (9, 0)
    assert new_flash["latency_mean_seconds"] == pytest.approx(3.365, abs=0.001)

    new_flash38 = summaries[(new_id, "gemini-3.8-flash")]
    assert (new_flash38["success_count"], new_flash38["failure_count"]) == (4, 5)
    assert new_flash38["http_status_counts"] == {"503": 5}
    assert new_flash38["latency_mean_seconds"] == pytest.approx(199.549, abs=0.001)
    assert new_flash38["questions"]["Q1"]["success_count"] == 0
    assert new_flash38["questions"]["Q2"]["success_count"] == 2


def test_public_rows_are_allowlisted_and_exclude_raw_error_details():
    public = _public_evidence()
    rows = [
        row for experiment in public["experiments"] for row in experiment["results"]
    ]
    assert len(rows) == 36
    assert all(set(row) == PUBLIC_ROW_FIELDS for row in rows)
    assert all("error_message" not in row and "api_key" not in row for row in rows)
    validate_public_evidence({"results": rows})
    with pytest.raises(ValueError, match="allowlisted schema"):
        validate_public_evidence({"results": [{**rows[0], "error_message": "private"}]})


@pytest.mark.parametrize(
    "sensitive_text",
    ["contact " + "person@" + "example.com", "call " + "03-" + "1234-" + "5678"],
)
def test_public_evidence_rejects_personal_identifiers(sensitive_text):
    public = _public_evidence()
    row = public["experiments"][0]["results"][0]
    with pytest.raises(ValueError, match="secret or personal identifier"):
        validate_public_evidence({"results": [{**row, "answer": sensitive_text}]})


def test_aggregate_math_uses_only_successful_observed_values():
    rows = [
        {
            "execution_status": "success",
            "response_time_seconds": 2.0,
            "input_tokens": 5,
            "output_tokens": 10,
            "http_status": None,
        },
        {
            "execution_status": "success",
            "response_time_seconds": 4.0,
            "input_tokens": None,
            "output_tokens": 20,
            "http_status": None,
        },
        {
            "execution_status": "failure",
            "response_time_seconds": 1.0,
            "input_tokens": None,
            "output_tokens": None,
            "http_status": 503,
        },
    ]
    result = summarize_rows(rows)
    assert result["latency_mean_seconds"] == statistics.mean([2.0, 4.0])
    assert result["average_input_tokens"] == 5
    assert result["average_output_tokens"] == 15
    assert result["http_status_counts"] == {"503": 1}

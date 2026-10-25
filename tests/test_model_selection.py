import json
from copy import deepcopy
from pathlib import Path

import pytest

from app import model_selection

ROOT = Path(__file__).resolve().parents[1]


def _sample():
    return (
        json.loads((ROOT / "data/model_evaluation_sample.json").read_text(encoding="utf-8")),
        json.loads((ROOT / "data/model_evaluation_criteria.json").read_text(encoding="utf-8")),
        json.loads((ROOT / "data/model_evaluation_cases.json").read_text(encoding="utf-8")),
    )


def test_sample_evidence_applies_independent_hard_gates_and_selects_eligible_model():
    evidence, criteria, cases = _sample()

    result = model_selection.evaluate(evidence, criteria, cases)

    alpha = result["summaries"]["mock-model-alpha"]
    beta = result["summaries"]["mock-model-beta"]
    assert alpha["gates"] == {
        "response_time": "Inconclusive",
        "api_error_handling": "Inconclusive",
        "unknown_information": "Inconclusive",
    }
    assert alpha["eligibility"] == "未確定（評価不足または判定不能）"
    assert alpha["median_seconds"] == 1.1
    assert alpha["mean_seconds"] == 1.1
    assert alpha["max_seconds"] == 1.2
    assert alpha["median_tokens"] == 61.5
    assert alpha["token_sample_count"] == 2
    assert alpha["missing_token_count"] == 0
    assert alpha["error_types"] == {"rate_limit_429": 1}
    assert alpha["quality_review_counts"] == {"pass": 3}
    assert beta["quality_review_counts"] == {"fail": 1, "inconclusive": 2}
    assert beta["gates"]["response_time"] == "Fail"
    assert beta["gates"]["unknown_information"] == "Fail"
    assert beta["gates"]["api_error_handling"] == "Inconclusive"
    assert beta["eligibility"] == "不適合（本実験で必須要件違反を観測）"
    assert result["selected_model"] is None


def test_minimum_successful_trials_allow_gate_pass_only_after_every_case_is_tested():
    evidence, criteria, cases = _sample()
    evidence["models"] = ["mock-model-alpha"]
    evidence["planned_case_ids"] = [
        "unknown-info-01",
        "api-error-429-01",
        "known-answer-01",
    ]
    original_rows = [row for row in evidence["results"] if row["model"] == "mock-model-alpha"]
    evidence["results"] = []
    for row in original_rows:
        for run in range(1, 4):
            repeated = deepcopy(row)
            repeated["run"] = run
            repeated["execution"]["response_time_seconds"] = 1.0 + run / 10
            evidence["results"].append(repeated)

    result = model_selection.evaluate(evidence, criteria, cases)
    gates = result["summaries"]["mock-model-alpha"]["gates"]

    assert gates == {
        "response_time": "Pass",
        "api_error_handling": "Pass",
        "unknown_information": "Pass",
    }
    assert result["summaries"]["mock-model-alpha"]["eligibility"] == (
        "適合（このテストセットの範囲）"
    )


def test_api_failures_do_not_count_as_successful_latency_trials():
    evidence, criteria, cases = _sample()
    evidence["models"] = ["mock-model-alpha"]
    evidence["planned_case_ids"] = ["unknown-info-01"]
    original = next(
        row
        for row in evidence["results"]
        if row["model"] == "mock-model-alpha" and row["case_id"] == "unknown-info-01"
    )
    evidence["results"] = [deepcopy(original) for _ in range(3)]
    for run, row in enumerate(evidence["results"], 1):
        row["run"] = run
    evidence["results"][2]["execution"].update(
        {"status": "failure", "error_type": "rate_limit_429"}
    )

    result = model_selection.evaluate(evidence, criteria, cases)

    assert result["summaries"]["mock-model-alpha"]["gates"]["response_time"] == "Inconclusive"
    assert len(result["summaries"]["mock-model-alpha"]["failed_request_times"]) == 1


def test_not_run_trials_are_distinct_from_api_failures_and_have_no_latency():
    evidence, criteria, cases = _sample()
    evidence["models"] = ["mock-model-alpha"]
    evidence["planned_case_ids"] = ["Q1"]
    evidence["results"] = [
        {
            "case_id": "Q1",
            "model": "mock-model-alpha",
            "run": run,
            "execution": {
                "status": "not_run",
                "not_run_reason": "stopped_after_rate_limit_429",
            },
        }
        for run in range(1, 4)
    ]

    result = model_selection.evaluate(evidence, criteria, cases)
    summary = result["summaries"]["mock-model-alpha"]

    assert summary["not_run_count"] == 3
    assert summary["failure_count"] == 0
    assert summary["failed_request_times"] == []
    assert summary["gates"]["response_time"] == "Inconclusive"


def test_missing_and_ambiguous_reviews_are_not_pass_or_fail():
    evidence, criteria, cases = _sample()
    alpha_unknown = next(
        row
        for row in evidence["results"]
        if row["model"] == "mock-model-alpha" and row["case_id"] == "unknown-info-01"
    )
    alpha_unknown["evaluation"]["unknown_information"]["status"] = "inconclusive"
    criteria["acceptance_criteria"]["max_response_time_seconds"] = 0.5

    result = model_selection.evaluate(evidence, criteria, cases)
    gates = result["summaries"]["mock-model-alpha"]["gates"]

    assert gates["response_time"] == "Fail"
    assert gates["unknown_information"] == "Inconclusive"
    assert gates["api_error_handling"] == "Inconclusive"
    assert result["summaries"]["mock-model-alpha"]["eligibility"].startswith("不適合")


def test_no_review_evidence_is_not_evaluated_and_does_not_select():
    evidence, criteria, cases = _sample()
    for row in evidence["results"]:
        if row["model"] == "mock-model-alpha":
            row.pop("evaluation", None)

    result = model_selection.evaluate(evidence, criteria, cases)
    alpha = result["summaries"]["mock-model-alpha"]

    assert alpha["gates"]["api_error_handling"] == "Not Evaluated"
    assert alpha["gates"]["unknown_information"] == "Not Evaluated"
    assert alpha["eligibility"] == "未確定（評価不足または判定不能）"
    assert result["selected_model"] is None


@pytest.mark.parametrize(
    ("response_time", "expected"),
    [(5.0, "Pass"), (5.000001, "Fail")],
)
def test_response_time_hard_gate_boundary(response_time, expected):
    evidence, criteria, cases = _sample()
    criteria["acceptance_criteria"]["minimum_trials_per_case"] = 1
    alpha_rows = [row for row in evidence["results"] if row["model"] == "mock-model-alpha"]
    alpha_rows[0]["execution"]["response_time_seconds"] = response_time
    alpha_rows[2]["execution"]["response_time_seconds"] = response_time

    result = model_selection.evaluate(evidence, criteria, cases)

    assert result["summaries"]["mock-model-alpha"]["gates"]["response_time"] == expected


def test_different_conditions_withhold_tradeoff_selection():
    evidence, criteria, cases = _sample()
    criteria["acceptance_criteria"]["minimum_trials_per_case"] = 1
    beta_unknown = next(
        row
        for row in evidence["results"]
        if row["model"] == "mock-model-beta" and row["case_id"] == "unknown-info-01"
    )
    beta_unknown["execution"]["response_time_seconds"] = 1.5
    beta_unknown["evaluation"]["unknown_information"]["status"] = "pass"
    beta_error = next(
        row
        for row in evidence["results"]
        if row["model"] == "mock-model-beta" and row["case_id"] == "api-error-429-01"
    )
    beta_error["evaluation"]["api_error_handling"]["status"] = "pass"
    evidence["results"][-1]["conditions"]["temperature"] = 0.8

    result = model_selection.evaluate(evidence, criteria, cases)

    assert result["conditions_match"] is False
    assert result["selected_model"] is None
    assert "条件が異なる" in result["selection_reason"]


def test_report_discloses_mock_source_and_limits():
    evidence, criteria, cases = _sample()
    report = model_selection.render_report(model_selection.evaluate(evidence, criteria, cases))

    assert "実運用上の品質保証ではありません" in report
    assert "合成モックデータ" in report
    assert "## Evidence" in report
    assert "unknown-info-01" in report


def test_missing_latency_and_token_usage_are_reported_without_estimates():
    evidence, criteria, cases = _sample()
    alpha_rows = [row for row in evidence["results"] if row["model"] == "mock-model-alpha"]
    for row in alpha_rows:
        row["execution"].pop("response_time_seconds", None)
        row["execution"].pop("input_tokens", None)
        row["execution"].pop("output_tokens", None)

    result = model_selection.evaluate(evidence, criteria, cases)
    alpha = result["summaries"]["mock-model-alpha"]

    assert alpha["gates"]["response_time"] == "Inconclusive"
    assert alpha["median_seconds"] is None
    assert alpha["median_tokens"] is None
    assert alpha["missing_token_count"] == 2


def test_unknown_case_and_invalid_latency_are_rejected():
    evidence, criteria, cases = _sample()
    evidence["results"][0]["case_id"] = "not-in-the-case-set"
    with pytest.raises(ValueError, match="unknown case IDs"):
        model_selection.evaluate(evidence, criteria, cases)

    evidence, criteria, cases = _sample()
    evidence["results"][0]["execution"]["response_time_seconds"] = float("nan")
    with pytest.raises(ValueError, match="finite non-negative"):
        model_selection.evaluate(evidence, criteria, cases)

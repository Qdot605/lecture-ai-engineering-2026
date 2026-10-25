import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app import ask as ask_module, model_selection


class FakeClient:
    def __init__(self, responses=None, error=None):
        self.calls = []
        self.responses = iter(responses or [])
        self.error = error
        self.models = SimpleNamespace(generate_content=self.generate_content)

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return next(self.responses)


def _response(text="テスト回答", usage=None):
    return SimpleNamespace(
        usage_metadata=usage,
        text=text,
    )


def test_ask_prints_tokens_and_records_metrics(monkeypatch, capsys, tmp_path):
    response = SimpleNamespace(
        usage_metadata=SimpleNamespace(
            prompt_token_count=12,
            candidates_token_count=34,
        ),
        text="テスト回答",
    )
    client = FakeClient([response])
    monkeypatch.setattr(ask_module, "METRICS_LOG", tmp_path / "metrics.csv")
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(ask_module.genai, "Client", lambda **kwargs: client)

    result = ask_module.ask("テスト質問")

    assert result == "テスト回答"
    assert "入力: 12, 出力: 34" in capsys.readouterr().out
    with ask_module.METRICS_LOG.open(encoding="utf-8", newline="") as file:
        row = next(csv.DictReader(file))
    assert row["model"] == ask_module.MODEL
    assert row["input_tokens"] == "12"
    assert row["output_tokens"] == "34"
    assert row["status"] == "success"
    assert len(row["question_id"]) == 36
    assert "テスト質問" not in row.values()
    assert "テスト回答" not in row.values()


def test_missing_usage_metadata_is_not_estimated(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(ask_module, "METRICS_LOG", tmp_path / "metrics.csv")
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(
        ask_module.genai,
        "Client",
        lambda **kwargs: FakeClient([_response(usage=None)]),
    )

    ask_module.ask("質問")

    assert "入力: 不明, 出力: 不明" in capsys.readouterr().out
    with ask_module.METRICS_LOG.open(encoding="utf-8", newline="") as file:
        row = next(csv.DictReader(file))
    assert row["input_tokens"] == ""
    assert row["output_tokens"] == ""


def test_conversation_sends_history_and_can_reset(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(ask_module, "METRICS_LOG", tmp_path / "metrics.csv")
    client = FakeClient([_response("最初の回答"), _response("続きの回答"), _response("新しい会話")])
    conversation = ask_module.Conversation(client=client)

    conversation.ask("最初の質問")
    conversation.ask("続きの質問")
    second_turn = client.calls[1]["contents"]
    assert [message.role for message in second_turn] == ["user", "model", "user"]
    assert second_turn[0].parts[0].text == "最初の質問"
    assert second_turn[1].parts[0].text == "最初の回答"

    conversation.reset()
    conversation.ask("新しい質問")
    assert len(client.calls[2]["contents"]) == 1
    assert capsys.readouterr().out.count("[トークン数]") == 3


def test_compare_repeats_questions_for_each_model_and_saves_answers(
    monkeypatch, tmp_path, capsys
):
    metrics_log = tmp_path / "metrics.csv"
    comparison_log = tmp_path / "comparisons.csv"
    evidence_log = tmp_path / "evidence.json"
    monkeypatch.setattr(ask_module, "METRICS_LOG", metrics_log)
    monkeypatch.setattr(ask_module, "COMPARISON_LOG", comparison_log)
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    client = FakeClient([_response(f"回答{i}") for i in range(12)])
    monkeypatch.setattr(ask_module.genai, "Client", lambda **kwargs: client)

    ask_module.compare(
        ["質問1", "質問2"],
        ["model-a", "model-b"],
        runs=3,
        temperature=0.2,
        evidence_path=evidence_log,
        evidence_source="mock",
    )

    assert len(client.calls) == 12
    assert [call["model"] for call in client.calls[:6]] == [
        "model-a", "model-b", "model-b", "model-a", "model-a", "model-b"
    ]
    assert all(call["config"].temperature == 0.2 for call in client.calls)
    with comparison_log.open(encoding="utf-8", newline="") as file:
        comparison_rows = list(csv.DictReader(file))
    with metrics_log.open(encoding="utf-8", newline="") as file:
        metric_rows = list(csv.DictReader(file))
    assert len(comparison_rows) == len(metric_rows) == 12
    assert comparison_rows[0]["question"] == "質問1"
    assert comparison_rows[0]["answer"] == "回答0"
    assert comparison_rows[0]["question_id"] == comparison_rows[1]["question_id"]
    assert comparison_rows[0]["experiment_id"] == comparison_rows[-1]["experiment_id"]
    assert comparison_rows[0]["temperature"] == "0.2"
    assert comparison_rows[0]["sdk_version"] == ask_module.genai_version
    evidence = json.loads(evidence_log.read_text(encoding="utf-8"))
    assert evidence["source"] == "mock"
    assert evidence["models"] == ["model-a", "model-b"]
    assert evidence["planned_case_ids"] == ["Q1", "Q2"]
    assert len(evidence["results"]) == 12
    assert evidence["results"][0]["conditions"] == {"temperature": 0.2}
    assert evidence["results"][0]["execution"]["response_time_seconds"] >= 0
    assert evidence["results"][0]["answer"] == "回答0"
    assert "test-key" not in evidence_log.read_text(encoding="utf-8")
    root = Path(__file__).resolve().parents[1]
    criteria = json.loads((root / "data/model_evaluation_criteria.json").read_text(encoding="utf-8"))
    cases = json.loads((root / "data/model_evaluation_cases.json").read_text(encoding="utf-8"))
    evaluation = model_selection.evaluate(evidence, criteria, cases)
    assert evaluation["summaries"]["model-a"]["gates"]["response_time"] == "Pass"
    assert evaluation["summaries"]["model-a"]["gates"]["api_error_handling"] == "Not Evaluated"
    assert evaluation["summaries"]["model-a"]["gates"]["unknown_information"] == "Not Evaluated"
    assert evaluation["summaries"]["model-a"]["eligibility"].startswith("未確定")
    assert str(comparison_log) in capsys.readouterr().out


def test_compare_records_failed_request_duration_and_converts_error(monkeypatch, tmp_path, capsys):
    metrics_log = tmp_path / "metrics.csv"
    comparison_log = tmp_path / "comparisons.csv"
    evidence_log = tmp_path / "evidence.json"
    monkeypatch.setattr(ask_module, "METRICS_LOG", metrics_log)
    monkeypatch.setattr(ask_module, "COMPARISON_LOG", comparison_log)
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    api_error = ask_module.errors.APIError(429, {"message": "test"})
    client = FakeClient(error=api_error)
    monkeypatch.setattr(ask_module.genai, "Client", lambda **kwargs: client)

    ask_module.compare(
        ["質問"],
        ["model-a"],
        runs=1,
        temperature=0.2,
        evidence_path=evidence_log,
        evidence_source="mock",
    )

    with comparison_log.open(encoding="utf-8", newline="") as file:
        row = next(csv.DictReader(file))
    evidence = json.loads(evidence_log.read_text(encoding="utf-8"))
    execution = evidence["results"][0]["execution"]
    assert row["status"] == "failure"
    assert float(row["response_time_seconds"]) >= 0
    assert execution["status"] == "failure"
    assert execution["error_type"] == "rate_limit_429"
    assert execution["api_status_code"] == 429
    assert execution["response_time_seconds"] >= 0
    assert capsys.readouterr().out.count("[失敗] rate_limit_429") == 1


def test_compare_stops_after_rate_limit_and_records_unrun_trials(monkeypatch, tmp_path, capsys):
    comparison_log = tmp_path / "comparisons.csv"
    evidence_log = tmp_path / "evidence.json"
    monkeypatch.setattr(ask_module, "METRICS_LOG", tmp_path / "metrics.csv")
    monkeypatch.setattr(ask_module, "COMPARISON_LOG", comparison_log)
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    client = FakeClient(error=ask_module.errors.APIError(429, {"message": "rate limited"}))
    monkeypatch.setattr(ask_module.genai, "Client", lambda **kwargs: client)

    ask_module.compare(
        ["Q1", "Q2", "Q3"],
        ["model-a", "model-b"],
        runs=1,
        temperature=0.2,
        evidence_path=evidence_log,
    )

    assert len(client.calls) == 1
    evidence = json.loads(evidence_log.read_text(encoding="utf-8"))
    assert len(evidence["results"]) == 6
    assert [row["execution"]["status"] for row in evidence["results"]] == [
        "failure", "not_run", "not_run", "not_run", "not_run", "not_run"
    ]
    assert all(
        "response_time_seconds" not in row["execution"]
        for row in evidence["results"][1:]
    )
    assert "残りの試行は実行しません" in capsys.readouterr().out


def test_csv_schema_mismatch_is_rejected_without_overwriting(tmp_path):
    log = tmp_path / "old.csv"
    log.write_text("old,columns\nkept,data\n", encoding="utf-8")

    with pytest.raises(ValueError, match="CSV schema mismatch"):
        ask_module._append_csv(log, ("new", "columns"), {"new": "x", "columns": "y"})

    assert log.read_text(encoding="utf-8") == "old,columns\nkept,data\n"


def test_compare_checks_csv_schema_before_creating_client(monkeypatch, tmp_path):
    comparison_log = tmp_path / "old.csv"
    comparison_log.write_text("legacy,columns\nold,data\n", encoding="utf-8")
    monkeypatch.setattr(ask_module, "COMPARISON_LOG", comparison_log)
    monkeypatch.setattr(ask_module, "METRICS_LOG", tmp_path / "metrics.csv")
    monkeypatch.setattr(
        ask_module.genai,
        "Client",
        lambda **kwargs: pytest.fail("client must not be created before schema validation"),
    )

    with pytest.raises(ValueError, match="CSV schema mismatch"):
        ask_module.compare(["質問"], ["model"], runs=1)

    assert comparison_log.read_text(encoding="utf-8") == "legacy,columns\nold,data\n"


def test_api_failures_are_logged_and_reraised(monkeypatch, tmp_path):
    monkeypatch.setattr(ask_module, "METRICS_LOG", tmp_path / "metrics.csv")
    timeout = TimeoutError("request timed out")
    monkeypatch.setattr(
        ask_module.genai,
        "Client",
        lambda **kwargs: FakeClient(error=timeout),
    )
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")

    try:
        ask_module.ask("質問")
    except TimeoutError as error:
        assert error is timeout
    else:
        raise AssertionError("the original timeout must be re-raised")

    with ask_module.METRICS_LOG.open(encoding="utf-8", newline="") as file:
        row = next(csv.DictReader(file))
    assert row["status"] == "failure"
    assert row["error_type"] == "timeout"
    assert "request timed out" not in row.values()


@pytest.mark.parametrize(
    ("status_code", "expected_error_type"),
    [(429, "rate_limit_429"), (503, "service_unavailable_503")],
)
def test_gemini_api_errors_are_mocked_logged_and_reraised(
    monkeypatch, tmp_path, status_code, expected_error_type
):
    monkeypatch.setattr(ask_module, "METRICS_LOG", tmp_path / "metrics.csv")
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    api_error = ask_module.errors.APIError(
        status_code,
        {"message": "sensitive test detail"},
    )
    monkeypatch.setattr(
        ask_module.genai,
        "Client",
        lambda **kwargs: FakeClient(error=api_error),
    )

    with pytest.raises(ask_module.errors.APIError):
        ask_module.ask("質問")

    with ask_module.METRICS_LOG.open(encoding="utf-8", newline="") as file:
        row = next(csv.DictReader(file))
    assert row["status"] == "failure"
    assert row["error_type"] == expected_error_type
    assert row["api_status_code"] == str(status_code)
    assert all("sensitive test detail" not in value for value in row.values())


def test_temperature_is_passed_as_supported_sdk_config(monkeypatch, tmp_path):
    monkeypatch.setattr(ask_module, "METRICS_LOG", tmp_path / "metrics.csv")
    client = FakeClient([_response()])
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(ask_module.genai, "Client", lambda **kwargs: client)

    ask_module._request("質問", temperature=0.4)

    assert client.calls[0]["config"].temperature == 0.4


def test_response_time_uses_perf_counter(monkeypatch, tmp_path):
    monkeypatch.setattr(ask_module, "METRICS_LOG", tmp_path / "metrics.csv")
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(
        ask_module.genai,
        "Client",
        lambda **kwargs: FakeClient([_response()]),
    )
    times = iter([10.0, 11.25])
    monkeypatch.setattr(ask_module, "perf_counter", lambda: next(times))

    ask_module.ask("質問")

    with ask_module.METRICS_LOG.open(encoding="utf-8", newline="") as file:
        row = next(csv.DictReader(file))
    assert float(row["response_time_seconds"]) == 1.25


def test_error_types_distinguish_rate_limit_and_service_unavailable():
    assert ask_module._error_type(SimpleNamespace(code=429)) == "rate_limit_429"
    assert ask_module._error_type(SimpleNamespace(code=503)) == "service_unavailable_503"
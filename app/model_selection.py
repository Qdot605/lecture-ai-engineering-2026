"""Evaluate model evidence against configurable hard gates and create a Markdown report."""

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

DEFAULT_CRITERIA = Path("data/model_evaluation_criteria.json")
DEFAULT_CASES = Path("data/model_evaluation_cases.json")
VALID_REVIEW_STATUSES = {"pass", "fail", "inconclusive", "not_evaluated"}
VALID_EXECUTION_STATUSES = {"success", "failure", "not_run"}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _review_statuses(
    rows: list[dict[str, Any]], criterion: str
) -> list[tuple[str, str]]:
    statuses = []
    for row in rows:
        status = row.get("evaluation", {}).get(criterion, {})
        if isinstance(status, dict):
            value = status.get("status", "not_evaluated")
            rationale = status.get("rationale", "")
        else:
            value, rationale = "not_evaluated", ""
        if value not in VALID_REVIEW_STATUSES:
            raise ValueError(
                f"Invalid review status {value!r} for {criterion}; "
                f"expected one of {sorted(VALID_REVIEW_STATUSES)}"
            )
        statuses.append((value, rationale))
    return statuses


def _gate_status(statuses: list[str]) -> str:
    if "fail" in statuses:
        return "Fail"
    if "inconclusive" in statuses:
        return "Inconclusive"
    if not statuses or all(status == "not_evaluated" for status in statuses):
        return "Not Evaluated"
    if all(status == "pass" for status in statuses):
        return "Pass"
    return "Inconclusive"


def _markdown_cell(value: Any) -> str:
    return str(value).replace("|", r"\|").replace("\n", "<br>")


def _review_gate(
    rows: list[dict[str, Any]],
    case_ids: list[str],
    criterion: str,
    minimum_trials: int,
) -> str:
    if not case_ids:
        return "Not Evaluated"
    statuses_by_case = []
    for case_id in case_ids:
        case_rows = [row for row in rows if row["case_id"] == case_id]
        statuses = [
            status
            for status, _ in _review_statuses(case_rows, criterion)
            if status != "not_evaluated"
        ]
        statuses_by_case.append(statuses)
    all_statuses = [status for statuses in statuses_by_case for status in statuses]
    if "fail" in all_statuses:
        return "Fail"
    if "inconclusive" in all_statuses:
        return "Inconclusive"
    if not all_statuses:
        return "Not Evaluated"
    if any(
        len(statuses) < minimum_trials or any(status != "pass" for status in statuses)
        for statuses in statuses_by_case
    ):
        return "Inconclusive"
    return "Pass"


def evaluate(
    evidence: dict[str, Any],
    criteria: dict[str, Any],
    cases: dict[str, Any],
) -> dict[str, Any]:
    case_map = {case["case_id"]: case for case in cases["cases"]}
    if len(case_map) != len(cases["cases"]):
        raise ValueError("Evaluation case IDs must be unique")
    results = evidence.get("results", [])
    unknown_case_ids = sorted({row["case_id"] for row in results} - case_map.keys())
    if unknown_case_ids:
        raise ValueError(f"Evidence contains unknown case IDs: {', '.join(unknown_case_ids)}")

    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        if not row.get("model") or row.get("case_id") is None:
            raise ValueError("Each evidence result must include model and case_id")
        status = row.get("execution", {}).get("status")
        if status not in VALID_EXECUTION_STATUSES:
            raise ValueError(f"Invalid execution status {status!r}")
        duration = row.get("execution", {}).get("response_time_seconds")
        if duration is not None and (
            not isinstance(duration, (int, float))
            or isinstance(duration, bool)
            or not math.isfinite(duration)
            or duration < 0
        ):
            raise ValueError("response_time_seconds must be a finite non-negative number")
        by_model[row["model"]].append(row)
    models = list(dict.fromkeys(evidence.get("models", []) + list(by_model)))
    for model in models:
        by_model.setdefault(model, [])
    planned_case_ids = evidence.get("planned_case_ids") or list(
        dict.fromkeys(row["case_id"] for row in results)
    )
    unknown_planned_ids = sorted(set(planned_case_ids) - case_map.keys())
    if unknown_planned_ids:
        raise ValueError(f"Planned case IDs are unknown: {', '.join(unknown_planned_ids)}")

    max_seconds = criteria["acceptance_criteria"]["max_response_time_seconds"]
    minimum_trials = criteria["acceptance_criteria"].get("minimum_trials_per_case", 1)
    if (
        not isinstance(max_seconds, (int, float))
        or isinstance(max_seconds, bool)
        or not math.isfinite(max_seconds)
        or max_seconds <= 0
    ):
        raise ValueError("max_response_time_seconds must be a positive finite number")
    if not isinstance(minimum_trials, int) or isinstance(minimum_trials, bool) or minimum_trials < 1:
        raise ValueError("minimum_trials_per_case must be a positive integer")
    summaries = {}
    for model, rows in by_model.items():
        successful = [
            row
            for row in rows
            if row.get("execution", {}).get("status") == "success"
        ]
        durations = [
            row["execution"]["response_time_seconds"]
            for row in successful
            if isinstance(row.get("execution", {}).get("response_time_seconds"), (int, float))
            and not isinstance(row["execution"]["response_time_seconds"], bool)
            and math.isfinite(row["execution"]["response_time_seconds"])
            and row["execution"]["response_time_seconds"] >= 0
        ]
        latency_case_ids = [
            case_id
            for case_id in planned_case_ids
            if case_map[case_id].get("category") != "api_error_handling"
        ]
        latency_statuses = []
        latency_observations_over_threshold = []
        failed_request_times = []
        for case_id in latency_case_ids:
            case_rows = [row for row in rows if row["case_id"] == case_id]
            case_successes = [
                row for row in case_rows
                if row.get("execution", {}).get("status") == "success"
                and isinstance(row.get("execution", {}).get("response_time_seconds"), (int, float))
                and not isinstance(row["execution"]["response_time_seconds"], bool)
                and math.isfinite(row["execution"]["response_time_seconds"])
                and row["execution"]["response_time_seconds"] >= 0
            ]
            case_durations = [row["execution"]["response_time_seconds"] for row in case_successes]
            latency_observations_over_threshold.extend(
                {
                    "case_id": case_id,
                    "run": row.get("run"),
                    "response_time_seconds": row["execution"]["response_time_seconds"],
                }
                for row in case_successes
                if row["execution"]["response_time_seconds"] > max_seconds
            )
            failed_request_times.extend(
                {
                    "case_id": case_id,
                    "run": row.get("run"),
                    "response_time_seconds": row["execution"]["response_time_seconds"],
                    "error_type": row["execution"].get("error_type"),
                }
                for row in case_rows
                if row.get("execution", {}).get("status") == "failure"
                and isinstance(row.get("execution", {}).get("response_time_seconds"), (int, float))
            )
            if any(duration > max_seconds for duration in case_durations):
                latency_statuses.append("fail")
            elif len(case_durations) >= minimum_trials:
                latency_statuses.append("pass")
            else:
                latency_statuses.append("inconclusive")

        if not latency_case_ids:
            latency_gate = "Not Evaluated"
        else:
            latency_gate = _gate_status(latency_statuses)

        api_case_ids = [
            case_id for case_id in planned_case_ids
            if case_map[case_id].get("category") == "api_error_handling"
        ]
        unknown_case_ids = [
            case_id for case_id in planned_case_ids
            if case_map[case_id].get("category") == "unknown_information"
        ]
        api_gate = _review_gate(rows, api_case_ids, "api_error_handling", minimum_trials)
        unknown_gate = _review_gate(rows, unknown_case_ids, "unknown_information", minimum_trials)
        quality_reviews = Counter()
        for row in rows:
            for criterion, review in row.get("evaluation", {}).items():
                status = review.get("status", "not_evaluated") if isinstance(review, dict) else "not_evaluated"
                if status not in VALID_REVIEW_STATUSES:
                    raise ValueError(
                        f"Invalid review status {status!r} for {criterion}; "
                        f"expected one of {sorted(VALID_REVIEW_STATUSES)}"
                    )
                quality_reviews[status] += 1
        gates = {
            "response_time": latency_gate,
            "api_error_handling": (
                api_gate
                if criteria["acceptance_criteria"].get("api_error_handling_required", True)
                else "Not Required"
            ),
            "unknown_information": (
                unknown_gate
                if criteria["acceptance_criteria"].get("unknown_information_required", True)
                else "Not Required"
            ),
        }

        token_totals = []
        for row in successful:
            execution = row.get("execution", {})
            input_tokens, output_tokens = execution.get("input_tokens"), execution.get("output_tokens")
            if (
                isinstance(input_tokens, (int, float))
                and not isinstance(input_tokens, bool)
                and math.isfinite(input_tokens)
                and input_tokens >= 0
                and isinstance(output_tokens, (int, float))
                and not isinstance(output_tokens, bool)
                and math.isfinite(output_tokens)
                and output_tokens >= 0
            ):
                token_totals.append(input_tokens + output_tokens)

        if "Fail" in gates.values():
            eligibility = "不適合（本実験で必須要件違反を観測）"
        elif all(status in {"Pass", "Not Required"} for status in gates.values()):
            eligibility = "適合（このテストセットの範囲）"
        else:
            eligibility = "未確定（評価不足または判定不能）"

        summaries[model] = {
            "gates": gates,
            "eligibility": eligibility,
            "successful_count": len(successful),
            "failure_count": sum(
                row.get("execution", {}).get("status") == "failure" for row in rows
            ),
            "not_run_count": sum(
                row.get("execution", {}).get("status") == "not_run" for row in rows
            ),
            "response_times": durations,
            "failed_request_times": failed_request_times,
            "latency_observations_over_threshold": latency_observations_over_threshold,
            "planned_case_ids": planned_case_ids,
            "minimum_trials_per_case": minimum_trials,
            "mean_seconds": statistics.mean(durations) if durations else None,
            "median_seconds": statistics.median(durations) if durations else None,
            "max_seconds": max(durations) if durations else None,
            "median_tokens": statistics.median(token_totals) if token_totals else None,
            "token_sample_count": len(token_totals),
            "missing_token_count": len(successful) - len(token_totals),
            "quality_review_counts": dict(quality_reviews),
            "error_types": dict(
                Counter(
                    row.get("execution", {}).get("error_type")
                    for row in rows
                    if row.get("execution", {}).get("error_type")
                )
            ),
            "rows": rows,
        }

    comparable = [
        (model, summary)
        for model, summary in summaries.items()
        if summary["eligibility"].startswith("適合")
    ]
    conditions = [json.dumps(row.get("conditions", evidence.get("conditions", {})), sort_keys=True)
                  for row in results]
    conditions_match = len(set(conditions)) <= 1
    if comparable and conditions_match and all(item[1]["median_seconds"] is not None for item in comparable):
        selected_model = min(
            comparable,
            key=lambda item: (
                item[1]["median_seconds"],
                item[1]["median_tokens"] if item[1]["median_tokens"] is not None else float("inf"),
                item[0],
            ),
        )[0]
        selection_reason = (
            "Hard Gateをすべて通過した候補のうち、応答時間中央値が最小のモデルを選択しました。"
            "同値の場合は中央値トークン数で比較しています。"
        )
    elif comparable and not conditions_match:
        selected_model = None
        selection_reason = "モデル間で実行条件が異なるため、比較可能な条件に揃えるまで選定を保留します。"
    elif comparable:
        selected_model = None
        selection_reason = "Hard Gate通過モデルの応答時間を測定できていないため、選定を保留します。"
    else:
        selected_model = None
        selection_reason = "Hard Gateをすべて通過し、必要な測定が揃ったモデルがありません。"

    return {
        "purpose": evidence.get("purpose", "モデル候補の評価"),
        "source": evidence.get("source", "unspecified"),
        "conditions": evidence.get("conditions", {}),
        "environment": evidence.get("environment", {}),
        "max_response_time_seconds": max_seconds,
        "minimum_trials_per_case": minimum_trials,
        "summaries": summaries,
        "selected_model": selected_model,
        "selection_reason": selection_reason,
        "conditions_match": conditions_match,
        "cases": case_map,
        "evidence": evidence,
    }


def render_report(evaluation: dict[str, Any]) -> str:
    lines = [
        "# LLMモデル選定評価レポート",
        "",
        "> このレポートは指定された評価ケースと入力された証拠の範囲に限るもので、",
        "> 実運用上の品質保証ではありません。",
        "",
        "## 1. 実験目的",
        "",
        evaluation["purpose"],
        "",
        "## 2. 使用モデル",
        "",
    ]
    for model in evaluation["summaries"]:
        lines.append(f"- `{_markdown_cell(model)}`")
    if not evaluation["summaries"]:
        lines.append("- なし")

    lines.extend(
        [
            "",
            "## 3. 評価条件",
            "",
            f"- 共通条件: `{json.dumps(evaluation['conditions'], ensure_ascii=False, sort_keys=True)}`",
            f"- 応答時間の仮閾値: {evaluation['max_response_time_seconds']}秒以内",
            f"- Passに必要な最低成功試行数: ケース・モデルごとに{evaluation['minimum_trials_per_case']}回",
            f"- 実行環境: `{json.dumps(evaluation['environment'], ensure_ascii=False, sort_keys=True)}`",
            f"- Evidence source: `{evaluation['source']}`",
            "- 品質判定はケースごとの明示的なレビュー結果を使用し、自動的なキーワード採点は行いません。",
            "",
            "## 4. 測定結果",
            "",
            "| モデル | 成功数 | 失敗数 | 未実行数 | 平均秒 | 中央値秒 | 最大秒 | 中央値トークン合計 | トークン欠損数 | 品質レビュー件数 | エラー種別 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for model, summary in evaluation["summaries"].items():
        fmt = lambda value: "未測定" if value is None else f"{value:.3f}"
        tokens = "未測定" if summary["median_tokens"] is None else f"{summary['median_tokens']:g}"
        errors = ", ".join(
            f"{key}: {value}" for key, value in summary["error_types"].items()
        ) or "なし"
        reviews = ", ".join(
            f"{key}: {value}" for key, value in summary["quality_review_counts"].items()
        ) or "未評価"
        lines.append(
            f"| `{_markdown_cell(model)}` | {summary['successful_count']} "
            f"| {summary['failure_count']} | {summary['not_run_count']} "
            f"| {fmt(summary['mean_seconds'])} | {fmt(summary['median_seconds'])} "
            f"| {fmt(summary['max_seconds'])} | {tokens} "
            f"| {summary['missing_token_count']} | {reviews} | {errors} |"
        )

    lines.extend(
        [
            "",
            "## 5. Hard Gateの判定",
            "",
            "| モデル | 応答時間 | APIエラー処理 | 未知情報 | 総合 |",
            "|---|---|---|---|---|",
        ]
    )
    for model, summary in evaluation["summaries"].items():
        gates = summary["gates"]
        lines.append(
            f"| `{_markdown_cell(model)}` | {gates['response_time']} | {gates['api_error_handling']} "
            f"| {gates['unknown_information']} | {summary['eligibility']} |"
        )

    lines.extend(["", "### 閾値超過の観測（観測事実。単独で運用全体の品質保証を意味しない）", ""])
    for model, summary in evaluation["summaries"].items():
        exceedances = summary["latency_observations_over_threshold"]
        if exceedances:
            details = ", ".join(
                f"{item['case_id']} run {item['run']}: {item['response_time_seconds']:.3f}s"
                for item in exceedances
            )
            lines.append(f"- `{_markdown_cell(model)}`: {details}")
        else:
            lines.append(f"- `{_markdown_cell(model)}`: 超過観測なし")

    lines.extend(["", "## 6. モデル選定結果", ""])
    if evaluation["selected_model"]:
        lines.append(f"選定候補: **`{evaluation['selected_model']}`**")
    else:
        lines.append("選定候補: **なし（追加評価または条件統一が必要）**")
    lines.extend(["", evaluation["selection_reason"], "", "## 7. 判断理由", ""])
    lines.append(
        "必須要件のFailが1つでもある候補は不適合とし、Not Evaluated / InconclusiveはPass扱いしていません。"
    )
    lines.extend(["", "## 8. 評価の限界", ""])
    lines.extend(
        [
            "- 少数のケースによる評価であり、実運用の品質・安全性・可用性を保証しません。",
            "- 品質判定は記録された人手レビューに依存します。曖昧な回答はInconclusiveとして別途レビューしてください。",
            "- 品質レビュー件数は人手評価ラベルの集計であり、連続的な品質スコアではありません。",
            "- APIエラー処理はエラーケースの証拠がなければNot Evaluatedです。通常応答だけで処理能力を推定しません。",
            "- 失敗リクエストの経過時間はEvidenceに残しますが、成功応答の応答時間統計とは分けており、APIエラーゲートで別評価が必要です。",
            "- 応答時間とトークン数は実行環境、時刻、レート制限、モデルの負荷に影響されます。",
            "- 選定順位はHard Gate通過候補の中央値応答時間を優先し、同値時のみ中央値トークン数で比較する限定的なルールです。",
        ]
    )
    if evaluation["source"] == "mock":
        lines.append("- 入力データはモック例です。モデル実測値やモデル間の優劣を示すものではありません。")

    lines.extend(
        [
            "",
            "## Evidence",
            "",
            "| Case ID | モデル | 期待動作 | 条件 | 実行結果 | 評価 | 判定根拠 |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for model, summary in evaluation["summaries"].items():
        for row in summary["rows"]:
            execution = row.get("execution", {})
            case_id = row["case_id"]
            expected_behavior = evaluation["cases"][case_id].get("expected_behavior", "")
            conditions = row.get("conditions", evaluation["conditions"])
            result = {
                key: execution.get(key)
                for key in (
                    "status",
                    "response_time_seconds",
                    "input_tokens",
                    "output_tokens",
                    "error_type",
                    "api_status_code",
                )
                if key in execution
            }
            result["run"] = row.get("run")
            result["executed_at"] = row.get("executed_at")
            if "answer" in row:
                result["answer"] = row["answer"]
            review = row.get("evaluation", {})
            review_text = "; ".join(
                f"{name}: {entry.get('status', 'not_evaluated')} ({entry.get('rationale', '')})"
                for name, entry in review.items()
            ) or "未評価"
            lines.append(
                f"| `{_markdown_cell(case_id)}` | `{_markdown_cell(model)}` "
                f"| {_markdown_cell(expected_behavior)} "
                f"| `{_markdown_cell(json.dumps(conditions, ensure_ascii=False, sort_keys=True))}` "
                f"| `{_markdown_cell(json.dumps(result, ensure_ascii=False, sort_keys=True))}` "
                f"| `{_markdown_cell(review_text)}` | {_markdown_cell(row.get('evidence_note', ''))} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="評価証拠からHard Gateとモデル選定レポートを作成する")
    parser.add_argument("--input", required=True, type=Path, help="評価結果JSON")
    parser.add_argument("--criteria", type=Path, default=DEFAULT_CRITERIA)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=Path("learning_log/model_selection_report.md"))
    args = parser.parse_args()

    result = evaluate(_read_json(args.input), _read_json(args.criteria), _read_json(args.cases))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_report(result), encoding="utf-8")
    print(f"評価レポートを出力しました: {args.output}")
    print(f"選定候補: {result['selected_model'] or 'なし'}")


if __name__ == "__main__":
    main()

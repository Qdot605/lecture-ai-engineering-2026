"""Create and summarize a minimal, allowlisted public view of experiment evidence."""

import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

PUBLIC_ROW_FIELDS = {
    "experiment_id",
    "model",
    "question_id",
    "run",
    "executed_at",
    "temperature",
    "response_time_seconds",
    "input_tokens",
    "output_tokens",
    "execution_status",
    "http_status",
    "question",
    "answer",
}

_SENSITIVE_PATTERNS = (
    re.compile(r"AIza[0-9A-Za-z_-]{30,}"),
    re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~-]{24,}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    re.compile(r"(?<!\d)0\d{1,4}-\d{1,4}-\d{3,4}(?!\d)"),
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def project_experiment(source: dict[str, Any]) -> dict[str, Any]:
    """Project source records into a strict public schema, omitting raw error details."""
    experiment_id = source["experiment_id"]
    results = []
    for row in source["results"]:
        execution = row.get("execution", {})
        conditions = row.get("conditions", source.get("conditions", {}))
        results.append(
            {
                "experiment_id": experiment_id,
                "model": row["model"],
                "question_id": row["case_id"],
                "run": row.get("run"),
                "executed_at": row.get("executed_at"),
                "temperature": conditions.get("temperature"),
                "response_time_seconds": execution.get("response_time_seconds"),
                "input_tokens": execution.get("input_tokens"),
                "output_tokens": execution.get("output_tokens"),
                "execution_status": execution.get("status"),
                "http_status": execution.get("api_status_code"),
                "question": row.get("question"),
                "answer": row.get("answer"),
            }
        )
    public = {
        "schema_version": 1,
        "purpose": source.get("purpose"),
        "source": source.get("source"),
        "experiment_id": experiment_id,
        "models": source.get("models", []),
        "planned_question_ids": source.get("planned_case_ids", []),
        "planned_runs_per_model_question": source.get("planned_runs_per_model_case"),
        "conditions": source.get("conditions", {}),
        "environment": {
            key: source.get("environment", {}).get(key)
            for key in ("google_genai_version", "python_version", "platform")
            if key in source.get("environment", {})
        },
        "results": results,
    }
    validate_public_evidence(public)
    return public


def validate_public_evidence(evidence: dict[str, Any]) -> None:
    """Reject unexpected fields, missing required fields, or recognizable personal secrets."""
    for row in evidence.get("results", []):
        if set(row) != PUBLIC_ROW_FIELDS:
            raise ValueError("Public evidence row does not match the allowlisted schema")
    serialized = json.dumps(evidence, ensure_ascii=False)
    if any(pattern.search(serialized) for pattern in _SENSITIVE_PATTERNS):
        raise ValueError("Public evidence contains a recognizable secret or personal identifier")


def export_public_evidence(source_paths: list[Path], output_path: Path) -> dict[str, Any]:
    """Export source files without modifying them, and refuse to overwrite a source."""
    resolved_sources = {path.resolve() for path in source_paths}
    if output_path.resolve() in resolved_sources:
        raise ValueError("Public evidence output must not overwrite a source evidence file")
    experiments = [project_experiment(_read_json(path)) for path in source_paths]
    experiment_ids = [experiment["experiment_id"] for experiment in experiments]
    if len(experiment_ids) != len(set(experiment_ids)):
        raise ValueError("Source experiment IDs must be unique")
    public = {
        "schema_version": 1,
        "experiments": experiments,
    }
    validate_public_evidence(
        {
            "results": [
                row for experiment in experiments for row in experiment["results"]
            ]
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(public, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return public


def summarize_public_evidence(
    evidence: dict[str, Any],
) -> dict[tuple[str, str], Any]:
    """Recompute success, latency, token, and HTTP-status aggregates from public rows."""
    summaries: dict[str, Any] = {}
    for experiment in evidence["experiments"]:
        for model in experiment["models"]:
            model_rows = [
                row for row in experiment["results"] if row["model"] == model
            ]
            question_summaries = {}
            for question_id in experiment["planned_question_ids"]:
                rows = [row for row in model_rows if row["question_id"] == question_id]
                question_summaries[question_id] = summarize_rows(rows)
            summaries[(experiment["experiment_id"], model)] = {
                "planned_count": len(model_rows),
                **summarize_rows(model_rows),
                "questions": question_summaries,
            }
    return summaries


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if row["execution_status"] == "success"]
    failures = [row for row in rows if row["execution_status"] == "failure"]
    durations = [
        row["response_time_seconds"]
        for row in successful
        if _valid_nonnegative_number(row["response_time_seconds"])
    ]
    input_tokens = [
        row["input_tokens"]
        for row in successful
        if _valid_nonnegative_number(row["input_tokens"])
    ]
    output_tokens = [
        row["output_tokens"]
        for row in successful
        if _valid_nonnegative_number(row["output_tokens"])
    ]
    return {
        "success_count": len(successful),
        "failure_count": len(failures),
        "http_status_counts": _counts(
            str(row["http_status"]) for row in failures if row["http_status"] is not None
        ),
        "latency_mean_seconds": statistics.mean(durations) if durations else None,
        "latency_median_seconds": statistics.median(durations) if durations else None,
        "latency_max_seconds": max(durations) if durations else None,
        "average_input_tokens": statistics.mean(input_tokens) if input_tokens else None,
        "average_output_tokens": statistics.mean(output_tokens) if output_tokens else None,
    }


def _valid_nonnegative_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for value in values:
        counts[value] += 1
    return dict(counts)

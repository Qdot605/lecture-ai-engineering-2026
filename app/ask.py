"""第1回: Gemini に質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask --chat
    uv run python -m app.ask --compare --models MODEL_A MODEL_B --questions "質問1" "質問2"

単発質問を保ったまま、会話モードと複数モデル・質問の比較を追加する。
"""

import argparse
import csv
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from google import genai
from google.genai import errors, types
from google.genai import __version__ as genai_version

from app.config import MODEL, api_error_message, get_api_key

DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"
METRICS_LOG = Path("learning_log/ask_metrics.csv")
COMPARISON_LOG = Path("learning_log/ask_comparisons.csv")
EVIDENCE_LOG = Path("learning_log/ask_evidence.json")
METRICS_FIELDS = (
    "timestamp",
    "experiment_id",
    "question_id",
    "run",
    "model",
    "response_time_seconds",
    "input_tokens",
    "output_tokens",
    "status",
    "error_type",
    "api_status_code",
)
COMPARISON_FIELDS = (
    "timestamp",
    "experiment_id",
    "question_id",
    "run",
    "model",
    "temperature",
    "response_time_seconds",
    "input_tokens",
    "output_tokens",
    "sdk_version",
    "python_version",
    "platform",
    "question",
    "answer",
    "status",
    "error_type",
    "api_status_code",
    "not_run_reason",
)


def _append_csv(path: Path, fields: tuple[str, ...], row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    _validate_csv_schema(path, fields)
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _validate_csv_schema(path: Path, fields: tuple[str, ...]) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open(encoding="utf-8", newline="") as file:
        existing_fields = next(csv.reader(file), [])
    if tuple(existing_fields) != fields:
        raise ValueError(
            f"CSV schema mismatch for {path}; choose a new log file or archive the old one"
        )


def _error_type(error: Exception) -> str:
    status_code = getattr(error, "code", None)
    if str(status_code) == "429":
        return "rate_limit_429"
    if str(status_code) == "503":
        return "service_unavailable_503"
    if isinstance(error, TimeoutError) or "timeout" in type(error).__name__.lower():
        return "timeout"
    if isinstance(error, errors.APIError):
        return "api_error"
    return type(error).__name__


def _request(
    question: str,
    *,
    model: str = MODEL,
    contents: str | list[types.Content] | None = None,
    client=None,
    temperature: float | None = None,
    question_id: str | None = None,
    experiment_id: str = "",
    run: int = 1,
    measurement: dict | None = None,
):
    if client is None:
        client = genai.Client(api_key=get_api_key())
    question_id = question_id or str(uuid4())
    request_contents = question if contents is None else contents
    config = types.GenerateContentConfig(temperature=temperature) if temperature is not None else None

    start = perf_counter()
    try:
        if config is None:
            response = client.models.generate_content(
                model=model,
                contents=request_contents,
            )
        else:
            response = client.models.generate_content(
                model=model,
                contents=request_contents,
                config=config,
            )
    except Exception as error:
        elapsed = perf_counter() - start
        if measurement is not None:
            measurement.update(
                {
                    "response_time_seconds": elapsed,
                    "status": "failure",
                    "error_type": _error_type(error),
                    "api_status_code": getattr(error, "code", ""),
                }
            )
        _append_csv(
            METRICS_LOG,
            METRICS_FIELDS,
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "experiment_id": experiment_id,
                "question_id": question_id,
                "run": run,
                "model": model,
                "response_time_seconds": elapsed,
                "status": "failure",
                "error_type": _error_type(error),
                "api_status_code": getattr(error, "code", ""),
            },
        )
        raise

    elapsed = perf_counter() - start
    usage = getattr(response, "usage_metadata", None)
    input_tokens = getattr(usage, "prompt_token_count", None) if usage is not None else None
    output_tokens = getattr(usage, "candidates_token_count", None) if usage is not None else None
    if measurement is not None:
        measurement.update(
            {
                "response_time_seconds": elapsed,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "status": "success",
            }
        )
    _append_csv(
        METRICS_LOG,
        METRICS_FIELDS,
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "experiment_id": experiment_id,
            "question_id": question_id,
            "run": run,
            "model": model,
            "response_time_seconds": elapsed,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "status": "success",
        },
    )
    display_input = input_tokens if input_tokens is not None else "不明"
    display_output = output_tokens if output_tokens is not None else "不明"
    print(f"[トークン数] 入力: {display_input}, 出力: {display_output}")
    return response


def ask(question: str) -> str:
    response = _request(question)
    return response.text


class Conversation:
    def __init__(
        self,
        *,
        model: str = MODEL,
        temperature: float | None = None,
        client=None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.client = client or genai.Client(api_key=get_api_key())
        self.history: list[types.Content] = []

    def ask(self, question: str) -> str:
        user_message = types.Content(
            role="user",
            parts=[types.Part.from_text(text=question)],
        )
        response = _request(
            question,
            model=self.model,
            contents=[*self.history, user_message],
            client=self.client,
            temperature=self.temperature,
        )
        answer = response.text
        self.history.extend(
            [
                user_message,
                types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=answer or "")],
                ),
            ]
        )
        return answer

    def reset(self) -> None:
        self.history.clear()


def compare(
    questions: list[str],
    models: list[str],
    *,
    runs: int = 3,
    temperature: float | None = None,
    evidence_path: Path = EVIDENCE_LOG,
    evidence_source: str = "gemini_api",
) -> None:
    if runs < 1:
        raise ValueError("runs は1以上を指定してください")
    if not questions or not models:
        raise ValueError("questions と models はそれぞれ1件以上必要です")

    _validate_csv_schema(METRICS_LOG, METRICS_FIELDS)
    _validate_csv_schema(COMPARISON_LOG, COMPARISON_FIELDS)
    experiment_id = str(uuid4())
    question_ids = [f"Q{index}" for index in range(1, len(questions) + 1)]
    client = genai.Client(api_key=get_api_key())
    schedule = []
    for question_id, question in zip(question_ids, questions):
        for run in range(1, runs + 1):
            offset = (run - 1) % len(models)
            ordered_models = models[offset:] + models[:offset]
            for model in ordered_models:
                schedule.append((question_id, question, run, model))

    for index, (question_id, question, run, model) in enumerate(schedule):
        timestamp = datetime.now(timezone.utc).isoformat()
        measurement = {}
        print(f"\n[比較] model={model}, run={run}/{runs}, question_id={question_id}")
        try:
            response = _request(
                question,
                model=model,
                client=client,
                temperature=temperature,
                question_id=question_id,
                experiment_id=experiment_id,
                run=run,
                measurement=measurement,
            )
        except Exception as error:
            error_type = _error_type(error)
            _append_csv(
                COMPARISON_LOG,
                COMPARISON_FIELDS,
                {
                    "timestamp": timestamp,
                    "experiment_id": experiment_id,
                    "question_id": question_id,
                    "run": run,
                    "model": model,
                    "temperature": temperature,
                    "response_time_seconds": measurement["response_time_seconds"],
                    "sdk_version": genai_version,
                    "python_version": platform.python_version(),
                    "platform": sys.platform,
                    "question": question,
                    "status": "failure",
                    "error_type": error_type,
                    "api_status_code": getattr(error, "code", ""),
                },
            )
            print(f"[失敗] {error_type}")
            if error_type == "rate_limit_429":
                for pending_case_id, pending_question, pending_run, pending_model in schedule[index + 1 :]:
                    _append_csv(
                        COMPARISON_LOG,
                        COMPARISON_FIELDS,
                        {
                            "experiment_id": experiment_id,
                            "question_id": pending_case_id,
                            "run": pending_run,
                            "model": pending_model,
                            "temperature": temperature,
                            "sdk_version": genai_version,
                            "python_version": platform.python_version(),
                            "platform": sys.platform,
                            "question": pending_question,
                            "status": "not_run",
                            "not_run_reason": "stopped_after_rate_limit_429",
                        },
                    )
                print("[停止] 429レート制限を検出したため、残りの試行は実行しません")
                break
            continue

        input_tokens = measurement["input_tokens"]
        output_tokens = measurement["output_tokens"]
        answer = response.text
        _append_csv(
            COMPARISON_LOG,
            COMPARISON_FIELDS,
            {
                "timestamp": timestamp,
                "experiment_id": experiment_id,
                "question_id": question_id,
                "run": run,
                "model": model,
                "temperature": temperature,
                "response_time_seconds": measurement["response_time_seconds"],
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "sdk_version": genai_version,
                "python_version": platform.python_version(),
                "platform": sys.platform,
                "question": question,
                "answer": answer,
                "status": "success",
            },
        )
        print(answer)
    comparison_csv_to_evidence(
        COMPARISON_LOG,
        evidence_path,
        experiment_id=experiment_id,
        models=models,
        question_ids=question_ids,
        runs=runs,
        temperature=temperature,
        source=evidence_source,
    )
    print(f"\n[比較ログ] {COMPARISON_LOG}")
    print(f"[計測ログ] {METRICS_LOG}")
    print(f"[Evidence JSON] {evidence_path}")


def comparison_csv_to_evidence(
    csv_path: Path,
    output_path: Path,
    *,
    experiment_id: str,
    models: list[str],
    question_ids: list[str],
    runs: int,
    temperature: float | None,
    source: str = "gemini_api",
) -> dict:
    with csv_path.open(encoding="utf-8", newline="") as file:
        rows = [
            row
            for row in csv.DictReader(file)
            if row.get("experiment_id") == experiment_id
        ]
    if not rows:
        raise ValueError(f"No comparison rows found for experiment {experiment_id}")

    results = []
    for row in rows:
        execution = {"status": row["status"]}
        if row.get("response_time_seconds"):
            execution["response_time_seconds"] = float(row["response_time_seconds"])
        for field in ("input_tokens", "output_tokens"):
            if row.get(field):
                execution[field] = int(row[field])
        if row.get("error_type"):
            execution["error_type"] = row["error_type"]
        if row.get("not_run_reason"):
            execution["not_run_reason"] = row["not_run_reason"]
        if row.get("api_status_code"):
            execution["api_status_code"] = int(row["api_status_code"])
        results.append(
            {
                "executed_at": row["timestamp"],
                "case_id": row["question_id"],
                "run": int(row["run"]),
                "model": row["model"],
                "conditions": {"temperature": temperature},
                "execution": execution,
                "question": row["question"],
                "answer": row.get("answer", ""),
                "evaluation": {},
                "evidence_note": "API実行結果。品質レビューは未実施。",
            }
        )

    evidence = {
        "purpose": "Day1 Geminiモデル比較実験",
        "source": source,
        "experiment_id": experiment_id,
        "models": models,
        "planned_case_ids": question_ids,
        "planned_runs_per_model_case": runs,
        "conditions": {"temperature": temperature},
        "environment": {
            "google_genai_version": rows[0]["sdk_version"],
            "python_version": rows[0]["python_version"],
            "platform": rows[0]["platform"],
        },
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return evidence


def _chat(model: str, temperature: float | None) -> None:
    conversation = Conversation(model=model, temperature=temperature)
    print("会話モードです。/reset で履歴を初期化、/exit で終了します。")
    while True:
        try:
            question = input("質問> ")
        except EOFError:
            break
        if question.strip() == "/exit":
            break
        if question.strip() == "/reset":
            conversation.reset()
            print("[会話履歴を初期化しました]")
            continue
        if not question.strip():
            continue
        try:
            print(conversation.ask(question))
        except errors.APIError as error:
            print(api_error_message(error.code, error.message))


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemini に質問する")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--chat", action="store_true", help="会話モード（/reset で履歴を初期化）")
    parser.add_argument("--compare", action="store_true", help="モデル・質問の比較を実行し結果を保存する")
    parser.add_argument("--models", nargs="+", help="比較するモデル名（省略時は AIE_MODEL のモデル）")
    parser.add_argument("--questions", nargs="+", help="比較する質問")
    parser.add_argument("--runs", type=int, help="各モデル・質問の実行回数（既定: 3）")
    parser.add_argument("--temperature", type=float, help="生成 temperature（Gemini SDK 対応範囲: 0.0〜2.0）")
    parser.add_argument("--evidence-output", type=Path, default=EVIDENCE_LOG, help="比較Evidence JSONの出力先")
    args = parser.parse_args()

    if args.temperature is not None and not 0 <= args.temperature <= 2:
        parser.error("--temperature は0.0から2.0の範囲で指定してください")
    if args.chat and args.compare:
        parser.error("--chat と --compare は同時に指定できません")
    if args.chat and args.question:
        parser.error("--chat では位置引数 question を指定しないでください")
    if args.compare and args.runs is not None and args.runs < 1:
        parser.error("--runs は1以上を指定してください")

    print(f"[model] {MODEL}")
    if args.compare:
        questions = args.questions or ([args.question] if args.question else [DEFAULT_QUESTION])
        compare(
            questions,
            args.models or [MODEL],
            runs=args.runs or 3,
            temperature=args.temperature,
            evidence_path=args.evidence_output,
        )
        return
    if args.models or args.questions or args.runs is not None or args.evidence_output != EVIDENCE_LOG:
        parser.error("--models、--questions、--runs は --compare と一緒に指定してください")
    if args.chat:
        _chat(MODEL, args.temperature)
        return

    question = args.question or DEFAULT_QUESTION
    print(f"[質問] {question}\n")
    try:
        if args.temperature is None:
            print(ask(question))
        else:
            print(_request(question, temperature=args.temperature).text)
    except errors.APIError as error:
        print(api_error_message(error.code, error.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()

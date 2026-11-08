"""エージェントの配線テスト。

TestModel は実際のLLMを呼ばない偽モデルで、登録されたツールを全部呼んでから
ダミーの回答を返す。APIキーも利用枠も使わずに、
「ツールが登録され、呼ばれ、結果がモデルに戻るか」を確かめられる。
"""

from pydantic_ai import DeferredToolRequests
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from pydantic_ai.models.test import TestModel

from app.agent import agent, approve_deferred_tools
from app import tools

BASE_TOOLS = {"search_glossary", "get_lecture_summary", "make_quiz", "save_learning_log"}


def _parts(result, kind):
    return [p for m in result.all_messages() for p in m.parts if isinstance(p, kind)]


def test_base_tools_are_registered_and_called():
    model = TestModel()
    with agent.override(model=model):
        result = agent.run_sync("LoRAとは？")

    offered = {t.name for t in model.last_model_request_parameters.function_tools}
    assert BASE_TOOLS <= offered  # 演習でツールを足しても壊れないよう「含む」で判定

    called = {p.tool_name for p in _parts(result, ToolCallPart)}
    assert BASE_TOOLS <= called
    # 承認が必要なツール（requires_approval=True）は、実行されずに承認待ちになるので戻り値が無い
    waiting = len(result.output.approvals) if isinstance(result.output, DeferredToolRequests) else 0
    assert len(_parts(result, ToolReturnPart)) + waiting == len(_parts(result, ToolCallPart))
    assert result.output


def test_glossary_result_reaches_the_model():
    model = TestModel(call_tools=["search_glossary"])
    with agent.override(model=model):
        result = agent.run_sync("RAGとは？")
    returns = _parts(result, ToolReturnPart)
    assert returns and returns[0].tool_name == "search_glossary"


class SaveLogTestModel(TestModel):
    def gen_tool_args(self, tool_def):
        if tool_def.name == "save_learning_log":
            return {"topic": "承認テスト", "content": "承認後に保存する"}
        return super().gen_tool_args(tool_def)


def _request_learning_log_save(log_path, monkeypatch):
    monkeypatch.setattr(tools, "LEARNING_LOG_PATH", log_path)
    model = SaveLogTestModel(call_tools=["save_learning_log"])
    with agent.override(model=model):
        result = agent.run_sync("学習記録を保存して")
    assert isinstance(result.output, DeferredToolRequests)
    assert len(result.output.approvals) == 1
    return result


def _resume_with_approval(result, deferred_results):
    with agent.override(model=TestModel(call_tools=[])):
        return agent.run_sync(
            message_history=result.all_messages(),
            deferred_tool_results=deferred_results,
        )


def test_learning_log_is_not_written_before_approval(tmp_path, monkeypatch):
    log_path = tmp_path / "agent_learning_log.md"

    _request_learning_log_save(log_path, monkeypatch)

    assert not log_path.exists()


def test_learning_log_is_appended_after_approval(tmp_path, monkeypatch, capsys):
    log_path = tmp_path / "agent_learning_log.md"
    log_path.write_text("# Existing log\n", encoding="utf-8")
    result = _request_learning_log_save(log_path, monkeypatch)
    monkeypatch.setattr("builtins.input", lambda _: "y")

    deferred_results = approve_deferred_tools(result.output)
    prompt = capsys.readouterr().out
    assert "承認テスト" in prompt
    assert "承認後に保存する" in prompt
    _resume_with_approval(result, deferred_results)

    saved = log_path.read_text(encoding="utf-8")
    assert saved.startswith("# Existing log\n")
    assert "### 学習テーマ\n承認テスト" in saved
    assert "### 学習内容\n承認後に保存する" in saved


def test_learning_log_is_not_written_when_approval_is_rejected(tmp_path, monkeypatch):
    log_path = tmp_path / "agent_learning_log.md"
    original = "# Existing log\n"
    log_path.write_text(original, encoding="utf-8")
    result = _request_learning_log_save(log_path, monkeypatch)
    monkeypatch.setattr("builtins.input", lambda _: "n")

    deferred_results = approve_deferred_tools(result.output)
    _resume_with_approval(result, deferred_results)

    assert log_path.read_text(encoding="utf-8") == original

"""ツールのテスト（LLMもAPIキーも使わない）。"""

import json

from app import tools


def test_glossary_ids_are_unique_and_related_links_exist():
    terms = tools.load_glossary()
    ids = [t["id"] for t in terms]
    assert len(ids) == len(set(ids))
    for t in terms:
        for r in t["related"]:
            assert r in ids, f"{t['id']} の related に存在しない id: {r}"


def test_search_finds_term_by_name_and_alias():
    for query in ["LoRA", "lora", "Low-Rank Adaptation"]:
        hits = json.loads(tools.search_glossary(query))
        assert hits[0]["term"] == "LoRA", query


def test_search_finds_course_specific_term():
    hits = json.loads(tools.search_glossary("lecNN-start"))
    assert hits[0]["term"] == "lecNN-start タグ"


def test_search_limits_number_of_results():
    hits = json.loads(tools.search_glossary("エージェント"))
    assert 1 <= len(hits) <= tools.MAX_RESULTS


def test_search_reports_not_found():
    assert "見つかりませんでした" in tools.search_glossary("存在しない用語xyz")


def test_make_quiz_uses_a_term_from_the_requested_lecture():
    terms = [t for t in tools.load_glossary() if t["lecture"] == 2]

    result = json.loads(tools.make_quiz(2))

    assert result["question"].startswith("「")
    assert result["question"].endswith("」の意味は？")
    selected_term = result["question"].removeprefix("「").removesuffix("」の意味は？")
    matching_terms = [t for t in terms if t["term"] == selected_term]
    assert matching_terms
    assert result["answer"] == matching_terms[0]["definition"]


def test_make_quiz_reports_when_lecture_has_no_terms():
    assert "第99回の用語が用語集にありません" in tools.make_quiz(99)


def test_save_learning_log_appends_markdown_and_preserves_existing_content(tmp_path, monkeypatch):
    log_path = tmp_path / "learning_log" / "agent_learning_log.md"
    log_path.parent.mkdir()
    log_path.write_text("# Existing log\n\n", encoding="utf-8")
    monkeypatch.setattr(tools, "LEARNING_LOG_PATH", log_path)

    result = tools.save_learning_log("Human-in-the-Loop", "承認後に記録を保存する。")

    saved = log_path.read_text(encoding="utf-8")
    assert result
    assert saved.startswith("# Existing log\n\n")
    assert "## " in saved
    assert "### 学習テーマ\nHuman-in-the-Loop" in saved
    assert "### 学習内容\n承認後に記録を保存する。" in saved


def test_save_learning_log_rejects_empty_values(tmp_path, monkeypatch):
    log_path = tmp_path / "agent_learning_log.md"
    monkeypatch.setattr(tools, "LEARNING_LOG_PATH", log_path)

    for topic, content in [("", "content"), ("  ", "content"), ("topic", ""), ("topic", " \t\n")]:
        assert "空の記録は保存しません" in tools.save_learning_log(topic, content)

    assert not log_path.exists()


def test_lecture_summary():
    assert "エージェント" in tools.get_lecture_summary(3)
    assert "まだありません" in tools.get_lecture_summary(99)


def test_default_question_of_lecture1_is_answerable_from_glossary():
    """第1回の既定の質問（app.ask）は、第3回のエージェントが用語集で正しく答えられること。"""
    from app.ask import DEFAULT_QUESTION

    assert "3レベル" in DEFAULT_QUESTION
    hits = json.loads(tools.search_glossary("3レベル"))
    assert hits[0]["term"] == "宿題の3レベル"

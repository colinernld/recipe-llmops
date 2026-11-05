"""Évaluation avec un faux juge : vérifie le branchement Ragas, pas l'avis de Gemini."""

import json

import pytest

pytest.importorskip("ragas")

from langchain_core.language_models.fake_chat_models import FakeListChatModel  # noqa: E402

from recette_core.data import format_rows, read_jsonl  # noqa: E402
from recette_core.evaluation import constraint_metrics, judge_metrics, text_metrics  # noqa: E402

ROWS = format_rows(read_jsonl("data/sample_recipes.jsonl"))


def test_perfect_copy_scores_high_but_has_no_novelty():
    rows = [{**r, "prediction": r["recipe"]} for r in ROWS]
    scores = {**constraint_metrics(rows, [r["recipe"] for r in ROWS]), **text_metrics(rows)}
    assert scores["all_constraints_rate"] == 1
    assert scores["novelty"] == 0
    assert scores["bleu"] > 0.9 and scores["rouge1_f"] > 0.9


def test_free_text_fails_constraints():
    rows = [{**r, "prediction": "Je ne sais pas."} for r in ROWS]
    scores = constraint_metrics(rows, [r["recipe"] for r in ROWS])
    assert scores["format_rate"] == 0 and scores["all_constraints_rate"] == 0


def test_judge_votes_are_averaged():
    verdict = json.dumps({"reason": "ok", "verdict": 1})
    judge = FakeListChatModel(responses=[verdict])
    rows = [{**r, "prediction": r["recipe"]} for r in ROWS[:2]]
    scores = judge_metrics(rows, judge)
    assert scores == {"judge_culinary_coherence": 1.0, "judge_food_safety": 1.0}

"""Évaluation : contraintes déterministes, métriques Ragas et jugement par LLM."""

from __future__ import annotations

from statistics import mean

from recette_core.contract import check_recipe, novelty


def constraint_metrics(rows: list[dict], corpus: list[str]) -> dict[str, float]:
    """Taux de respect des contraintes. Ce sont les métriques qui décident du déploiement."""
    reports = [
        check_recipe(row["prediction"], row["ingredients"], row["appliances"], row["max_minutes"])
        for row in rows
    ]
    rate = lambda key: mean(float(report[key]) for report in reports)  # noqa: E731
    scores = {
        "format_rate": rate("format_ok"),
        "ingredient_compliance": rate("ingredients_ok"),
        "appliance_compliance": rate("appliances_ok"),
        "time_compliance": rate("time_ok"),
        "all_constraints_rate": rate("all_ok"),
        "novelty": mean(novelty(row["prediction"], corpus) for row in rows),
    }
    # Le modèle peut réussir dans une langue et échouer dans l'autre.
    for language in sorted({row.get("language", "fr") for row in rows}):
        flags = [
            float(report["all_ok"])
            for row, report in zip(rows, reports)
            if row.get("language", "fr") == language
        ]
        scores[f"all_constraints_rate_{language}"] = mean(flags)
    return scores


def text_metrics(rows: list[dict]) -> dict[str, float]:
    """Proximité avec la recette de référence. Utile pour comparer, pas pour décider."""
    from ragas import SingleTurnSample
    from ragas.metrics import BleuScore, RougeScore

    samples = [
        SingleTurnSample(user_input=r["user_prompt"], response=r["prediction"], reference=r["recipe"])
        for r in rows
    ]
    bleu, rouge = BleuScore(), RougeScore(rouge_type="rouge1", mode="fmeasure")
    return {
        "bleu": mean(bleu.single_turn_score(sample) for sample in samples),
        "rouge1_f": mean(rouge.single_turn_score(sample) for sample in samples),
    }


JUDGE_ASPECTS = {
    "judge_culinary_coherence": (
        "La réponse, en français ou en anglais, est-elle une recette cohérente : les étapes "
        "s'enchaînent, utilisent les ingrédients annoncés et donnent un plat réalisable ?"
    ),
    "judge_food_safety": (
        "La recette évite-t-elle tout danger alimentaire, comme une viande mal cuite ou un "
        "mélange d'ingrédients non comestibles ?"
    ),
}


def judge_metrics(rows: list[dict], chat_model, strictness: int = 3) -> dict[str, float]:
    """LLM-as-a-judge : `strictness` votes par exemple, comme conseillé en cours."""
    from ragas import SingleTurnSample
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import AspectCritic

    llm = LangchainLLMWrapper(chat_model)
    scores: dict[str, float] = {}
    for name, definition in JUDGE_ASPECTS.items():
        metric = AspectCritic(name=name, definition=definition, llm=llm, strictness=strictness)
        values = [
            metric.single_turn_score(
                SingleTurnSample(user_input=row["user_prompt"], response=row["prediction"])
            )
            for row in rows
        ]
        scores[name] = mean(float(value) for value in values)
    return scores


def vertex_judge(model: str, project: str, location: str):
    from langchain_google_vertexai import ChatVertexAI

    return ChatVertexAI(model_name=model, project=project, location=location, temperature=0)

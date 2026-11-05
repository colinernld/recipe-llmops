"""Génération d'une recette avec garde-fous et traces Langfuse."""

from __future__ import annotations

import json
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from langfuse import get_client, observe

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "shared"))

from recette_core.contract import (  # noqa: E402
    FOOD_SAFETY_NOTES,
    build_user_prompt,
    check_recipe,
    format_prompt,
    needs_food_safety_note,
    to_markdown,
)
from src.constants import ENDPOINT_ID, PROJECT_ID, REGION  # noqa: E402

PARAMETERS = {"max_new_tokens": 450, "temperature": 0.8, "top_p": 0.9, "do_sample": True}
MAX_ATTEMPTS = 2


@dataclass
class ModelReply:
    text: str
    model: str
    version: str


class ModelBackend(Protocol):
    def predict(self, prompt: str, parameters: dict) -> ModelReply: ...


class VertexBackend:
    def __init__(self, endpoint_id: str):
        from google.cloud import aiplatform

        aiplatform.init(project=PROJECT_ID, location=REGION)
        self.endpoint = aiplatform.Endpoint(endpoint_id)

    def predict(self, prompt: str, parameters: dict) -> ModelReply:
        response = self.endpoint.predict(instances=[{"input": prompt}], parameters=parameters)
        return ModelReply(
            text=response.predictions[0],
            model=response.model_resource_name or "recette-frigo",
            version=str(response.model_version_id),
        )


class DemoBackend:
    """Recettes écrites à la main, pour tester l'interface sans endpoint."""

    def __init__(self):
        sample = ROOT / "data" / "sample_recipes.jsonl"
        self.rows = [json.loads(line) for line in sample.read_text(encoding="utf-8").splitlines()]

    def predict(self, prompt: str, parameters: dict) -> ModelReply:
        language = "en" if "Fridge:" in prompt else "fr"
        rows = [row for row in self.rows if row["language"] == language]
        return ModelReply(text=random.choice(rows)["recipe"], model="demo", version="0")


def get_backend() -> ModelBackend:
    if os.getenv("DEMO_MODE") == "1" or not ENDPOINT_ID:
        return DemoBackend()
    return VertexBackend(ENDPOINT_ID)


@dataclass
class RecipeResult:
    text: str
    trace_id: str | None
    report: dict
    attempts: int
    warnings: list[str] = field(default_factory=list)
    demo: bool = False


PROBLEM_TEXTS = {
    "fr": {
        "format": "La réponse du modèle n'a pas le format attendu.",
        "ingredients": "Ingrédients en plus : {}.",
        "appliances": "Appareils en plus : {}.",
        "time": "Temps annoncé : {} min pour {} max.",
    },
    "en": {
        "format": "The model's answer does not have the expected format.",
        "ingredients": "Extra ingredients: {}.",
        "appliances": "Extra appliances: {}.",
        "time": "Stated time: {} min for a {} min limit.",
    },
}


def describe_problems(report: dict, max_minutes: int, language: str = "fr") -> list[str]:
    texts = PROBLEM_TEXTS[language]
    if not report["format_ok"]:
        return [texts["format"]]
    problems = []
    if report["extra_ingredients"]:
        problems.append(texts["ingredients"].format(", ".join(report["extra_ingredients"])))
    if report["extra_appliances"]:
        problems.append(texts["appliances"].format(", ".join(report["extra_appliances"])))
    if not report["time_ok"]:
        problems.append(texts["time"].format(report["total_minutes"], max_minutes))
    return problems


@observe(name="Demande de recette")
def generate_recipe(
    backend: ModelBackend,
    ingredients: list[str],
    appliances: list[str],
    max_minutes: int,
    session_id: str,
    language: str = "fr",
    cuisine: str = "surprise",
) -> RecipeResult:
    langfuse = get_client()
    user_prompt = build_user_prompt(ingredients, appliances, max_minutes, language, cuisine)
    prompt = format_prompt(user_prompt, language)
    langfuse.update_current_trace(
        session_id=session_id, input=user_prompt, tags=[language, cuisine]
    )

    text, report = "", {}
    attempts = 0
    for attempts in range(1, MAX_ATTEMPTS + 1):
        with langfuse.start_as_current_observation(
            as_type="generation", name="Génération Phi-3"
        ) as generation:
            reply = backend.predict(prompt, PARAMETERS)
            generation.update(
                input=prompt,
                output=reply.text,
                model=reply.model,
                version=reply.version,
                metadata={"attempt": attempts, **PARAMETERS},
            )
        text = reply.text
        report = check_recipe(text, ingredients, appliances, max_minutes)
        if report["all_ok"]:
            break

    warnings = describe_problems(report, max_minutes, language)
    langfuse.update_current_trace(output=text, metadata={"attempts": attempts})
    for name in ("format_ok", "ingredients_ok", "appliances_ok", "time_ok", "all_ok"):
        langfuse.score_current_trace(name=name, value=float(report[name]))
    display = to_markdown(text, language)
    if needs_food_safety_note(text):
        display = f"{display}\n\n{FOOD_SAFETY_NOTES[language]}"
    trace_id = langfuse.get_current_trace_id()
    return RecipeResult(display, trace_id, report, attempts, warnings, reply.model == "demo")


def send_feedback(trace_id: str | None, name: str, value: float, comment: str | None = None) -> None:
    """Rattache un retour utilisateur à la recette exacte qui l'a déclenché."""
    if trace_id:
        # Un identifiant stable : changer sa note remplace la précédente.
        get_client().create_score(
            trace_id=trace_id,
            name=name,
            value=value,
            comment=comment,
            score_id=f"{trace_id}-{name}",
        )

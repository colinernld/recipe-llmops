"""Rejoue dans Langfuse les recettes générées par le modèle fine-tuné pendant le pipeline.

Plan B quand l'endpoint ne peut pas être déployé (pas de GPU disponible) : chaque
prédiction de data/eval/finetuned.jsonl passe par generate_recipe, avec les mêmes
contrôles, scores et traces qu'en production. Les traces portent le tag « replay ».

Limites : la latence est fausse (aucun appel au GPU) et il n'y a qu'une tentative
par frigo, car le pipeline n'a produit qu'une recette par frigo.

    uv run python scripts/replay_predictions.py
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import recipe_service as service  # noqa: E402
from src.recipe_service import ModelReply, generate_recipe  # noqa: E402

MODEL_NAME = "recette-frigo-phi-3"


class ReplayBackend:
    """Renvoie la recette que le modèle fine-tuné a réellement produite pour ce frigo."""

    tags = ["replay"]

    def __init__(self, prediction: str):
        self.prediction = prediction

    def predict(self, prompt: str, parameters: dict) -> ModelReply:
        return ModelReply(text=self.prediction, model=MODEL_NAME, version="1")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(ROOT / "data" / "eval" / "finetuned.jsonl"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines()]
    rows = rows[: args.limit]
    # Une seule prédiction par frigo : une relance renverrait la même recette.
    service.MAX_ATTEMPTS = 1
    session_id = f"replay-{datetime.now():%Y%m%d-%H%M}"

    passed = 0
    for row in rows:
        result = generate_recipe(
            ReplayBackend(row["prediction"]),
            row["ingredients"],
            row["appliances"],
            row["max_minutes"],
            session_id,
            row["language"],
            row.get("cuisine") or "surprise",
        )
        passed += result.report["all_ok"]

    service.get_client().flush()
    print(f"{len(rows)} recettes rejouées, {passed} respectent les 3 contraintes.")
    print(f"Session Langfuse : {session_id}")


if __name__ == "__main__":
    main()

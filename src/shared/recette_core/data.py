"""Nettoyage indépendant du modèle, puis formatage propre à Phi-3."""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path

from recette_core.contract import check_recipe, format_training_text, normalize, parse_recipe


def local_path(uri: str) -> str:
    """Sur Vertex AI, un bucket est monté sous /gcs/ dans le conteneur."""
    return uri.replace("gs://", "/gcs/", 1) if uri.startswith("gs://") else uri


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_jsonl(rows: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_rows(rows: list[dict]) -> tuple[list[dict], dict[str, int]]:
    """Retire les HTML, les exemples hors contrat et les doublons de titre."""
    kept, stats, titles = [], Counter(raw=len(rows)), set()
    for row in rows:
        recipe_text = re.sub(r"<[^>]+>", "", str(row.get("recipe", ""))).strip()
        try:
            report = check_recipe(
                recipe_text, row["ingredients"], row["appliances"], row["max_minutes"]
            )
        except (KeyError, TypeError):
            stats["rejected_schema"] += 1
            continue
        if not report["format_ok"]:
            stats["rejected_format"] += 1
        elif not report["all_ok"]:
            stats["rejected_constraints"] += 1
        else:
            title = normalize(parse_recipe(recipe_text).title)
            if title in titles:
                stats["rejected_duplicate"] += 1
                continue
            titles.add(title)
            kept.append({**row, "recipe": recipe_text})
    stats["kept"] = len(kept)
    return kept, dict(stats)


def split_rows(
    rows: list[dict], seed: int, val_ratio: float, test_ratio: float
) -> tuple[list[dict], list[dict], list[dict]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * val_ratio))
    n_test = max(1, round(len(shuffled) * test_ratio))
    return shuffled[n_val + n_test :], shuffled[:n_val], shuffled[n_val : n_val + n_test]


def format_rows(rows: list[dict]) -> list[dict]:
    formatted = []
    for row in rows:
        prompt, completion = format_training_text(
            row["user_prompt"], row["recipe"], row.get("language", "fr")
        )
        formatted.append({**row, "prompt": prompt, "completion": completion})
    return formatted

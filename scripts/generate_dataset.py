"""Crée un dataset de recettes du monde entier, en français et en anglais, avec Gemini.

Le script tire au hasard un frigo, des appareils, un temps, une cuisine et un parti pris
créatif, demande une recette structurée, puis la contrôle avec les règles du contrat.
Il reprend là où il s'est arrêté si le fichier de sortie existe déjà.

Exemple :
    uv run --env-file .env python scripts/generate_dataset.py --n 3000
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "shared"))

from recette_core.contract import (  # noqa: E402
    APPLIANCE_NAMES_EN,
    PANTRY_STAPLES,
    Recipe,
    build_user_prompt,
    check_recipe,
    render_recipe,
)

# Chaque entrée donne le nom français, puis le nom anglais.
CATALOG = {
    "protein": [
        ("œufs", "eggs"), ("tofu", "tofu"), ("restes de poulet rôti", "leftover roast chicken"),
        ("blanc de poulet", "chicken breast"), ("thon en boîte", "canned tuna"),
        ("saucisses", "sausages"), ("lardons", "bacon"), ("jambon", "ham"),
        ("pavé de saumon", "salmon fillet"), ("crevettes", "shrimp"),
        ("bœuf haché", "ground beef"), ("côtes de porc", "pork chops"),
        ("agneau haché", "ground lamb"), ("pois chiches", "chickpeas"),
        ("haricots noirs", "black beans"), ("lentilles corail", "red lentils"),
        ("houmous", "hummus"),
    ],
    "starch": [
        ("riz cuit", "cooked rice"), ("pâtes", "pasta"), ("pommes de terre", "potatoes"),
        ("patate douce", "sweet potato"), ("pain rassis", "stale bread"),
        ("tortillas", "tortillas"), ("nouilles de riz", "rice noodles"), ("quinoa", "quinoa"),
        ("flocons d'avoine", "rolled oats"), ("pâte feuilletée", "puff pastry"),
        ("farine", "flour"), ("maïs", "corn"), ("semoule", "couscous"),
    ],
    "vegetable": [
        ("épinards", "spinach"), ("carottes", "carrots"), ("courgettes", "zucchini"),
        ("kimchi", "kimchi"), ("tomates", "tomatoes"), ("oignons", "onions"),
        ("champignons", "mushrooms"), ("poivron", "bell pepper"), ("aubergine", "eggplant"),
        ("chou", "cabbage"), ("concombre", "cucumber"), ("petits pois", "frozen peas"),
        ("brocoli", "broccoli"), ("chou-fleur", "cauliflower"), ("poireau", "leek"),
        ("betterave", "beetroot"), ("potiron", "pumpkin"), ("avocat", "avocado"),
        ("tomates en boîte", "canned tomatoes"), ("olives", "olives"), ("cornichons", "pickles"),
    ],
    "dairy": [
        ("feta", "feta"), ("mozzarella", "mozzarella"), ("yaourt nature", "plain yogurt"),
        ("cheddar", "cheddar"), ("parmesan", "parmesan"), ("crème fraîche", "heavy cream"),
        ("beurre", "butter"), ("lait", "milk"), ("fromage frais", "cream cheese"),
        ("chèvre frais", "goat cheese"), ("lait de coco", "coconut milk"),
    ],
    "flavor": [
        ("ail", "garlic"), ("gingembre frais", "fresh ginger"), ("citron", "lemon"),
        ("sauce soja", "soy sauce"), ("pâte de miso", "miso paste"), ("harissa", "harissa"),
        ("moutarde", "mustard"), ("tahini", "tahini"), ("beurre de cacahuète", "peanut butter"),
        ("miel", "honey"), ("basilic frais", "fresh basil"),
        ("coriandre fraîche", "fresh cilantro"), ("menthe", "mint"),
        ("piment en flocons", "chili flakes"), ("câpres", "capers"),
        ("paprika fumé", "smoked paprika"), ("cumin", "cumin"), ("curry en poudre", "curry powder"),
    ],
    "wildcard": [
        ("pommes", "apples"), ("bananes", "bananas"), ("ananas", "pineapple"),
        ("mangue", "mango"), ("oranges", "oranges"), ("cacahuètes", "peanuts"),
        ("noix", "walnuts"), ("dattes", "dates"), ("raisins secs", "raisins"),
    ],
}

CUISINES = [
    ("coréenne", "Korean"), ("éthiopienne", "Ethiopian"), ("péruvienne", "Peruvian"),
    ("libanaise", "Lebanese"), ("vietnamienne", "Vietnamese"), ("nigériane", "Nigerian"),
    ("géorgienne", "Georgian"), ("mexicaine", "Mexican"), ("japonaise", "Japanese"),
    ("indienne", "Indian"), ("thaïlandaise", "Thai"), ("marocaine", "Moroccan"),
    ("turque", "Turkish"), ("grecque", "Greek"), ("italienne", "Italian"),
    ("française", "French"), ("espagnole", "Spanish"), ("brésilienne", "Brazilian"),
    ("indonésienne", "Indonesian"), ("philippine", "Filipino"), ("iranienne", "Iranian"),
    ("sénégalaise", "Senegalese"), ("ghanéenne", "Ghanaian"), ("sichuanaise", "Sichuan Chinese"),
    ("argentine", "Argentinian"), ("polonaise", "Polish"), ("suédoise", "Swedish"),
    ("jamaïcaine", "Jamaican"),
]

TWISTS = [
    "a fusion of two cuisines",
    "a technique from one cuisine applied to an ingredient from another",
    "an unexpected reinvention of a well-known dish",
    "a surprising contrast of textures",
    "a street-food style reinvention",
]

# Noms canoniques, traduits au moment du tirage.
APPLIANCE_SETS = [
    ["plaques"], ["four"], ["micro-ondes"], ["airfryer"], ["plaques", "micro-ondes"],
    ["four", "plaques"], ["plaques", "mixeur"], ["plaques", "airfryer"],
    ["cuiseur-riz", "plaques"], ["four", "plaques", "mixeur"], ["mijoteuse"],
    ["plaques", "autocuiseur"],
]
MINUTES = [10, 15, 20, 30, 45, 60, 90]
LANGUAGE_NAMES = {"fr": "French", "en": "English"}

RECIPE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "title": {"type": "STRING"},
        "total_minutes": {"type": "INTEGER"},
        "appliances": {"type": "ARRAY", "items": {"type": "STRING"}},
        "ingredients": {"type": "ARRAY", "items": {"type": "STRING"}},
        "steps": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["title", "total_minutes", "appliances", "ingredients", "steps"],
}

Generate = Callable[[str], dict]


def _pick(pair: tuple[str, str], language: str) -> str:
    return pair[0] if language == "fr" else pair[1]


def sample_fridge(rng: random.Random, language: str) -> list[str]:
    pairs = [rng.choice(CATALOG["protein"]), rng.choice(CATALOG["starch"])]
    pairs += rng.sample(CATALOG["vegetable"], rng.randint(2, 3))
    if rng.random() < 0.6:
        pairs.append(rng.choice(CATALOG["dairy"]))
    pairs += rng.sample(CATALOG["flavor"], rng.randint(1, 2))
    if rng.random() < 0.2:
        pairs.append(rng.choice(CATALOG["wildcard"]))
    rng.shuffle(pairs)
    return [_pick(pair, language) for pair in pairs]


def sample_scenario(rng: random.Random) -> dict:
    language = rng.choice(["fr", "en"])
    appliances = rng.choice(APPLIANCE_SETS)
    if language == "en":
        appliances = [APPLIANCE_NAMES_EN[name] for name in appliances]
    surprise = rng.random() < 0.1
    return {
        "language": language,
        "ingredients": sample_fridge(rng, language),
        "appliances": list(appliances),
        "max_minutes": rng.choice(MINUTES),
        "cuisine": "surprise" if surprise else _pick(rng.choice(CUISINES), language),
        "twist": rng.choice(TWISTS),
    }


def build_generation_prompt(scenario: dict) -> str:
    language = scenario["language"]
    example = "poulet : 200 g" if language == "fr" else "chicken: 200 g"
    separator = '"nom : quantité"' if language == "fr" else '"name: quantity"'
    cuisine = scenario["cuisine"]
    inspiration = (
        "from any cuisine in the world, ideally a less common one"
        if cuisine == "surprise"
        else f"inspired by this cuisine: {cuisine}"
    )
    return f"""You write training data for a cooking assistant.
Invent an original, realistic and safe recipe {inspiration}.
Creative angle: {scenario["twist"]}. Give it a creative title.
Write everything in {LANGUAGE_NAMES[language]}.

Strict rules:
- Use only ingredients from this list: {", ".join(scenario["ingredients"])}.
- You may also use: {", ".join(PANTRY_STAPLES[language])}.
- You do not have to use every ingredient, only those that work together.
- Use only these appliances: {", ".join(scenario["appliances"])}.
- "appliances" lists exactly the appliances used, with these exact names.
- "total_minutes" covers preparation and cooking, and must not exceed {scenario["max_minutes"]}.
- Each ingredient follows the format {separator}, for example "{example}".
- Write 3 to 7 short steps, without numbers.
- Meat, fish and eggs must be cooked safely."""


def to_example(scenario: dict, payload: dict) -> dict | None:
    """Transforme la réponse du modèle en exemple, ou None si elle est invalide."""
    try:
        recipe = Recipe(
            title=str(payload["title"]).strip(),
            total_minutes=int(payload["total_minutes"]),
            appliances=list(payload["appliances"]),
            ingredients=list(payload["ingredients"]),
            steps=list(payload["steps"]),
        )
    except (KeyError, TypeError, ValueError):
        return None
    language = scenario["language"]
    text = render_recipe(recipe, language)
    report = check_recipe(
        text, scenario["ingredients"], scenario["appliances"], scenario["max_minutes"]
    )
    if not report["all_ok"]:
        return None
    return {
        **scenario,
        "user_prompt": build_user_prompt(
            scenario["ingredients"],
            scenario["appliances"],
            scenario["max_minutes"],
            language,
            scenario["cuisine"],
        ),
        "recipe": text,
    }


def gemini_generator(model: str, project: str, location: str) -> Generate:
    from google import genai
    from google.genai import types

    client = genai.Client(vertexai=True, project=project, location=location)
    config = types.GenerateContentConfig(
        temperature=1.0,
        response_mime_type="application/json",
        response_schema=RECIPE_SCHEMA,
        # Réflexion réduite : environ 3,7 fois moins cher.
        thinking_config=types.ThinkingConfig(thinking_level="LOW"),
    )

    def generate(prompt: str) -> dict:
        response = client.models.generate_content(model=model, contents=prompt, config=config)
        return json.loads(response.text)

    return generate


def build_dataset(
    generate: Generate, n: int, seed: int, output: Path, workers: int = 8, max_attempts: int = 4
) -> dict:
    """Écrit `n` exemples valides. S'arrête après `max_attempts` × n essais au plus."""
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = output.read_text(encoding="utf-8").splitlines() if output.exists() else []
    kept, rejected, errors = len(existing), 0, 0
    # La graine dépend de l'avancement pour ne pas rejouer les mêmes situations à la reprise.
    rng = random.Random(seed + kept)
    budget = max(0, n - kept) * max_attempts

    def work(scenario: dict) -> dict | None:
        return to_example(scenario, generate(build_generation_prompt(scenario)))

    with output.open("a", encoding="utf-8") as sink, ThreadPoolExecutor(workers) as pool:
        while kept < n and budget > 0:
            batch = [sample_scenario(rng) for _ in range(min(workers * 4, budget))]
            budget -= len(batch)
            futures = [pool.submit(work, scenario) for scenario in batch]
            for future in as_completed(futures):
                try:
                    example = future.result()
                except Exception as error:  # Quota ou réseau : on compte et on continue.
                    errors += 1
                    print(f"Erreur de génération : {error}", file=sys.stderr)
                    continue
                if example is None:
                    rejected += 1
                elif kept < n:
                    sink.write(json.dumps(example, ensure_ascii=False) + "\n")
                    sink.flush()
                    kept += 1
            print(f"{kept}/{n} gardés, {rejected} rejetés, {errors} erreurs")
    return {"kept": kept, "rejected": rejected, "errors": errors}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/recipes.jsonl"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    generate = gemini_generator(
        model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
        project=os.environ["PROJECT_ID"],
        location=os.getenv("GEMINI_LOCATION", "global"),
    )
    print(build_dataset(generate, args.n, args.seed, args.output, args.workers))


if __name__ == "__main__":
    main()

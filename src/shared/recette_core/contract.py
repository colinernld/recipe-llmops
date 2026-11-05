"""Contrat partagé : format des recettes et règles vérifiables.

Ce module n'utilise que la bibliothèque standard. Il sert à la fois à la création
du dataset, à l'évaluation dans le pipeline et aux garde-fous de l'application,
pour qu'une même règle ne puisse pas diverger d'une étape à l'autre.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

SYSTEM_PROMPTS = {
    "fr": (
        "Tu es un chef cuisinier créatif et prudent, inspiré par les cuisines du monde. "
        "Tu inventes une recette originale en français. Tu utilises uniquement les "
        "ingrédients donnés et les basiques du placard, uniquement les appareils donnés, "
        "et tu respectes le temps maximum. Tu refuses tout ce qui ne concerne pas la cuisine."
    ),
    "en": (
        "You are a creative and careful chef, inspired by cuisines from around the world. "
        "You invent an original recipe in English. You only use the given ingredients and "
        "pantry staples, only the given appliances, and you respect the maximum time. "
        "You refuse anything unrelated to cooking."
    ),
}

# Basiques supposés présents dans toutes les cuisines.
PANTRY_STAPLES = {
    "fr": ("eau", "sel", "poivre", "huile", "huile d'olive", "vinaigre", "sucre"),
    "en": ("water", "salt", "pepper", "oil", "olive oil", "vinegar", "sugar"),
}
_ALL_STAPLES = (*PANTRY_STAPLES["fr"], *PANTRY_STAPLES["en"])

# Synonymes vers un nom d'appareil unique. Les clés sont déjà normalisées.
APPLIANCE_ALIASES = {
    "four": "four",
    "plaque": "plaques",
    "plaques": "plaques",
    "plaque de cuisson": "plaques",
    "plaques de cuisson": "plaques",
    "induction": "plaques",
    "gaziniere": "plaques",
    "feu": "plaques",
    "micro onde": "micro-ondes",
    "micro ondes": "micro-ondes",
    "micro-onde": "micro-ondes",
    "micro-ondes": "micro-ondes",
    "microonde": "micro-ondes",
    "airfryer": "airfryer",
    "air fryer": "airfryer",
    "friteuse a air": "airfryer",
    "mixeur": "mixeur",
    "blender": "mixeur",
    "robot": "robot",
    "robot de cuisine": "robot",
    "mijoteuse": "mijoteuse",
    "cocotte minute": "autocuiseur",
    "cocotte-minute": "autocuiseur",
    "autocuiseur": "autocuiseur",
    "grille pain": "grille-pain",
    "grille-pain": "grille-pain",
    "cuiseur a riz": "cuiseur-riz",
    "cuiseur-riz": "cuiseur-riz",
    "poele": "plaques",
    "casserole": "plaques",
    "stove": "plaques",
    "stovetop": "plaques",
    "hob": "plaques",
    "cooktop": "plaques",
    "pan": "plaques",
    "pot": "plaques",
    "oven": "four",
    "microwave": "micro-ondes",
    "blender": "mixeur",
    "food processor": "robot",
    "slow cooker": "mijoteuse",
    "pressure cooker": "autocuiseur",
    "toaster": "grille-pain",
    "rice cooker": "cuiseur-riz",
}
APPLIANCE_NAMES_EN = {
    "plaques": "stove",
    "four": "oven",
    "micro-ondes": "microwave",
    "airfryer": "air fryer",
    "mixeur": "blender",
    "robot": "food processor",
    "mijoteuse": "slow cooker",
    "autocuiseur": "pressure cooker",
    "grille-pain": "toaster",
    "cuiseur-riz": "rice cooker",
}
KNOWN_APPLIANCES = tuple(sorted(set(APPLIANCE_ALIASES.values())))

# Mots trop généraux pour prouver qu'il s'agit du même ingrédient.
_STOPWORDS = {
    "de", "du", "des", "la", "le", "les", "un", "une", "d", "l", "a", "au", "aux", "et", "en",
    "of", "the", "an", "and", "fresh", "frais", "fraiche", "canned", "boite", "frozen",
    "surgele", "surgelee", "cooked", "cuit", "leftover", "stale", "sauce", "paste", "powder",
    "blanc", "filet", "fillet",
}

MIN_MINUTES, MAX_MINUTES = 5, 240
MAX_INPUT_CHARS = 600

_INJECTION = re.compile(
    r"(ignore|oublie|contourne|ignorer|oublier|forget|bypass|disregard).{0,40}"
    r"(instruction|consigne|regle|rule|prompt)"
    r"|prompt systeme|system prompt|<\|",
    re.IGNORECASE,
)
_NON_FOOD = re.compile(
    r"\b(javel|lessive|essence|detergent|white spirit|medicament|antigel|mort aux rats"
    r"|bleach|gasoline|petrol|medicine|antifreeze|rat poison)\b"
)
# Pour ces aliments, une cuisson insuffisante présente un risque connu.
_RISKY_FOOD = re.compile(
    r"\b(poulet|dinde|volaille|porc|saucisses?|steak hache|hache|agneau"
    r"|chicken|turkey|pork|sausages?|ground beef|ground lamb)\b"
)
FOOD_SAFETY_NOTES = {
    "fr": "Sécurité : fais bien cuire la viande à cœur avant de servir.",
    "en": "Safety: cook meat all the way through before serving.",
}
_LABELS = {
    "fr": ("Titre", "Temps total", "Appareils", "Ingrédients", "Étapes"),
    "en": ("Title", "Total time", "Appliances", "Ingredients", "Steps"),
}
_INPUT_MESSAGES = {
    "fr": {
        "long": f"La saisie dépasse {MAX_INPUT_CHARS} caractères.",
        "injection": "Je ne peux répondre qu'à une demande de recette.",
        "non_food": "Un des produits indiqués n'est pas comestible.",
        "no_ingredient": "Indique au moins un ingrédient.",
        "no_appliance": "Indique au moins un appareil.",
        "time": f"Indique un temps entre {MIN_MINUTES} et {MAX_MINUTES} minutes.",
    },
    "en": {
        "long": f"Your input is longer than {MAX_INPUT_CHARS} characters.",
        "injection": "I can only help with recipe requests.",
        "non_food": "One of the listed products is not edible.",
        "no_ingredient": "Please list at least one ingredient.",
        "no_appliance": "Please list at least one appliance.",
        "time": f"Please give a time between {MIN_MINUTES} and {MAX_MINUTES} minutes.",
    },
}


@dataclass
class Recipe:
    title: str
    total_minutes: int
    appliances: list[str]
    ingredients: list[str]
    steps: list[str] = field(default_factory=list)


def normalize(text: str) -> str:
    """Minuscules, sans accents, sans ponctuation inutile."""
    text = text.lower().replace("œ", "oe").replace("æ", "ae").replace("’", "'")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", normalize(text))
    singular = {w[:-1] if len(w) > 3 and w[-1] in "sx" else w for w in words}
    return singular - _STOPWORDS


def canonical_appliance(name: str) -> str:
    key = normalize(name)
    return APPLIANCE_ALIASES.get(key, key)


def split_list(text: str) -> list[str]:
    """Découpe une saisie libre : virgules, points-virgules, retours ou « et »."""
    parts = re.split(r"[,;\n]|\s(?:et|and)\s", text)
    return [p.strip(" .-•\t") for p in parts if p.strip(" .-•\t")]


def parse_minutes(text: str) -> int | None:
    """Comprend « 45 », « 45 min », « 1h », « 1h30 » ou « 2 heures »."""
    clean = normalize(text)
    match = re.search(r"(\d+)\s*h(?:eures?|ours?|rs?)?\s*(\d+)?", clean)
    if match:
        return int(match.group(1)) * 60 + int(match.group(2) or 0)
    match = re.search(r"\d+", clean)
    return int(match.group()) if match else None


def check_input(
    ingredients: str, appliances: str, minutes: int | None, language: str = "fr"
) -> list[str]:
    """Garde-fou d'entrée : renvoie les problèmes à montrer à l'utilisateur."""
    messages = _INPUT_MESSAGES[language]
    problems = []
    joined = normalize(f"{ingredients} {appliances}")
    if len(joined) > MAX_INPUT_CHARS:
        problems.append(messages["long"])
    if _INJECTION.search(joined):
        problems.append(messages["injection"])
    if _NON_FOOD.search(joined):
        problems.append(messages["non_food"])
    if not split_list(ingredients):
        problems.append(messages["no_ingredient"])
    if not split_list(appliances):
        problems.append(messages["no_appliance"])
    if minutes is None or not MIN_MINUTES <= minutes <= MAX_MINUTES:
        problems.append(messages["time"])
    return problems


def build_user_prompt(
    ingredients: list[str],
    appliances: list[str],
    minutes: int,
    language: str = "fr",
    cuisine: str = "surprise",
) -> str:
    if language == "en":
        return (
            f"Fridge: {', '.join(ingredients)}\n"
            f"Appliances: {', '.join(appliances)}\n"
            f"Maximum time: {minutes} min\n"
            f"Cuisine: {cuisine}\n"
            "Invent an original recipe."
        )
    return (
        f"Frigo : {', '.join(ingredients)}\n"
        f"Appareils : {', '.join(appliances)}\n"
        f"Temps maximum : {minutes} min\n"
        f"Cuisine : {cuisine}\n"
        "Invente une recette originale."
    )


def format_prompt(user_prompt: str, language: str = "fr") -> str:
    """Modèle de conversation Phi-3, sans la réponse."""
    system = SYSTEM_PROMPTS[language]
    return f"<|system|>\n{system}<|end|>\n<|user|>\n{user_prompt}<|end|>\n<|assistant|>\n"


def format_training_text(
    user_prompt: str, recipe_text: str, language: str = "fr"
) -> tuple[str, str]:
    """Renvoie (prompt, réponse attendue) au format d'entraînement Phi-3."""
    return format_prompt(user_prompt, language), f"{recipe_text}<|end|>\n<|endoftext|>"


def render_recipe(recipe: Recipe, language: str = "fr") -> str:
    title, time, appliances, ingredients, steps = _LABELS[language]
    separator = " :" if language == "fr" else ":"
    ingredient_lines = "\n".join(f"- {line}" for line in recipe.ingredients)
    step_lines = "\n".join(f"{i}. {step}" for i, step in enumerate(recipe.steps, 1))
    return (
        f"{title}{separator} {recipe.title}\n"
        f"{time}{separator} {recipe.total_minutes} min\n"
        f"{appliances}{separator} {', '.join(recipe.appliances)}\n"
        f"{ingredients}{separator}\n{ingredient_lines}\n"
        f"{steps}{separator}\n{step_lines}"
    )


def parse_recipe(text: str) -> Recipe | None:
    """Relit une recette générée, en français ou en anglais. Renvoie None si hors format."""
    pattern = re.compile(
        r"(?:Titre|Title)\s*:\s*(?P<title>.+?)\s*\n+"
        r"(?:Temps total|Total time)\s*:\s*(?P<minutes>\d+)\s*min\w*\s*\n+"
        r"(?:Appareils|Appliances)\s*:\s*(?P<appliances>.+?)\s*\n+"
        r"(?:Ingr[ée]dients|Ingredients)\s*:\s*\n(?P<ingredients>.+?)\n+"
        r"(?:[ÉE]tapes|Steps)\s*:\s*\n(?P<steps>.+)",
        re.DOTALL | re.IGNORECASE,
    )
    match = pattern.search(text)
    if not match:
        return None
    ingredients = re.findall(r"^\s*[-•]\s*(.+?)\s*$", match["ingredients"], re.MULTILINE)
    steps = re.findall(r"^\s*\d+[.)]\s*(.+?)\s*$", match["steps"], re.MULTILINE)
    if not ingredients or not steps:
        return None
    return Recipe(
        title=match["title"],
        total_minutes=int(match["minutes"]),
        appliances=[a.strip() for a in match["appliances"].split(",") if a.strip()],
        ingredients=ingredients,
        steps=steps,
    )


def to_markdown(text: str, language: str = "fr") -> str:
    """Mise en page pour l'interface. Laisse le texte tel quel s'il est hors format."""
    recipe = parse_recipe(text)
    if recipe is None:
        return text
    _, time, appliances, ingredients, steps = _LABELS[language]
    separator = " :" if language == "fr" else ":"
    ingredient_lines = "\n".join(f"- {line}" for line in recipe.ingredients)
    step_lines = "\n".join(f"{i}. {step}" for i, step in enumerate(recipe.steps, 1))
    return (
        f"## {recipe.title}\n\n"
        f"**{time}{separator}** {recipe.total_minutes} min\n\n"
        f"**{appliances}{separator}** {', '.join(recipe.appliances)}\n\n"
        f"### {ingredients}\n\n{ingredient_lines}\n\n"
        f"### {steps}\n\n{step_lines}"
    )


def ingredient_name(line: str) -> str:
    """« poulet : 200 g » devient « poulet »."""
    return line.split(":", 1)[0].strip()


def _is_available(name: str, available: list[str]) -> bool:
    wanted = _tokens(name)
    if not wanted:
        return True
    # Un basique doit correspondre en entier : « bell pepper » n'est pas du poivre.
    if any(wanted <= _tokens(staple) for staple in _ALL_STAPLES):
        return True
    return any(wanted & _tokens(item) for item in available)


def forbidden_ingredients(recipe: Recipe, fridge: list[str]) -> list[str]:
    names = [ingredient_name(line) for line in recipe.ingredients]
    return [name for name in names if not _is_available(name, fridge)]


def forbidden_appliances(recipe: Recipe, appliances: list[str]) -> list[str]:
    allowed = {canonical_appliance(a) for a in appliances}
    return [a for a in recipe.appliances if canonical_appliance(a) not in allowed]


def check_recipe(
    text: str, fridge: list[str], appliances: list[str], max_minutes: int
) -> dict[str, object]:
    """Contrôle déterministe d'une recette par rapport à la demande."""
    recipe = parse_recipe(text)
    if recipe is None:
        return {
            "format_ok": False,
            "ingredients_ok": False,
            "appliances_ok": False,
            "time_ok": False,
            "all_ok": False,
            "extra_ingredients": [],
            "extra_appliances": [],
            "total_minutes": None,
        }
    extra_ingredients = forbidden_ingredients(recipe, fridge)
    extra_appliances = forbidden_appliances(recipe, appliances)
    time_ok = 0 < recipe.total_minutes <= max_minutes
    return {
        "format_ok": True,
        "ingredients_ok": not extra_ingredients,
        "appliances_ok": not extra_appliances,
        "time_ok": time_ok,
        "all_ok": time_ok and not extra_ingredients and not extra_appliances,
        "extra_ingredients": extra_ingredients,
        "extra_appliances": extra_appliances,
        "total_minutes": recipe.total_minutes,
    }


def needs_food_safety_note(text: str) -> bool:
    return bool(_RISKY_FOOD.search(normalize(text)))


def similarity(left: str, right: str) -> float:
    """Similarité de Jaccard sur les mots : 1 signifie un contenu identique."""
    a, b = _tokens(left), _tokens(right)
    return len(a & b) / len(a | b) if a | b else 0.0


def novelty(text: str, corpus: list[str]) -> float:
    """1 moins la similarité avec la recette la plus proche du corpus."""
    return 1.0 - max((similarity(text, other) for other in corpus), default=0.0)

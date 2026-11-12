"""Interface Chainlit : frigo, appareils, temps, cuisine, recette, puis feedback.

Seuls les ingrédients se tapent au clavier. Tout le reste se choisit avec des boutons.
"""

import chainlit as cl

from src.recipe_service import generate_recipe, get_backend, send_feedback
from src.shared.recette_core.contract import APPLIANCE_NAMES_EN, check_input, split_list

TITLE = "".join(
    '<span class="gap"></span>' if letter == " " else f"<span>{letter}</span>"
    for letter in "Recette Frigo"
)
MAGNET_TITLE = f'<div class="magnets" aria-label="Recette Frigo">{TITLE}</div>'

# Appareils présents dans le dataset, avec leur nom canonique en clé.
APPLIANCES = {
    "plaques": ("🔥 Plaques", "🔥 Stove"),
    "four": ("♨️ Four", "♨️ Oven"),
    "micro-ondes": ("📡 Micro-ondes", "📡 Microwave"),
    "airfryer": ("🍟 Airfryer", "🍟 Air fryer"),
    "mixeur": ("🌀 Mixeur", "🌀 Blender"),
    "cuiseur-riz": ("🍚 Cuiseur à riz", "🍚 Rice cooker"),
    "mijoteuse": ("🍲 Mijoteuse", "🍲 Slow cooker"),
    "autocuiseur": ("⏲️ Autocuiseur", "⏲️ Pressure cooker"),
}
# Durées proposées, en minutes. « + d'1 h » vaut 90, la limite haute du dataset.
TIMES = [
    (15, "15 min", "15 min"),
    (30, "30 min", "30 min"),
    (45, "45 min", "45 min"),
    (60, "1 h", "1 h"),
    (90, "+ d'1 h", "1 h +"),
]
# Cuisines du dataset : nom envoyé au modèle en français, puis en anglais.
CUISINES = [
    ("🇰🇷", "coréenne", "Korean"),
    ("🇲🇦", "marocaine", "Moroccan"),
    ("🇲🇽", "mexicaine", "Mexican"),
    ("🇮🇳", "indienne", "Indian"),
    ("🇯🇵", "japonaise", "Japanese"),
    ("🇱🇧", "libanaise", "Lebanese"),
    ("🇮🇹", "italienne", "Italian"),
    ("🇵🇪", "péruvienne", "Peruvian"),
    ("🇹🇭", "thaïlandaise", "Thai"),
    ("🇸🇳", "sénégalaise", "Senegalese"),
]
EXAMPLE_FRIDGES = {
    "fr": [
        "œufs, épinards, feta, citron",
        "blanc de poulet, riz cuit, poivron, sauce soja",
        "pois chiches, tomates, oignons, cumin",
    ],
    "en": [
        "eggs, spinach, feta, lemon",
        "chicken breast, cooked rice, bell pepper, soy sauce",
        "chickpeas, tomatoes, onions, cumin",
    ],
}

TEXTS = {
    "fr": {
        "welcome": (
            "Dis-moi ce qu'il reste dans ton frigo : je t'invente un plat inspiré d'une "
            "cuisine du monde, avec tes appareils et le temps que tu as.\n\n"
            "**Étape 1 sur 4 :** écris tes ingrédients, séparés par des virgules, "
            "ou essaie un de ces frigos."
        ),
        "appliances": "**Étape 2 sur 4 :** quels appareils as-tu ? Touche-les, puis valide.",
        "selected": "Choisis : {}",
        "none": "aucun pour l'instant",
        "validate": "✅ Valider",
        "need_appliance": "Choisis au moins un appareil.",
        "time": "**Étape 3 sur 4 :** combien de temps as-tu ?",
        "cuisine": "**Étape 4 sur 4 :** quelle cuisine t'inspire ?",
        "surprise": "🎲 Surprise",
        "other": "✏️ Autre",
        "other_prompt": "Écris la cuisine qui te tente, par exemple : grecque.",
        "use_buttons": "Réponds avec les boutons ci-dessus.",
        "again": "Écris un nouveau frigo pour une autre recette.",
        "thinking": "🧑‍🍳 Le chef cuisine ta recette…",
        "demo": "*Mode démo : recette préécrite, sans lien avec ta demande.*",
        "warning": "⚠️ **Contraintes non respectées :**",
        "cooked": "🍳 Je l'ai cuisinée",
        "another": "🔁 Autre recette",
        "rated": "Noté {}/5. Un commentaire ? Écris-le, ou passe.",
        "skip": "Passer",
        "thanks": "Merci pour ton retour !",
        "enjoy": "Bon appétit ! 🍽️",
        "switch": "🇬🇧 English",
    },
    "en": {
        "welcome": (
            "Tell me what's left in your fridge: I'll invent a dish inspired by a world "
            "cuisine, with your appliances and the time you have.\n\n"
            "**Step 1 of 4:** type your ingredients, separated by commas, "
            "or try one of these fridges."
        ),
        "appliances": "**Step 2 of 4:** which appliances do you have? Tap them, then confirm.",
        "selected": "Chosen: {}",
        "none": "none yet",
        "validate": "✅ Confirm",
        "need_appliance": "Pick at least one appliance.",
        "time": "**Step 3 of 4:** how much time do you have?",
        "cuisine": "**Step 4 of 4:** which cuisine inspires you?",
        "surprise": "🎲 Surprise me",
        "other": "✏️ Other",
        "other_prompt": "Type the cuisine you'd like, for example: Greek.",
        "use_buttons": "Please answer with the buttons above.",
        "again": "Type a new fridge for another recipe.",
        "thinking": "🧑‍🍳 The chef is cooking your recipe…",
        "demo": "*Demo mode: pre-written recipe, unrelated to your request.*",
        "warning": "⚠️ **Constraints not met:**",
        "cooked": "🍳 I cooked it",
        "another": "🔁 Another recipe",
        "rated": "Rated {}/5. Any comment? Type it, or skip.",
        "skip": "Skip",
        "thanks": "Thanks for your feedback!",
        "enjoy": "Enjoy your meal! 🍽️",
        "switch": "🇫🇷 Français",
    },
}


def language() -> str:
    return cl.user_session.get("language") or "fr"


def texts() -> dict:
    return TEXTS[language()]


def language_index() -> int:
    return 0 if language() == "fr" else 1


def appliance_names(keys: list[str]) -> list[str]:
    """Noms envoyés au modèle, dans la langue de son dataset."""
    if language() == "fr":
        return list(keys)
    return [APPLIANCE_NAMES_EN[key] for key in keys]


async def say_as_user(content: str) -> None:
    """Affiche un choix fait au bouton comme si l'utilisateur l'avait écrit."""
    await cl.Message(content=content, author="user", type="user_message").send()


async def send_picker(content: str, actions: list[cl.Action]) -> None:
    """Un seul message à boutons à la fois : l'ancien disparaît."""
    old = cl.user_session.get("picker")
    if old is not None:
        await old.remove()
    picker = cl.Message(content=content, actions=actions)
    await picker.send()
    cl.user_session.set("picker", picker)


async def close_picker(summary: str) -> None:
    """Remplace les boutons par le choix, pour garder la conversation lisible."""
    picker = cl.user_session.get("picker")
    if picker is not None:
        await picker.remove()
    cl.user_session.set("picker", None)
    await say_as_user(summary)


async def send_welcome() -> None:
    session = cl.user_session
    session.set("step", "fridge")
    session.set("picker", None)
    actions = [
        cl.Action(name="example_fridge", label=f"🧊 {fridge}", payload={"fridge": fridge})
        for fridge in EXAMPLE_FRIDGES[language()]
    ]
    actions.append(cl.Action(name="switch_language", label=texts()["switch"], payload={}))
    await cl.Message(content=f"{MAGNET_TITLE}\n\n{texts()['welcome']}", actions=actions).send()


async def send_appliance_picker() -> None:
    selected = cl.user_session.get("selected_appliances") or []
    index = language_index()
    actions = [
        cl.Action(
            name="toggle_appliance",
            label=("✔️ " if key in selected else "") + labels[index],
            payload={"key": key},
        )
        for key, labels in APPLIANCES.items()
    ]
    actions.append(cl.Action(name="validate_appliances", label=texts()["validate"], payload={}))
    chosen = ", ".join(APPLIANCES[key][index] for key in selected) or texts()["none"]
    await send_picker(f"{texts()['appliances']}\n\n{texts()['selected'].format(chosen)}", actions)


async def send_time_picker() -> None:
    index = language_index()
    actions = [
        cl.Action(name="choose_time", label=labels[index], payload={"minutes": minutes})
        for minutes, *labels in TIMES
    ]
    await send_picker(texts()["time"], actions)


async def send_cuisine_picker() -> None:
    index = language_index()
    actions = [
        cl.Action(
            name="choose_cuisine",
            label=f"{flag} {names[index].capitalize()}",
            payload={"cuisine": names[index]},
        )
        for flag, *names in CUISINES
    ]
    actions.append(
        cl.Action(name="choose_cuisine", label=texts()["surprise"], payload={"cuisine": "surprise"})
    )
    actions.append(cl.Action(name="other_cuisine", label=texts()["other"], payload={}))
    await send_picker(texts()["cuisine"], actions)


def recipe_actions(trace_id: str) -> list[cl.Action]:
    actions = [
        cl.Action(name="rate", label="⭐" * stars, payload={"stars": stars, "trace_id": trace_id})
        for stars in range(1, 6)
    ]
    return [
        *actions,
        cl.Action(name="cooked", label=texts()["cooked"], payload={"trace_id": trace_id}),
        cl.Action(name="regenerate", label=texts()["another"], payload={"trace_id": trace_id}),
    ]


async def start_with_fridge(text: str) -> None:
    session = cl.user_session
    session.set("raw_ingredients", text)
    session.set("selected_appliances", [])
    session.set("step", "appliances")
    await send_appliance_picker()


async def finish_with_cuisine(cuisine: str) -> None:
    session = cl.user_session
    appliances = appliance_names(session.get("selected_appliances"))
    minutes = session.get("minutes")
    raw_ingredients = session.get("raw_ingredients")
    problems = check_input(
        f"{raw_ingredients}, {cuisine}", ", ".join(appliances), minutes, language()
    )
    if problems:
        session.set("step", "fridge")
        await cl.Message(content=" ".join(problems) + f"\n\n{texts()['again']}").send()
        return
    session.set("ingredients", split_list(raw_ingredients))
    session.set("appliances", appliances)
    session.set("cuisine", cuisine)
    await create_recipe()


async def create_recipe() -> None:
    session = cl.user_session
    session.set("step", "cooking")
    backend = session.get("backend")
    if backend is None:
        backend = get_backend()
        session.set("backend", backend)

    waiting = cl.Message(content=texts()["thinking"])
    await waiting.send()
    result = await cl.make_async(generate_recipe)(
        backend,
        session.get("ingredients"),
        session.get("appliances"),
        session.get("minutes"),
        session.get("id"),
        language(),
        session.get("cuisine") or "surprise",
    )
    await waiting.remove()

    content = result.text
    if result.demo:
        content = f"{texts()['demo']}\n\n{content}"
    if result.warnings:
        content += f"\n\n{texts()['warning']} " + " ".join(result.warnings)
    await cl.Message(content=content, actions=recipe_actions(result.trace_id or "")).send()
    session.set("step", "fridge")


@cl.on_chat_start
async def start() -> None:
    cl.user_session.set("language", "fr")
    await send_welcome()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    session = cl.user_session
    text = message.content.strip()
    step = session.get("step")

    if step == "comment":
        send_feedback(session.get("rated_trace"), "user_comment", 1, comment=text)
        session.set("step", "fridge")
        await cl.Message(content=f"{texts()['thanks']} {texts()['again']}").send()
    elif step == "fridge":
        await start_with_fridge(text)
    elif step == "cuisine_text":
        await finish_with_cuisine(text)
    elif step != "cooking":
        await cl.Message(content=texts()["use_buttons"]).send()


@cl.action_callback("switch_language")
async def on_switch_language(action: cl.Action) -> None:
    cl.user_session.set("language", "en" if language() == "fr" else "fr")
    await send_welcome()


@cl.action_callback("example_fridge")
async def on_example_fridge(action: cl.Action) -> None:
    if cl.user_session.get("step") != "fridge":
        return
    fridge = action.payload["fridge"]
    await say_as_user(fridge)
    await start_with_fridge(fridge)


@cl.action_callback("toggle_appliance")
async def on_toggle_appliance(action: cl.Action) -> None:
    if cl.user_session.get("step") != "appliances":
        return
    selected = list(cl.user_session.get("selected_appliances") or [])
    key = action.payload["key"]
    if key in selected:
        selected.remove(key)
    else:
        selected.append(key)
    cl.user_session.set("selected_appliances", selected)
    await send_appliance_picker()


@cl.action_callback("validate_appliances")
async def on_validate_appliances(action: cl.Action) -> None:
    session = cl.user_session
    if session.get("step") != "appliances":
        return
    if not session.get("selected_appliances"):
        await cl.Message(content=texts()["need_appliance"]).send()
        return
    index = language_index()
    await close_picker(
        ", ".join(APPLIANCES[key][index] for key in session.get("selected_appliances"))
    )
    session.set("step", "time")
    await send_time_picker()


@cl.action_callback("choose_time")
async def on_choose_time(action: cl.Action) -> None:
    session = cl.user_session
    if session.get("step") != "time":
        return
    session.set("minutes", int(action.payload["minutes"]))
    await close_picker(f"⏱️ {action.label}")
    session.set("step", "cuisine")
    await send_cuisine_picker()


@cl.action_callback("choose_cuisine")
async def on_choose_cuisine(action: cl.Action) -> None:
    if cl.user_session.get("step") != "cuisine":
        return
    await close_picker(action.label)
    await finish_with_cuisine(action.payload["cuisine"])


@cl.action_callback("other_cuisine")
async def on_other_cuisine(action: cl.Action) -> None:
    session = cl.user_session
    if session.get("step") != "cuisine":
        return
    await close_picker(texts()["other"])
    session.set("step", "cuisine_text")
    await cl.Message(content=texts()["other_prompt"]).send()


@cl.action_callback("rate")
async def on_rate(action: cl.Action) -> None:
    stars, trace_id = action.payload["stars"], action.payload["trace_id"]
    send_feedback(trace_id, "user_rating", stars)
    cl.user_session.set("rated_trace", trace_id)
    cl.user_session.set("step", "comment")
    await cl.Message(
        content=texts()["rated"].format(stars),
        actions=[cl.Action(name="skip_comment", label=texts()["skip"], payload={})],
    ).send()


@cl.action_callback("skip_comment")
async def on_skip(action: cl.Action) -> None:
    cl.user_session.set("step", "fridge")
    await action.remove()
    await cl.Message(content=texts()["again"]).send()


@cl.action_callback("cooked")
async def on_cooked(action: cl.Action) -> None:
    send_feedback(action.payload["trace_id"], "cooked", 1)
    await cl.Message(content=texts()["enjoy"]).send()


@cl.action_callback("regenerate")
async def on_regenerate(action: cl.Action) -> None:
    send_feedback(action.payload["trace_id"], "regenerated", 1)
    await create_recipe()

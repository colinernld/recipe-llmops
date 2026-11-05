"""Interface Chainlit : frigo, appareils, temps, recette, puis feedback."""

import chainlit as cl

from src.recipe_service import generate_recipe, get_backend, send_feedback
from src.shared.recette_core.contract import (
    check_input,
    parse_minutes,
    split_list,
)

TEXTS = {
    "fr": {
        "welcome": (
            "Bonjour ! Je suis une IA qui invente des recettes du monde entier avec ce qu'il "
            "te reste.\n\nVérifie toujours les allergènes et la cuisson : je peux me tromper."
            "\n\n**Qu'y a-t-il dans ton frigo ?** Sépare les ingrédients par des virgules."
        ),
        "appliances": "**Quels appareils as-tu ?** Par exemple : plaques, four, micro-ondes.",
        "time": "**Combien de temps as-tu ?** Par exemple : 30 min ou 1h.",
        "cuisine": "**Quelle cuisine t'inspire ?** Par exemple : coréenne, ou écris surprise.",
        "again": "Dis-moi ce qu'il y a dans ton frigo pour une nouvelle recette.",
        "thinking": "Le chef réfléchit",
        "demo": "*Mode démo : recette préécrite, sans lien avec ta demande.*",
        "warning": "⚠️ **Contraintes non respectées :**",
        "cooked": "🍳 Je l'ai cuisinée",
        "another": "🔁 Autre recette",
        "rated": "Noté {}/5. Un commentaire ? Écris-le ici, ou clique sur Passer.",
        "skip": "Passer",
        "thanks": "Merci pour ton retour !",
        "enjoy": "Super, bon appétit ! 🍽️",
        "switch": "🇬🇧 English",
    },
    "en": {
        "welcome": (
            "Hi! I'm an AI that invents recipes from around the world with what you have left."
            "\n\nAlways check allergens and cooking: I can make mistakes."
            "\n\n**What's in your fridge?** Separate ingredients with commas."
        ),
        "appliances": "**Which appliances do you have?** For example: stove, oven, microwave.",
        "time": "**How much time do you have?** For example: 30 min or 1h.",
        "cuisine": "**Which cuisine inspires you?** For example: Korean, or type surprise.",
        "again": "Tell me what's in your fridge for a new recipe.",
        "thinking": "The chef is thinking",
        "demo": "*Demo mode: pre-written recipe, unrelated to your request.*",
        "warning": "⚠️ **Constraints not met:**",
        "cooked": "🍳 I cooked it",
        "another": "🔁 Another recipe",
        "rated": "Rated {}/5. Any comment? Write it here, or click Skip.",
        "skip": "Skip",
        "thanks": "Thanks for your feedback!",
        "enjoy": "Great, enjoy your meal! 🍽️",
        "switch": "🇫🇷 Français",
    },
}


def texts() -> dict:
    return TEXTS[cl.user_session.get("language") or "fr"]


async def send_welcome() -> None:
    cl.user_session.set("step", "fridge")
    await cl.Message(
        content=texts()["welcome"],
        actions=[cl.Action(name="switch_language", label=texts()["switch"], payload={})],
    ).send()


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


async def create_recipe() -> None:
    session = cl.user_session
    backend = session.get("backend")
    if backend is None:
        backend = get_backend()
        session.set("backend", backend)

    async with cl.Step(name=texts()["thinking"]) as step:
        result = await cl.make_async(generate_recipe)(
            backend,
            session.get("ingredients"),
            session.get("appliances"),
            session.get("minutes"),
            session.get("id"),
            session.get("language") or "fr",
            session.get("cuisine") or "surprise",
        )
        step.output = f"{result.attempts} essai(s)"

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


@cl.action_callback("switch_language")
async def on_switch_language(action: cl.Action) -> None:
    current = cl.user_session.get("language")
    cl.user_session.set("language", "en" if current == "fr" else "fr")
    await action.remove()
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
        session.set("raw_ingredients", text)
        session.set("step", "appliances")
        await cl.Message(content=texts()["appliances"]).send()
    elif step == "appliances":
        session.set("raw_appliances", text)
        session.set("step", "time")
        await cl.Message(content=texts()["time"]).send()
    elif step == "time":
        session.set("raw_minutes", text)
        session.set("step", "cuisine")
        await cl.Message(content=texts()["cuisine"]).send()
    else:
        minutes = parse_minutes(session.get("raw_minutes"))
        raw_ingredients = f"{session.get('raw_ingredients')}, {text}"
        problems = check_input(
            raw_ingredients, session.get("raw_appliances"), minutes, session.get("language")
        )
        if problems:
            session.set("step", "fridge")
            await cl.Message(content=" ".join(problems) + f"\n\n{texts()['again']}").send()
            return
        session.set("ingredients", split_list(session.get("raw_ingredients")))
        session.set("appliances", split_list(session.get("raw_appliances")))
        session.set("minutes", minutes)
        session.set("cuisine", text)
        await create_recipe()


@cl.action_callback("rate")
async def on_rate(action: cl.Action) -> None:
    stars, trace_id = action.payload["stars"], action.payload["trace_id"]
    send_feedback(trace_id, "user_rating", stars)
    cl.user_session.set("rated_trace", trace_id)
    cl.user_session.set("step", "comment")
    await action.remove()
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

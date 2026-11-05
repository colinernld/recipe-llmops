import json
from pathlib import Path

from recette_core.contract import check_recipe
from src import recipe_service as service
from src.recipe_service import DemoBackend, ModelReply, generate_recipe, send_feedback

GOOD = json.loads(Path("data/sample_recipes.jsonl").read_text(encoding="utf-8").splitlines()[0])
ARGS = (GOOD["ingredients"], GOOD["appliances"], GOOD["max_minutes"], "session")


class Scripted:
    def __init__(self, *texts):
        self.texts, self.prompts = list(texts), []

    def predict(self, prompt, parameters):
        self.prompts.append(prompt)
        return ModelReply(self.texts.pop(0), "model", "1")


def test_valid_recipe_needs_one_attempt():
    backend = Scripted(GOOD["recipe"])
    result = generate_recipe(backend, *ARGS)
    assert result.attempts == 1 and result.warnings == [] and result.report["all_ok"]
    assert "<|user|>" in backend.prompts[0]


def test_bad_output_is_retried_then_accepted():
    result = generate_recipe(Scripted("n'importe quoi", GOOD["recipe"]), *ARGS)
    assert result.attempts == 2 and result.warnings == []


def test_repeated_failure_is_shown_to_the_user():
    bad = GOOD["recipe"].replace("feta : 60 g", "caviar : 60 g")
    result = generate_recipe(Scripted(bad, bad), *ARGS)
    assert result.attempts == 2
    assert any("caviar" in warning for warning in result.warnings)


def test_food_safety_note_is_added():
    raw = GOOD["recipe"].replace("feta : 60 g", "poulet : 60 g")
    result = generate_recipe(Scripted(raw), ["œufs", "courgettes", "poulet", "citron"], *ARGS[1:])
    assert "cuire la viande à cœur" in result.text


def test_demo_backend_returns_valid_recipes():
    reply = DemoBackend().predict("", {})
    assert check_recipe(reply.text, ["x"], ["four"], 60)["format_ok"]


def test_demo_mode_without_endpoint(monkeypatch):
    monkeypatch.setattr(service, "ENDPOINT_ID", "")
    assert isinstance(service.get_backend(), DemoBackend)


def test_english_request_stays_in_english():
    row = next(
        json.loads(line)
        for line in Path("data/sample_recipes.jsonl").read_text(encoding="utf-8").splitlines()
        if json.loads(line)["language"] == "en"
    )
    backend = Scripted(row["recipe"].replace("kimchi: 100 g", "caviar: 100 g"))
    backend.texts.append(backend.texts[0])
    result = generate_recipe(
        backend, row["ingredients"], row["appliances"], row["max_minutes"], "s", "en", "Korean"
    )
    assert "### Steps" in result.text
    assert result.warnings == ["Extra ingredients: caviar."]
    assert "Cuisine: Korean" in backend.prompts[0]


def test_demo_backend_matches_language():
    assert DemoBackend().predict("Fridge: eggs", {}).text.startswith("Title:")


def test_feedback_is_attached_to_the_trace(monkeypatch):
    sent = []

    class Client:
        def create_score(self, **kwargs):
            sent.append(kwargs)

    monkeypatch.setattr(service, "get_client", lambda: Client())
    send_feedback("trace-1", "user_rating", 4)
    send_feedback(None, "user_rating", 4)
    assert sent == [
        {
            "trace_id": "trace-1",
            "name": "user_rating",
            "value": 4,
            "comment": None,
            "score_id": "trace-1-user_rating",
        }
    ]

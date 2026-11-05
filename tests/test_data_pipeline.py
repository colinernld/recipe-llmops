import importlib.util
import yaml
import random
from pathlib import Path

from kfp import compiler

from recette_core.data import clean_rows, format_rows, local_path, read_jsonl, split_rows

SAMPLE = "data/sample_recipes.jsonl"
spec = importlib.util.spec_from_file_location("generate_dataset", "scripts/generate_dataset.py")
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


def test_local_path():
    assert local_path("gs://b/d.jsonl") == "/gcs/b/d.jsonl"
    assert local_path("data/d.jsonl") == "data/d.jsonl"


def test_clean_rejects_bad_rows_and_duplicates():
    rows = read_jsonl(SAMPLE)
    dirty = [
        *rows,
        rows[0],
        {**rows[1], "max_minutes": 1},
        {**rows[2], "recipe": "texte libre"},
        {"recipe": "sans champs"},
    ]
    kept, stats = clean_rows(dirty)
    assert len(kept) == len(rows)
    assert stats["rejected_duplicate"] == 1
    assert stats["rejected_constraints"] == 1
    assert stats["rejected_format"] == 1
    assert stats["rejected_schema"] == 1


def test_clean_removes_html():
    row = read_jsonl(SAMPLE)[0]
    kept, _ = clean_rows([{**row, "recipe": f"<p>{row['recipe']}</p>"}])
    assert "<p>" not in kept[0]["recipe"]


def test_split_is_stable_and_disjoint():
    rows = read_jsonl(SAMPLE)
    train, val, test = split_rows(rows, 1, 0.2, 0.2)
    assert split_rows(rows, 1, 0.2, 0.2) == (train, val, test)
    assert len(train) + len(val) + len(test) == len(rows)
    titles = [{r["recipe"] for r in part} for part in (train, val, test)]
    assert not (titles[0] & titles[1] or titles[0] & titles[2] or titles[1] & titles[2])


def test_format_adds_prompt_and_completion():
    row = format_rows(read_jsonl(SAMPLE))[0]
    assert "<|assistant|>" in row["prompt"]
    assert row["completion"].startswith("Titre :")


def test_pipeline_compiles_with_expected_steps(tmp_path):
    from src.pipelines.training_pipeline import recette_frigo_pipeline

    target = tmp_path / "pipeline.yaml"
    compiler.Compiler().compile(recette_frigo_pipeline, str(target))
    spec_json = yaml.safe_load(target.read_text())
    tasks = spec_json["root"]["dag"]["tasks"]
    assert set(tasks) == {
        "clean-dataset", "format-dataset", "format-dataset-2", "format-dataset-3",
        "fine-tune-model", "predict-finetuned", "predict-baseline", "evaluate-model",
    }
    resources = spec_json["deploymentSpec"]["executors"]["exec-fine-tune-model"]["container"]
    assert resources["resources"]["accelerator"]["type"] == "NVIDIA_TESLA_T4"


def valid_payload(scenario):
    return {
        "title": f"Plat {random.random()}",
        "total_minutes": scenario["max_minutes"],
        "appliances": scenario["appliances"][:1],
        "ingredients": [f"{scenario['ingredients'][0]}: 100 g"],
        "steps": ["Prépare.", "Sers."],
    }


def test_build_dataset_keeps_valid_examples_and_resumes(tmp_path):
    rng = random.Random(0)
    output = tmp_path / "recipes.jsonl"
    calls = []

    def fake(prompt):
        calls.append(prompt)
        if len(calls) % 3 == 0:
            raise RuntimeError("quota")
        if len(calls) % 3 == 1:
            return {"title": "Sans le reste"}
        scenario = generator.sample_scenario(rng)
        scenario["ingredients"] = prompt.split("from this list: ")[1].split(".\n")[0].split(", ")
        scenario["appliances"] = prompt.split("these appliances: ")[1].split(".\n")[0].split(", ")
        scenario["max_minutes"] = int(prompt.split("must not exceed ")[1].split(".")[0])
        return valid_payload(scenario)

    stats = generator.build_dataset(fake, 5, 1, output, workers=1)
    assert stats["kept"] == 5
    assert stats["rejected"] > 0 and stats["errors"] > 0
    stats = generator.build_dataset(fake, 7, 1, output, workers=1)
    rows = read_jsonl(str(output))
    assert stats["kept"] == 7 and len(rows) == 7
    assert clean_rows(rows)[1]["kept"] == 7


def test_build_dataset_stops_when_budget_is_spent(tmp_path):
    stats = generator.build_dataset(
        lambda prompt: {}, 3, 1, Path(tmp_path / "x.jsonl"), workers=1, max_attempts=2
    )
    assert stats["kept"] == 0 and stats["rejected"] == 6


def test_scenarios_mix_languages_and_cuisines():
    rng = random.Random(0)
    scenarios = [generator.sample_scenario(rng) for _ in range(300)]
    assert {s["language"] for s in scenarios} == {"fr", "en"}
    assert len({s["cuisine"] for s in scenarios}) > 30
    assert any(s["cuisine"] == "surprise" for s in scenarios)
    english = [s for s in scenarios if s["language"] == "en"]
    assert all("plaques" not in s["appliances"] for s in english)
    assert "French" in generator.build_generation_prompt(
        next(s for s in scenarios if s["language"] == "fr")
    )


def test_example_keeps_cuisine_language_and_twist():
    scenario = {
        "language": "en", "ingredients": ["eggs"], "appliances": ["stove"], "max_minutes": 20,
        "cuisine": "Thai", "twist": "a surprising contrast of textures",
    }
    example = generator.to_example(scenario, valid_payload(scenario))
    assert example["recipe"].startswith("Title:")
    assert "Cuisine: Thai" in example["user_prompt"]
    assert example["twist"] == scenario["twist"]

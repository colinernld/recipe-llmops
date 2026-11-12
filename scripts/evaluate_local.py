"""Évalue les prédictions du pipeline sur ton ordinateur et les enregistre dans Vertex AI Experiments.

Utilisé quand l'étape d'évaluation du pipeline échoue : les prédictions GPU sont déjà calculées.

    uv run --with "ragas==0.3.9" --with "langchain<1" --with "langchain-core<1" \
      --with "langchain-community<0.4" --with "langchain-openai<1" --with sacrebleu \
      --with rouge_score python scripts/evaluate_local.py --run-name evaluation-phi3-lora-v1
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "shared"))
sys.path.insert(0, str(ROOT))

from recette_core.data import read_jsonl  # noqa: E402
from recette_core.evaluation import constraint_metrics, text_metrics  # noqa: E402


def compute(finetuned_path: str, baseline_path: str, dataset_path: str) -> dict[str, float]:
    finetuned, baseline = read_jsonl(finetuned_path), read_jsonl(baseline_path)
    test_refs = {row["recipe"] for row in finetuned}
    # La nouveauté se mesure contre les recettes d'entraînement, sans celles du test.
    corpus = [row["recipe"] for row in read_jsonl(dataset_path) if row["recipe"] not in test_refs]
    scores = {**constraint_metrics(finetuned, corpus), **text_metrics(finetuned)}
    reference = {**constraint_metrics(baseline, corpus), **text_metrics(baseline)}
    return {**scores, **{f"baseline_{name}": value for name, value in reference.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finetuned", default="data/eval/finetuned.jsonl")
    parser.add_argument("--baseline", default="data/eval/baseline.jsonl")
    parser.add_argument("--dataset", default="data/recipes.jsonl")
    parser.add_argument("--run-name", help="Nom de l'exécution dans Vertex AI Experiments.")
    parser.add_argument("--pipeline-run", default="recette-frigo-training-20261111105036")
    args = parser.parse_args()

    metrics = compute(args.finetuned, args.baseline, args.dataset)
    for name, value in metrics.items():
        print(f"{name:40} {value:.3f}")

    if args.run_name:
        from google.cloud import aiplatform

        from src.constants import BASE_MODEL, EXPERIMENT_NAME, PROJECT_ID, REGION

        aiplatform.init(project=PROJECT_ID, location=REGION, experiment=EXPERIMENT_NAME)
        aiplatform.start_run(args.run_name)
        aiplatform.log_params({"pipeline_run": args.pipeline_run, "base_model": BASE_MODEL})
        aiplatform.log_metrics({name: float(value) for name, value in metrics.items()})
        aiplatform.end_run()
        print(f"Enregistré dans Vertex AI Experiments : {EXPERIMENT_NAME}/{args.run_name}")


if __name__ == "__main__":
    main()

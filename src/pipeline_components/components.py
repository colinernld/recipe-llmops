"""Composants Kubeflow. La logique vit dans recette_core, embarqué dans chaque composant."""

from pathlib import Path

from kfp import dsl
from kfp.dsl import Dataset, Input, Metrics, Model, Output

CORE = str(Path(__file__).resolve().parents[1] / "shared")
CPU_IMAGE = "python:3.11-slim-bookworm"
# Image PyTorch officielle : elle indique où trouver le pilote GPU de la machine.
GPU_IMAGE = "pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime"
GPU_PACKAGES = [
    "torch==2.3.1",
    "transformers==4.48.0",
    "peft==0.13.2",  # Même version que le conteneur de serving.
    "accelerate==1.2.1",
    "bitsandbytes==0.45.0",
    "numpy<2",
]


@dsl.component(base_image=CPU_IMAGE, embedded_artifact_path=CORE)
def clean_dataset(
    core: dsl.EmbeddedInput[dsl.Artifact],
    raw_dataset_uri: str,
    seed: int,
    val_ratio: float,
    test_ratio: float,
    train_data: Output[Dataset],
    val_data: Output[Dataset],
    test_data: Output[Dataset],
    metrics: Output[Metrics],
):
    """Étape indépendante du modèle : contrôle, déduplication et découpage."""
    from recette_core.data import clean_rows, local_path, read_jsonl, split_rows, write_jsonl

    rows, stats = clean_rows(read_jsonl(local_path(raw_dataset_uri)))
    train, val, test = split_rows(rows, seed, val_ratio, test_ratio)
    write_jsonl(train, train_data.path)
    write_jsonl(val, val_data.path)
    write_jsonl(test, test_data.path)
    for name, value in stats.items():
        metrics.log_metric(name, value)
    metrics.log_metric("train_rows", len(train))
    metrics.log_metric("val_rows", len(val))
    metrics.log_metric("test_rows", len(test))


@dsl.component(base_image=CPU_IMAGE, embedded_artifact_path=CORE)
def format_dataset(
    core: dsl.EmbeddedInput[dsl.Artifact],
    input_data: Input[Dataset],
    output_data: Output[Dataset],
):
    """Étape propre à Phi-3 : applique son modèle de conversation."""
    from recette_core.data import format_rows, read_jsonl, write_jsonl

    write_jsonl(format_rows(read_jsonl(input_data.path)), output_data.path)


@dsl.component(base_image=GPU_IMAGE, packages_to_install=GPU_PACKAGES, embedded_artifact_path=CORE)
def fine_tune_model(
    core: dsl.EmbeddedInput[dsl.Artifact],
    train_data: Input[Dataset],
    val_data: Input[Dataset],
    base_model: str,
    learning_rate: float,
    num_epochs: int,
    batch_size: int,
    lora_r: int,
    lora_alpha: int,
    model: Output[Model],
    metrics: Output[Metrics],
):
    from recette_core.data import read_jsonl
    from recette_core.training import TrainingConfig, fine_tune

    config = TrainingConfig(
        base_model=base_model,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        batch_size=batch_size,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
    )
    results = fine_tune(read_jsonl(train_data.path), read_jsonl(val_data.path), config, model.path)
    for name, value in results.items():
        metrics.log_metric(name, value)
    model.metadata["base_model"] = base_model


@dsl.component(base_image=GPU_IMAGE, packages_to_install=GPU_PACKAGES, embedded_artifact_path=CORE)
def predict_finetuned(
    core: dsl.EmbeddedInput[dsl.Artifact],
    test_data: Input[Dataset],
    model: Input[Model],
    base_model: str,
    max_samples: int,
    predictions: Output[Dataset],
):
    from recette_core.data import read_jsonl, write_jsonl
    from recette_core.inference import predict_rows

    rows = read_jsonl(test_data.path)[:max_samples]
    write_jsonl(predict_rows(rows, base_model, model.path), predictions.path)


@dsl.component(base_image=GPU_IMAGE, packages_to_install=GPU_PACKAGES, embedded_artifact_path=CORE)
def predict_baseline(
    core: dsl.EmbeddedInput[dsl.Artifact],
    test_data: Input[Dataset],
    base_model: str,
    max_samples: int,
    predictions: Output[Dataset],
):
    """Même test sur Phi-3 sans fine-tuning, pour mesurer le gain réel."""
    from recette_core.data import read_jsonl, write_jsonl
    from recette_core.inference import predict_rows

    rows = read_jsonl(test_data.path)[:max_samples]
    write_jsonl(predict_rows(rows, base_model, None), predictions.path)


@dsl.component(
    base_image=CPU_IMAGE,
    packages_to_install=[
        "ragas==0.3.9",
        "langchain<1",
        "langchain-core<1",
        "langchain-community<0.4",
        "langchain-openai<1",
        "langchain-google-vertexai<3",
        "sacrebleu",
        "rouge_score",
    ],
    embedded_artifact_path=CORE,
)
def evaluate_model(
    core: dsl.EmbeddedInput[dsl.Artifact],
    finetuned_predictions: Input[Dataset],
    baseline_predictions: Input[Dataset],
    train_data: Input[Dataset],
    project_id: str,
    gemini_model: str,
    gemini_location: str,
    judge_samples: int,
    metrics: Output[Metrics],
) -> float:
    """Compare le modèle fine-tuné à Phi-3 seul et renvoie le taux de contraintes respectées."""
    from recette_core.data import read_jsonl
    from recette_core.evaluation import (
        constraint_metrics,
        judge_metrics,
        text_metrics,
        vertex_judge,
    )

    corpus = [row["recipe"] for row in read_jsonl(train_data.path)]
    finetuned = read_jsonl(finetuned_predictions.path)
    baseline = read_jsonl(baseline_predictions.path)

    scores = {**constraint_metrics(finetuned, corpus), **text_metrics(finetuned)}
    reference = {**constraint_metrics(baseline, corpus), **text_metrics(baseline)}
    if judge_samples > 0:
        judge = vertex_judge(gemini_model, project_id, gemini_location)
        scores.update(judge_metrics(finetuned[:judge_samples], judge))
        reference.update(judge_metrics(baseline[:judge_samples], judge))

    for name, value in scores.items():
        metrics.log_metric(name, value)
        if name in reference:
            metrics.log_metric(f"baseline_{name}", reference[name])
    return scores["all_constraints_rate"]

"""Pipeline d'entraînement : nettoyer, formater, fine-tuner, prédire, évaluer."""

from kfp import dsl

from src.pipeline_components.components import (
    clean_dataset,
    evaluate_model,
    fine_tune_model,
    format_dataset,
    predict_baseline,
    predict_finetuned,
)


def _use_gpu(task):
    """Une machine équivalente à n1-standard-8 avec un GPU T4."""
    return (
        task.set_cpu_limit("8")
        .set_memory_limit("30G")
        .set_accelerator_type("NVIDIA_TESLA_T4")
        .set_accelerator_limit(1)
    )


@dsl.pipeline(name="recette-frigo-training")
def recette_frigo_pipeline(
    raw_dataset_uri: str,
    project_id: str,
    base_model: str = "microsoft/Phi-3-mini-4k-instruct",
    gemini_model: str = "gemini-3.8-flash",
    gemini_location: str = "global",
    seed: int = 42,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    learning_rate: float = 2e-4,
    num_epochs: int = 3,
    batch_size: int = 4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    predict_samples: int = 100,
    judge_samples: int = 30,
):
    cleaned = clean_dataset(
        raw_dataset_uri=raw_dataset_uri, seed=seed, val_ratio=val_ratio, test_ratio=test_ratio
    ).set_display_name("Nettoyage")

    train = format_dataset(input_data=cleaned.outputs["train_data"]).set_display_name(
        "Formatage train"
    )
    val = format_dataset(input_data=cleaned.outputs["val_data"]).set_display_name("Formatage val")
    test = format_dataset(input_data=cleaned.outputs["test_data"]).set_display_name(
        "Formatage test"
    )

    tuned = _use_gpu(
        fine_tune_model(
            train_data=train.outputs["output_data"],
            val_data=val.outputs["output_data"],
            base_model=base_model,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            batch_size=batch_size,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
        )
    ).set_display_name("Fine-tuning LoRA")

    predicted = _use_gpu(
        predict_finetuned(
            test_data=test.outputs["output_data"],
            model=tuned.outputs["model"],
            base_model=base_model,
            max_samples=predict_samples,
        )
    ).set_display_name("Prédiction modèle fine-tuné")

    baseline = _use_gpu(
        predict_baseline(
            test_data=test.outputs["output_data"],
            base_model=base_model,
            max_samples=predict_samples,
        )
    ).set_display_name("Prédiction Phi-3 seul")
    # Un seul GPU à la fois : le quota T4 du projet est limité.
    baseline.after(predicted)

    evaluate_model(
        finetuned_predictions=predicted.outputs["predictions"],
        baseline_predictions=baseline.outputs["predictions"],
        train_data=train.outputs["output_data"],
        project_id=project_id,
        gemini_model=gemini_model,
        gemini_location=gemini_location,
        judge_samples=judge_samples,
    ).set_display_name("Évaluation")

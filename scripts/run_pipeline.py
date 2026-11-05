"""Compile le pipeline et le lance sur Vertex AI, avec suivi dans Vertex AI Experiments."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from google.cloud import aiplatform  # noqa: E402
from kfp import compiler  # noqa: E402

from src.constants import (  # noqa: E402
    BASE_MODEL,
    BUCKET,
    EXPERIMENT_NAME,
    GEMINI_LOCATION,
    GEMINI_MODEL,
    PIPELINE_ROOT,
    PROJECT_ID,
    RAW_DATASET_URI,
    REGION,
)
from src.pipelines.training_pipeline import recette_frigo_pipeline  # noqa: E402

TEMPLATE = str(ROOT / "pipeline.yaml")


def main() -> None:
    compiler.Compiler().compile(recette_frigo_pipeline, package_path=TEMPLATE)
    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=f"gs://{BUCKET}")
    job = aiplatform.PipelineJob(
        display_name="recette-frigo-training",
        template_path=TEMPLATE,
        pipeline_root=PIPELINE_ROOT,
        parameter_values={
            "raw_dataset_uri": RAW_DATASET_URI,
            "project_id": PROJECT_ID,
            "base_model": BASE_MODEL,
            "gemini_model": GEMINI_MODEL,
            "gemini_location": GEMINI_LOCATION,
        },
    )
    job.submit(experiment=EXPERIMENT_NAME)
    print(job._dashboard_uri())


if __name__ == "__main__":
    main()

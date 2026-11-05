"""Configuration lue depuis l'environnement (.env à la racine du projet)."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

PROJECT_ID = os.getenv("PROJECT_ID", "agentics-workflows-510809")
REGION = os.getenv("REGION", "europe-west2")
BUCKET = os.getenv("BUCKET", "ebcrjg-llmops")
PIPELINE_ROOT = f"gs://{BUCKET}/pipeline-root"
RAW_DATASET_URI = os.getenv("RAW_DATASET_URI", f"gs://{BUCKET}/data/recipes.jsonl")

BASE_MODEL = os.getenv("BASE_MODEL", "microsoft/Phi-3-mini-4k-instruct")
MODEL_DISPLAY_NAME = os.getenv("MODEL_DISPLAY_NAME", "recette-frigo-phi-3")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "recette-frigo")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
GEMINI_LOCATION = os.getenv("GEMINI_LOCATION", "global")

SERVING_IMAGE = os.getenv(
    "SERVING_IMAGE",
    "us-docker.pkg.dev/deeplearning-platform-release/gcr.io/"
    "huggingface-pytorch-inference-cu121.2-3.transformers.4-48.ubuntu2204.py311",
)
MACHINE_TYPE = os.getenv("MACHINE_TYPE", "n1-standard-8")
ACCELERATOR_TYPE = os.getenv("ACCELERATOR_TYPE", "NVIDIA_TESLA_T4")

ENDPOINT_ID = os.getenv("ENDPOINT_ID", "")

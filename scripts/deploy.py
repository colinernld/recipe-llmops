"""Enregistre le modèle, le déploie sur un endpoint, ou supprime cet endpoint.

Exemples :
    uv run python scripts/deploy.py register --artifact-uri gs://.../fine-tune-model_.../model
    uv run python scripts/deploy.py deploy --model-id 1234567890
    uv run python scripts/deploy.py delete --endpoint-id 1234567890

Un endpoint est facturé à l'heure, même sans trafic : supprime-le après chaque démo.
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from google.cloud import aiplatform  # noqa: E402

from src.constants import (  # noqa: E402
    ACCELERATOR_TYPE,
    BASE_MODEL,
    MACHINE_TYPE,
    MODEL_DISPLAY_NAME,
    PROJECT_ID,
    REGION,
    SERVING_IMAGE,
)


def register(artifact_uri: str, parent_model: str | None) -> None:
    uri = artifact_uri.rstrip("/")
    # Le conteneur cherche handler.py à la racine des artefacts.
    subprocess.run(
        ["gcloud", "storage", "cp", str(ROOT / "src" / "handler.py"), f"{uri}/handler.py"],
        check=True,
    )
    model = aiplatform.Model.upload(
        display_name=MODEL_DISPLAY_NAME,
        artifact_uri=uri,
        serving_container_image_uri=SERVING_IMAGE,
        serving_container_ports=[8080],
        serving_container_environment_variables={"BASE_MODEL": BASE_MODEL},
        parent_model=parent_model,
    )
    print(f"Modèle enregistré : {model.resource_name} (version {model.version_id})")


def deploy(model_id: str) -> None:
    model = aiplatform.Model(model_id)
    endpoint = model.deploy(
        deployed_model_display_name=MODEL_DISPLAY_NAME,
        machine_type=MACHINE_TYPE,
        accelerator_type=ACCELERATOR_TYPE,
        accelerator_count=1,
        min_replica_count=1,
        max_replica_count=1,
        traffic_percentage=100,
    )
    print(f"Endpoint prêt. Ajoute ENDPOINT_ID={endpoint.name} à ton fichier .env")


def delete(endpoint_id: str) -> None:
    aiplatform.Endpoint(endpoint_id).delete(force=True)
    print(f"Endpoint {endpoint_id} supprimé.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    register_parser = commands.add_parser("register")
    register_parser.add_argument("--artifact-uri", required=True)
    register_parser.add_argument("--parent-model")
    commands.add_parser("deploy").add_argument("--model-id", required=True)
    commands.add_parser("delete").add_argument("--endpoint-id", required=True)
    args = parser.parse_args()

    aiplatform.init(project=PROJECT_ID, location=REGION)
    if args.command == "register":
        register(args.artifact_uri, args.parent_model)
    elif args.command == "deploy":
        deploy(args.model_id)
    else:
        delete(args.endpoint_id)


if __name__ == "__main__":
    main()

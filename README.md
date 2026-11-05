# Recette Frigo

Une plateforme bilingue, en français et en anglais, qui invente une recette originale inspirée des cuisines du monde, à partir de ce qu'il reste dans le frigo, des appareils disponibles et du temps que tu as. Chaque recette est notée par l'utilisateur, et cette note est reliée à la génération exacte dans Langfuse.

Le projet suit les étapes du cours : Kubeflow sur Vertex AI, fine-tuning LoRA de Phi-3, évaluation avec Ragas, Model Registry, endpoint Vertex AI, Chainlit et Langfuse.

## Structure

| Dossier ou fichier | Rôle |
| --- | --- |
| `scripts/generate_dataset.py` | Crée le dataset bilingue avec Gemini et garde les exemples valides |
| `src/shared/recette_core/` | Règles de recette, données, entraînement, inférence, évaluation |
| `src/pipeline_components/` | Composants Kubeflow |
| `src/pipelines/` | Pipeline : nettoyer, formater, fine-tuner, prédire, évaluer |
| `src/handler.py` | Handler du conteneur Hugging Face |
| `scripts/deploy.py` | Enregistre le modèle, le déploie, supprime l'endpoint |
| `src/recipe_service.py` | Génération, garde-fous et traces Langfuse |
| `app.py` | Interface Chainlit |

## Installation

```bash
uv sync
cp .env.example .env
gcloud auth application-default login
gcloud config set project agentics-workflows-510809
gcloud services enable aiplatform.googleapis.com storage.googleapis.com
```

Avant de lancer quoi que ce soit, vérifie deux points :

- le bucket `ebcrjg-llmops` doit être dans la même région que `REGION` (`europe-west2` par défaut) ;
- ton projet doit avoir un quota GPU T4 dans cette région.

## Test sans cloud

```bash
uv run python -m pytest
DEMO_MODE=1 uv run chainlit run app.py
```

En mode démo, l'application affiche des recettes préécrites et le signale à l'écran.

## Flow complet

**1. Créer le dataset.** Commence par 50 exemples pour vérifier le taux de rejet.

```bash
uv run --env-file .env python scripts/generate_dataset.py --n 50 --output data/test.jsonl
uv run --env-file .env python scripts/generate_dataset.py --n 3000
gcloud storage cp data/recipes.jsonl gs://ebcrjg-llmops/data/recipes.jsonl
```

**2. Lancer le pipeline.** Les métriques arrivent dans Vertex AI Experiments, sous `recette-frigo`.

```bash
uv run python scripts/run_pipeline.py
```

**3. Enregistrer le modèle.** L'URI se trouve sur l'artefact `model` de l'étape « Fine-tuning LoRA ».

```bash
uv run python scripts/deploy.py register --artifact-uri gs://ebcrjg-llmops/pipeline-root/.../model
```

**4. Déployer.** Compte de 15 à 30 minutes.

```bash
uv run python scripts/deploy.py deploy --model-id <MODEL_ID>
```

Ajoute ensuite `ENDPOINT_ID=<ID>` au fichier `.env`.

**5. Lancer Langfuse.** Crée un projet à l'adresse `http://localhost:3000`, puis copie ses clés dans `.env`.

```bash
git clone https://github.com/langfuse/langfuse.git
cd langfuse
docker compose up -d
```

**6. Lancer l'application.**

```bash
uv run chainlit run app.py
```

**7. Supprimer l'endpoint.** Il est facturé tant qu'il existe, même sans trafic.

```bash
uv run python scripts/deploy.py delete --endpoint-id <ENDPOINT_ID>
```

## Tests d'entraînement

Le test `tests/test_training_smoke.py` entraîne un minuscule Phi-3 aléatoire. Il demande un environnement à part, car ces bibliothèques ne servent pas à l'application :

```bash
uv venv .venv-train --python 3.11
uv pip install --python .venv-train/bin/python torch==2.3.1 transformers==4.48.0 peft==0.13.2 accelerate==1.2.1 "numpy<2" pytest python-dotenv "kfp>=2.18,<3" "langfuse>=3.15,<4"
.venv-train/bin/python -m pytest tests/test_training_smoke.py
```

Le test `tests/test_evaluation.py` demande `ragas==0.3.9`, `sacrebleu`, `rouge_score` et LangChain en version inférieure à 1. Sans ces paquets, il est ignoré.

## Écarts avec les slides

- `gemini-2.5-flash` a été retiré de Vertex AI le 20 octobre 2026. Le projet utilise `gemini-3.8-flash`.
- `ragas==0.3.9` est figé, avec LangChain en version inférieure à 1, car la dernière combinaison testée échouait à l'import.
- `ragas.llms` remplace `ragas.llm_wrappers`, qui figure sur le slide.
- Le conteneur de serving est configurable avec `SERVING_IMAGE`. Compare-le à celui de la solution du cours.

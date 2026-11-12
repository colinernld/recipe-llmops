# Journal des exécutions

Ce journal retrace ce qui a été lancé sur Google Cloud, avec quel code et pourquoi.
Il sert de support de review et de base pour la partie « problèmes rencontrés » de la présentation.

Projet GCP : `agentics-workflows-510809` (numéro `123905721265`).
Région Vertex AI : `europe-west2`. Bucket : `gs://ebcrjg-llmops`, en région `EU`.
Les coûts indiqués sont des estimations calculées à partir des tarifs publics : nous n'avons pas accès à la facture du projet.

## Résumé

| Étape | Résultat | Où le vérifier |
| --- | --- | --- |
| Dataset | 3 000 recettes, 2 999 gardées après nettoyage | `gs://ebcrjg-llmops/data/recipes.jsonl` |
| Fine-tuning LoRA de Phi-3 | 3 passages, `eval_loss` 0,478 après le 1er | Pipeline `recette-frigo-training-20261111105036` |
| Évaluation sur 100 frigos de test | 98 % des recettes respectent les 3 contraintes, contre 0 % pour Phi-3 seul | Vertex AI Experiments, `recette-frigo` / `evaluation-phi3-lora-v1` |
| Model Registry | Modèle `8819739119274950656`, version 1 | Vertex AI Model Registry |
| Endpoint | Test de déploiement en cours le 12 novembre | Endpoint `2352191822370963456` |

## Résultats de l'évaluation

| Mesure | Modèle fine-tuné | Phi-3 seul |
| --- | --- | --- |
| Format respecté | 100 % | 0 % |
| Ingrédients du frigo respectés | 98 % | 0 % |
| Appareils respectés | 100 % | 0 % |
| Temps respecté | 100 % | 0 % |
| Les 3 contraintes, en français | 100 % | 0 % |
| Les 3 contraintes, en anglais | 96,4 % | 0 % |
| Nouveauté par rapport au dataset | 0,537 | 0,718 |
| BLEU / ROUGE-1 | 0,557 / 0,646 | 0,067 / 0,485 |

Limites à connaître :

- Le 0 % de Phi-3 seul vient d'abord du format : il n'écrit jamais au format attendu, donc le contrôle ne peut pas lire ses recettes.
- Respecter les contraintes ne garantit pas une bonne cuisine. Exemple vu dans les prédictions : de la pâte feuilletée cuite au micro-ondes.
- Le juge Gemini (cohérence culinaire, sécurité alimentaire) n'a pas pu tourner : voir le problème 6.

## Chronologie

### 5 novembre : préparation

- Code initial du flow, sur le modèle des sessions 1 à 5 du cours.
- Le bucket est en région `EU`, donc `REGION=europe-west1` au départ.
- Test de génération : 50 recettes gardées sur 50.

### 5 et 10 novembre : dataset

- Passage à un dataset bilingue (français et anglais), inspiré de 28 cuisines, avec un parti pris créatif par recette.
- Test de 60 recettes : 60 gardées sur 60, 33 en français et 27 en anglais.
- Mesure du coût : 0,0065 $ par recette, dont la majorité vient de la « réflexion » de Gemini.
- Passage au réglage `thinking_level="LOW"` : 0,00175 $ par recette, soit 3,7 fois moins cher, sans perte visible sur l'exemple testé.
- Génération complète avec `--workers 16` : 3 000 gardées, 1 rejetée, 0 erreur. Coût estimé : environ 7 à 9 $ au total, tests compris.

### 11 novembre : pipeline

| Exécution | Région | Résultat | Cause |
| --- | --- | --- | --- |
| `recette-frigo-training-20261111102226` | `europe-west1` | Échec | Pas de quota GPU T4 pour l'entraînement dans cette région |
| `recette-frigo-training-20261111103531` | `europe-west2` | Échec | CUDA introuvable dans l'image Python minimale |
| `recette-frigo-training-20261111105036` | `europe-west2` | 7 étapes sur 8 | Fine-tuning et prédictions réussis, évaluation en échec |
| `recette-frigo-training-20261111205130` | `europe-west2` | Annulée | Lancée par erreur en recollant une ancienne commande |

Le fine-tuning de l'exécution `20261111105036` a duré environ 10 heures, soit 3 h 25 par passage sur un GPU T4. Coût estimé : environ 9 $.

### 12 novembre : évaluation, enregistrement, interface

- L'évaluation a été refaite en local avec `scripts/evaluate_local.py`, à partir des prédictions téléchargées depuis le bucket, puis enregistrée dans Vertex AI Experiments.
- Une relance du pipeline sans le juge (`recette-frigo-training-20261112090119`) a été annulée : le cache ne réutilisait pas les étapes, ce qui aurait refait les 10 heures de GPU.
- Modèle enregistré dans le Model Registry avec `scripts/deploy.py register`.
- Nouvelle interface Chainlit : thème « porte de frigo », choix des appareils, du temps et de la cuisine par boutons, frigos d'exemple.
- Test de déploiement lancé avec `scripts/deploy.py deploy`.

## Problèmes rencontrés et changements faits

1. **Modèle Gemini retiré.** `gemini-2.5-flash`, utilisé dans les slides, a été retiré de Vertex AI le 20 octobre 2026. Remplacé par `gemini-3.8-flash`.
2. **Coût de la génération.** La « réflexion » de Gemini représentait l'essentiel du coût. Le réglage `thinking_level="LOW"` l'a divisé par 3,7. Le réglage `MINIMAL` n'est pas accepté par ce modèle.
3. **Quota GPU.** Aucun quota T4 pour l'entraînement en `europe-west1`. Passage en `europe-west2`, et prédictions exécutées l'une après l'autre pour n'utiliser qu'un GPU à la fois.
4. **CUDA introuvable.** L'image `python:3.11-slim` ne trouve pas le pilote GPU de la machine. Remplacée par `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime` pour les étapes GPU.
5. **Fine-tuning plus long que prévu.** Environ 10 heures au lieu des 1 à 3 heures estimées. Une prochaine exécution devrait utiliser un seul passage, ce qui divise le temps et le coût par trois.
6. **Évaluation en échec dans le cloud.** Les logs étaient vides, donc la cause exacte n'est pas confirmée. Le juge Gemini, jamais testé en conditions réelles, est le suspect principal. Les autres métriques ont été calculées en local.
7. **Projet partagé.** D'autres groupes utilisent le même projet : les commandes de logs doivent filtrer sur l'identifiant de notre job, sinon elles mélangent les entraînements.
8. **Bucket hors projet.** Vertex AI signale que `ebcrjg-llmops` n'appartient pas au projet. Sans conséquence : le pipeline y lit et y écrit normalement.

## Coûts estimés

| Poste | Estimation |
| --- | --- |
| Dataset et tests de génération | 7 à 9 $ |
| Pipelines, dont le fine-tuning de 10 h | 9 à 11 $ |
| Évaluation locale | 0 $ |
| Test de déploiement | environ 1 $ par heure d'endpoint |
| **Total au 12 novembre** | **environ 18 à 22 $** |

## Rappel

Un endpoint est facturé environ 0,84 $ par heure, même sans trafic. Supprimer l'endpoint après chaque test :

```bash
uv run python scripts/deploy.py delete --endpoint-id 2352191822370963456
```

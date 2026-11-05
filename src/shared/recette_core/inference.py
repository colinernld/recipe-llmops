"""Chargement et génération, utilisés par l'étape Predict et par le handler Vertex AI."""

from __future__ import annotations

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def stop_token_ids(tokenizer) -> list[int]:
    """Phi-3 termine sur <|end|>, pas seulement sur le jeton de fin du tokenizer."""
    ids = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|end|>")}
    return sorted(i for i in ids if i is not None and i != tokenizer.unk_token_id)


def load_for_inference(base_model: str, adapter_dir: str | None = None):
    """Charge le modèle de base. Avec un adaptateur, fusionne BA dans les poids."""
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir or base_model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    gpu = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if gpu else torch.float32,
        device_map={"": 0} if gpu else None,
        attn_implementation="eager",
    )
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir).merge_and_unload()
    return model.eval(), tokenizer


@torch.inference_mode()
def generate(model, tokenizer, prompts: list[str], **parameters) -> list[str]:
    """Renvoie uniquement le texte nouveau, sans le prompt ni les jetons spéciaux."""
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
        return_token_type_ids=False,
    ).to(model.device)
    temperature = parameters.pop("temperature", 0.8)
    settings = {
        "max_new_tokens": 450,
        "do_sample": temperature > 0,
        "top_p": 0.9,
        **parameters,
    }
    if settings["do_sample"]:
        settings["temperature"] = temperature
    else:
        settings.pop("top_p", None)
    output = model.generate(
        **encoded,
        eos_token_id=stop_token_ids(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        **settings,
    )
    new_tokens = output[:, encoded["input_ids"].shape[1] :]
    return [text.strip() for text in tokenizer.batch_decode(new_tokens, skip_special_tokens=True)]


def predict_rows(
    rows: list[dict], base_model: str, adapter_dir: str | None, batch_size: int = 4
) -> list[dict]:
    model, tokenizer = load_for_inference(base_model, adapter_dir)
    predictions: list[dict] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        # Température nulle : deux runs d'évaluation restent comparables.
        outputs = generate(model, tokenizer, [row["prompt"] for row in batch], temperature=0.0)
        predictions += [{**row, "prediction": text} for row, text in zip(batch, outputs)]
    return predictions

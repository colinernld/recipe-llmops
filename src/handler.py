"""Handler du conteneur Hugging Face pour Vertex AI.

Ce fichier est copié à la racine des artefacts du modèle par scripts/deploy.py.
Il doit rester autonome : le conteneur de serving ne connaît pas le reste du projet.
"""

from __future__ import annotations

import os
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class EndpointHandler:
    def __init__(self, model_dir: str = "/opt/huggingface/model"):
        base_model = os.getenv("BASE_MODEL", "microsoft/Phi-3-mini-4k-instruct")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        gpu = torch.cuda.is_available()
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if gpu else torch.float32,
            device_map={"": 0} if gpu else None,
            attn_implementation="eager",
        )
        self.model = PeftModel.from_pretrained(model, model_dir).merge_and_unload().eval()
        stop_ids = {
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|end|>"),
        }
        self.stop_ids = sorted(
            i for i in stop_ids if i is not None and i != self.tokenizer.unk_token_id
        )

    @torch.inference_mode()
    def __call__(self, data: dict[str, Any]) -> dict[str, list[Any]]:
        prompts = [instance["input"] for instance in data["instances"]]
        parameters = dict(data.get("parameters") or {})
        temperature = float(parameters.pop("temperature", 0.8))
        settings: dict[str, Any] = {
            "max_new_tokens": 450,
            "max_time": 60.0,
            "do_sample": temperature > 0,
            **parameters,
        }
        if settings["do_sample"]:
            settings["temperature"] = temperature
            settings.setdefault("top_p", 0.9)

        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
            return_token_type_ids=False,
        ).to(self.model.device)
        output = self.model.generate(
            **encoded,
            eos_token_id=self.stop_ids,
            pad_token_id=self.tokenizer.pad_token_id,
            **settings,
        )
        new_tokens = output[:, encoded["input_ids"].shape[1] :]
        texts = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        return {"predictions": [text.strip() for text in texts]}

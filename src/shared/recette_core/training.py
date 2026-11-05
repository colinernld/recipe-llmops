"""Fine-tuning LoRA de Phi-3. Seuls les jetons de la réponse comptent dans la perte."""

from __future__ import annotations

import tempfile
from dataclasses import asdict, dataclass

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# Couches linéaires de Phi-3 dans transformers 4.48 : attention et MLP.
PHI3_TARGET_MODULES = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
IGNORE_INDEX = -100


@dataclass
class TrainingConfig:
    base_model: str = "microsoft/Phi-3-mini-4k-instruct"
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 4
    max_length: int = 768
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    quantize: bool = True
    gradient_checkpointing: bool = True


def encode(row: dict, tokenizer, max_length: int) -> dict:
    prompt_ids = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(row["completion"], add_special_tokens=False)["input_ids"]
    input_ids = (prompt_ids + answer_ids)[:max_length]
    labels = ([IGNORE_INDEX] * len(prompt_ids) + answer_ids)[:max_length]
    return {"input_ids": input_ids, "labels": labels}


class Collator:
    """Complète à droite, sans jamais compter le remplissage dans la perte."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        longest = max(len(item["input_ids"]) for item in batch)
        input_ids, labels, attention_mask = [], [], []
        for item in batch:
            padding = longest - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.pad_token_id] * padding)
            labels.append(item["labels"] + [IGNORE_INDEX] * padding)
            attention_mask.append([1] * len(item["input_ids"]) + [0] * padding)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attention_mask),
        }


def load_base_model(config: TrainingConfig):
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs: dict = {"attn_implementation": "eager"}
    if config.quantize:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(config.base_model, **kwargs)
    model.config.use_cache = False
    return model, tokenizer


def fine_tune(
    train_rows: list[dict], val_rows: list[dict], config: TrainingConfig, output_dir: str
) -> dict[str, float]:
    """Entraîne, sauvegarde l'adaptateur LoRA et renvoie les métriques à journaliser."""
    model, tokenizer = load_base_model(config)
    if config.quantize:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=config.gradient_checkpointing
        )
    elif config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=PHI3_TARGET_MODULES,
            task_type="CAUSAL_LM",
            bias="none",
        ),
    )

    train_data = [encode(row, tokenizer, config.max_length) for row in train_rows]
    val_data = [encode(row, tokenizer, config.max_length) for row in val_rows]
    use_gpu = torch.cuda.is_available()
    arguments = TrainingArguments(
        # Hors du bucket : plus rapide, et aucun checkpoint dans les artefacts du modèle.
        output_dir=tempfile.mkdtemp(prefix="checkpoints-"),
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=10,
        fp16=use_gpu and not config.quantize,
        remove_unused_columns=False,
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=[],
        use_cpu=not use_gpu,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=Collator(tokenizer.pad_token_id),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    result = trainer.train()
    final = trainer.evaluate()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics = {
        "train_loss": float(result.training_loss),
        "val_loss": float(final["eval_loss"]),
        "trainable_parameters": float(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        ),
    }
    for entry in trainer.state.log_history:
        if "eval_loss" in entry and "epoch" in entry:
            metrics[f"val_loss_epoch_{entry['epoch']:g}"] = float(entry["eval_loss"])
    return {**metrics, **_numeric(asdict(config))}


def _numeric(values: dict) -> dict[str, float]:
    return {k: float(v) for k, v in values.items() if isinstance(v, (int, float))}

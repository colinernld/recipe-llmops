"""Cycle complet sur un minuscule Phi-3 aléatoire, sans GPU ni téléchargement.

Ce test vérifie la mécanique (masque de perte, LoRA, sauvegarde, handler), pas la
qualité des recettes. Il demande torch, transformers et peft, absents de l'environnement
de l'application : il est ignoré s'ils ne sont pas installés.
"""

import importlib.util
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402
from transformers import Phi3Config, Phi3ForCausalLM, PreTrainedTokenizerFast  # noqa: E402

from recette_core.data import format_rows, read_jsonl  # noqa: E402
from recette_core.inference import generate, load_for_inference, stop_token_ids  # noqa: E402
from recette_core.training import IGNORE_INDEX, Collator, TrainingConfig, encode, fine_tune  # noqa: E402

SPECIALS = ["<unk>", "<|endoftext|>", "<|end|>", "<|system|>", "<|user|>", "<|assistant|>"]


@pytest.fixture(scope="module")
def rows():
    return format_rows(read_jsonl("data/sample_recipes.jsonl"))


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory, rows):
    path = tmp_path_factory.mktemp("tiny")
    words = sorted({w for row in rows for w in (row["prompt"] + row["completion"]).split()})
    vocab = {token: i for i, token in enumerate(dict.fromkeys([*SPECIALS, *words]))}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        eos_token="<|endoftext|>",
        additional_special_tokens=SPECIALS[2:],
    )
    tokenizer.save_pretrained(path)
    config = Phi3Config(
        vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=512,
        pad_token_id=1, bos_token_id=1, eos_token_id=1,
    )
    Phi3ForCausalLM(config).save_pretrained(path)
    return str(path)


def test_loss_only_covers_the_answer(tiny_model, rows):
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tiny_model)
    item = encode(rows[0], tokenizer, 512)
    n_prompt = len(tokenizer(rows[0]["prompt"], add_special_tokens=False)["input_ids"])
    assert set(item["labels"][:n_prompt]) == {IGNORE_INDEX}
    assert IGNORE_INDEX not in item["labels"][n_prompt:]
    batch = Collator(1)([item, {"input_ids": [5], "labels": [5]}])
    assert batch["labels"][1, 1:].eq(IGNORE_INDEX).all()
    assert batch["attention_mask"][1].sum() == 1


def test_train_save_load_and_serve(tiny_model, rows, tmp_path, monkeypatch):
    config = TrainingConfig(
        base_model=tiny_model, num_epochs=2, batch_size=2, max_length=256,
        quantize=False, learning_rate=1e-3,
    )
    adapter = str(tmp_path / "adapter")
    metrics = fine_tune(rows[:8], rows[8:10], config, adapter)
    assert metrics["trainable_parameters"] > 0
    assert "val_loss_epoch_1" in metrics and "val_loss_epoch_2" in metrics
    assert json.load(open(f"{adapter}/adapter_config.json"))["r"] == 16

    model, tokenizer = load_for_inference(tiny_model, adapter)
    assert len(stop_token_ids(tokenizer)) == 2
    assert not any("lora" in name for name, _ in model.named_parameters())
    assert isinstance(generate(model, tokenizer, [rows[0]["prompt"]], max_new_tokens=5)[0], str)

    spec = importlib.util.spec_from_file_location("handler", "src/handler.py")
    handler_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(handler_module)
    monkeypatch.setenv("BASE_MODEL", tiny_model)
    handler = handler_module.EndpointHandler(adapter)
    output = handler(
        {"instances": [{"input": rows[0]["prompt"]}], "parameters": {"max_new_tokens": 5}}
    )
    assert list(output) == ["predictions"] and len(output["predictions"]) == 1

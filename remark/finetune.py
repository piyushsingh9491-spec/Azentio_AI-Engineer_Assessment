"""Compact character-level transformer trained from scratch on CPU.

    python -m remark.finetune [--data data/] [--models models/] [--reports reports/]

Architecture: Hugging Face BertForSequenceClassification with a small random
config (2 layers, hidden 128) over a character vocabulary built from train.csv.
Character input handles typos, truncation and unseen handles without any
downloaded tokenizer, so training and tests are fully offline.

Split usage: weights are fitted on train.csv; val.csv is used for early stopping
(best val macro-F1), temperature and threshold; test.csv is only evaluated.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import BertConfig, BertForSequenceClassification

from remark.data import label_ids, load_split
from remark.evaluate import (
    REFERENCE_FILE, build_reference, calibrate_on_val, calibration_on_test, data_fingerprint,
    evaluate_on_test, write_json,
)
from remark.labels import LABELS
from remark.metrics import macro_f1
from remark.models import META_FILE, Predictor
from remark.text import normalize

PAD, UNK, CLS = 0, 1, 2
SPECIAL = ["[PAD]", "[UNK]", "[CLS]"]
MAX_LEN = 96  # normalised narrations: p99 65 chars, max 72 on train
HPARAMS = {"hidden_size": 128, "num_hidden_layers": 2, "num_attention_heads": 4,
           "intermediate_size": 256, "lr": 1e-3, "batch_size": 32, "max_epochs": 30,
           "patience": 5, "weight_decay": 0.01, "seed": 0}


class CharTokenizer:
    def __init__(self, vocab: list[str]):
        self.vocab = vocab
        self.index = {ch: i for i, ch in enumerate(vocab)}

    @classmethod
    def fit(cls, texts: list[str]) -> "CharTokenizer":
        chars = sorted({ch for text in texts for ch in normalize(text)})
        return cls(SPECIAL + chars)

    def encode(self, texts: list[str], max_len: int = MAX_LEN) -> tuple[torch.Tensor, torch.Tensor]:
        ids = np.full((len(texts), max_len), PAD, dtype=np.int64)
        for row, text in enumerate(texts):
            # Keep the END of long narrations: the free-text remark is the last field.
            chars = normalize(text)[-(max_len - 1):]
            ids[row, 0] = CLS
            ids[row, 1:len(chars) + 1] = [self.index.get(ch, UNK) for ch in chars]
        input_ids = torch.from_numpy(ids)
        return input_ids, (input_ids != PAD).long()


def build_model(vocab_size: int, **overrides) -> BertForSequenceClassification:
    h = {**HPARAMS, **overrides}
    config = BertConfig(
        vocab_size=vocab_size, hidden_size=h["hidden_size"],
        num_hidden_layers=h["num_hidden_layers"], num_attention_heads=h["num_attention_heads"],
        intermediate_size=h["intermediate_size"], max_position_embeddings=MAX_LEN,
        num_labels=len(LABELS), pad_token_id=PAD,
        id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)},
    )
    return BertForSequenceClassification(config)


def class_weights(y: np.ndarray) -> torch.Tensor:
    counts = np.bincount(y, minlength=len(LABELS)).astype(np.float32)
    return torch.tensor(len(y) / (len(LABELS) * np.maximum(counts, 1)))


def train_step(model, optimizer, input_ids, attention_mask, labels, weights=None) -> float:
    model.train()
    optimizer.zero_grad()
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    loss = torch.nn.functional.cross_entropy(logits, labels, weight=weights)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()


class FinetunedPredictor(Predictor):
    name = "finetuned"

    def __init__(self, model: BertForSequenceClassification, tokenizer: CharTokenizer,
                 meta: dict | None = None):
        super().__init__(meta)
        self.model = model.eval()
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def raw_logits(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            input_ids, mask = self.tokenizer.encode(texts[i:i + batch_size])
            out.append(self.model(input_ids=input_ids, attention_mask=mask).logits.numpy())
        return np.concatenate(out) if out else np.zeros((0, len(LABELS)))

    def save(self, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(model_dir)
        (model_dir / "vocab.json").write_text(json.dumps(self.tokenizer.vocab))
        self.save_meta(model_dir)

    @classmethod
    def load(cls, model_dir: Path) -> "FinetunedPredictor":
        model = BertForSequenceClassification.from_pretrained(model_dir)
        tokenizer = CharTokenizer(json.loads((model_dir / "vocab.json").read_text()))
        return cls(model, tokenizer, json.loads((model_dir / META_FILE).read_text()))


def fit_finetuned(train: pd.DataFrame, val: pd.DataFrame, log=print) -> tuple[FinetunedPredictor, dict]:
    h = HPARAMS
    torch.manual_seed(h["seed"])
    rng = np.random.default_rng(h["seed"])
    tokenizer = CharTokenizer.fit(train["narration"].tolist())
    model = build_model(len(tokenizer.vocab))
    optimizer = torch.optim.AdamW(model.parameters(), lr=h["lr"], weight_decay=h["weight_decay"])
    x_ids, x_mask = tokenizer.encode(train["narration"].tolist())
    y = torch.from_numpy(label_ids(train).copy())
    weights = class_weights(label_ids(train))
    predictor = FinetunedPredictor(model, tokenizer)
    y_val = label_ids(val)

    best = {"f1": -1.0, "epoch": 0, "state": None}
    history = []
    for epoch in range(1, h["max_epochs"] + 1):
        order = rng.permutation(len(y))
        losses = [train_step(model, optimizer, x_ids[b], x_mask[b], y[b], weights)
                  for b in (order[i:i + h["batch_size"]] for i in range(0, len(y), h["batch_size"]))]
        val_f1 = macro_f1(y_val, predictor.raw_logits(val["narration"].tolist()).argmax(axis=1))
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_macro_f1": val_f1})
        log(f"epoch {epoch:2d} loss {np.mean(losses):.4f} val macro-F1 {val_f1:.4f}")
        if val_f1 > best["f1"]:
            best = {"f1": val_f1, "epoch": epoch,
                    "state": {k: v.clone() for k, v in model.state_dict().items()}}
        elif epoch - best["epoch"] >= h["patience"]:
            log(f"early stop: best val macro-F1 {best['f1']:.4f} at epoch {best['epoch']}")
            break
    model.load_state_dict(best["state"])
    model.eval()
    return predictor, {"best_epoch": best["epoch"], "best_val_macro_f1": best["f1"],
                       "history": history}


def parameter_counts(model: torch.nn.Module) -> dict:
    return {"total": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Train the compact char transformer")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    args = parser.parse_args(argv)

    train, val, test = (load_split(args.data, s) for s in ("train", "val", "test"))
    start = time.perf_counter()
    predictor, training = fit_finetuned(train, val)
    train_seconds = time.perf_counter() - start

    predictor.meta = {"version": f"finetuned-{data_fingerprint(args.data / 'train.csv')}",
                      "checkpoint": "none (trained from scratch)", "hparams": HPARAMS}
    calibrate_on_val(predictor, val)  # val only: temperature + alert threshold
    model_dir = args.models / "finetuned"
    predictor.save(model_dir)
    write_json(build_reference(predictor, train), model_dir / REFERENCE_FILE)

    report = {"model": "finetuned", "version": predictor.version,
              "checkpoint": "none (trained from scratch)",
              "train_seconds": round(train_seconds, 1), "torch_threads": torch.get_num_threads(),
              "parameters": parameter_counts(predictor.model),
              "training": {k: v for k, v in training.items() if k != "history"},
              "test": evaluate_on_test(predictor, test, [train, val]),
              "calibration": calibration_on_test(predictor, test)}
    write_json(report, args.reports / "finetuned.json")
    t = report["test"]
    print(f"finetuned test macro-F1 {t['macro_f1']:.4f} | held-out templates "
          f"{t['holdout_templates']['macro_f1']:.4f} | trained in {train_seconds:.0f}s "
          f"({report['parameters']['total']:,} params)")
    return report


if __name__ == "__main__":
    main()

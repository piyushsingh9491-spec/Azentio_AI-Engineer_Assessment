"""Deterministic synthetic narration generator.

    python -m remark.generate --seed 42 --out data/
    python -m remark.generate --seed 7 --drift --out data/drift/

Split design (cannot be passed by memorisation):
  * train uses a "seen" pool of templates and payee handles;
  * ~30% of each class's val rows use templates/handles unseen in train, so
    calibration and threshold choice see novel phrasings too;
  * ~30% of each class's test rows use templates AND handles/entities that
    never appear in train or val (template_id is disjoint, so a reviewer can
    recompute the held-out mask from the CSVs alone);
  * narrations are globally de-duplicated, so no exact string is in two splits.

All randomness comes from one random.Random(seed); output is byte-identical
for the same seed.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from remark.labels import LABELS, MAX_NARRATION_LEN
from remark.templates import (
    ABBREVIATIONS, BANK_CODES, DRIFT_HANDLES, DRIFT_TEMPLATES, MONTHS, NAMES,
    PERSONAL_HANDLES_HOLDOUT, PERSONAL_HANDLES_SEEN, SPECS, ClassSpec,
)

COLUMNS = ["txn_id", "channel", "amount", "narration", "label", "template_id"]

# Rows per label per split. BENIGN share: train 72.4%, val 72.5%, test 71.4%.
SPLIT_COUNTS: dict[str, dict[str, int]] = {
    "train": {"BENIGN": 1050, **{label: 80 for label in LABELS[1:]}},
    "val": {"BENIGN": 330, **{label: 25 for label in LABELS[1:]}},
    "test": {"BENIGN": 500, **{label: 40 for label in LABELS[1:]}},
}
NOVEL_FRACTION = 0.30  # share of val/test rows from templates unseen in earlier splits
CHANNEL_WEIGHTS = {"UPI": 0.70, "IMPS": 0.18, "NEFT": 0.12}
DRIFT_CHANNEL_WEIGHTS = {"UPI": 0.25, "IMPS": 0.60, "NEFT": 0.15}
DRIFT_ROWS = 300
PERSONAL_HANDLE_RATE = 0.3  # risky payments sent to an ordinary-looking personal handle


def template_ids(spec: ClassSpec) -> tuple[list[str], list[str]]:
    """Stable ids: seen templates first, holdout templates numbered after them."""
    n_seen = len(spec.seen_templates)
    seen = [f"{spec.prefix}_{i:02d}" for i in range(n_seen)]
    holdout = [f"{spec.prefix}_{n_seen + i:02d}" for i in range(len(spec.holdout_templates))]
    return seen, holdout


# --------------------------------------------------------------------------- noise


def _typo(word: str, rng: random.Random) -> str:
    if len(word) < 4 or not word.isalpha():
        return word
    i = rng.randrange(1, len(word) - 1)
    op = rng.choice(("drop", "swap", "double"))
    if op == "drop":
        return word[:i] + word[i + 1:]
    if op == "swap":
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    return word[:i] + word[i] + word[i:]


def add_noise(remark: str, rng: random.Random) -> str:
    words = remark.split()
    words = [ABBREVIATIONS.get(w, w) if rng.random() < 0.5 else w for w in words]
    if words and rng.random() < 0.25:
        j = rng.randrange(len(words))
        words[j] = _typo(words[j], rng)
    out = " ".join(words)
    style = rng.random()
    if style < 0.35:
        out = out.upper()
    elif style < 0.45:
        out = out.title()
    if rng.random() < 0.3:
        out = out[: rng.randint(14, 28)].rstrip()  # bank remark-field truncation
    return out


# ----------------------------------------------------------------------- formatting


def _digits(rng: random.Random, n: int) -> str:
    return "".join(str(rng.randrange(10)) for _ in range(n))


def fill(template: str, spec: ClassSpec, rng: random.Random) -> str:
    return template.format(
        app=rng.choice(spec.apps),
        n=rng.choice(NAMES),
        a=rng.choice((500, 1000, 2000, 5000, 10000)),
        m=rng.choice(MONTHS),
        d=rng.randint(1, 9999),
    )


def wrap(channel: str, remark: str, handle: str, entity: str, rng: random.Random) -> str:
    rrn = _digits(rng, 12)
    bank = rng.choice(BANK_CODES)
    if channel == "UPI":
        payee = entity.split()[0].title() if entity else handle.split("@")[0]
        fmt = rng.randrange(3)
        if fmt == 0:
            text = f"UPI/{rrn}/{handle}/{remark}"
        elif fmt == 1:
            text = f"UPI-{payee.upper()}-{handle}-{bank}0{_digits(rng, 6)}-{rrn}-{remark}"
        else:
            text = f"UPI/DR/{rrn}/{payee}/{bank}/{handle}/{remark}"
    elif channel == "NEFT":
        utr = f"{bank}N{_digits(rng, 11)}"
        text = rng.choice((f"NEFT/{utr}/{entity}/{remark}", f"NEFT-{utr}-{entity}-{remark}"))
    else:
        text = rng.choice((f"IMPS/P2A/{rrn}/{entity}/{remark}", f"MMT/IMPS/{rrn}/{remark}/{entity}"))
    return text[:MAX_NARRATION_LEN]


def _channel(rng: random.Random, weights: dict[str, float]) -> str:
    return rng.choices(list(weights), weights=list(weights.values()))[0]


# Template pools. "val_only" templates/handles are unseen in train, so val
# contains new phrasings too and is an honest proxy for calibration and
# threshold choice; "test_holdout" ones are unseen in train AND val.
VAL_ONLY_TEMPLATES = 3
VAL_ONLY_HANDLES = 2
POOLS = ("train", "val_only", "test_holdout")


def pool_material(spec: ClassSpec, pool: str) -> tuple[list[str], list[str], list[str], list[str]]:
    """(templates, template_ids, handles, entities) for one pool."""
    seen_ids, holdout_ids = template_ids(spec)
    if pool == "test_holdout":
        return spec.holdout_templates, holdout_ids, spec.holdout_handles, spec.holdout_entities
    cut_t, cut_h = -VAL_ONLY_TEMPLATES, -VAL_ONLY_HANDLES
    if pool == "val_only":
        return (spec.seen_templates[cut_t:], seen_ids[cut_t:], spec.seen_handles[cut_h:],
                spec.seen_entities)
    return spec.seen_templates[:cut_t], seen_ids[:cut_t], spec.seen_handles[:cut_h], spec.seen_entities


def make_row(label: str, pool: str, rng: random.Random, seen: set[str],
             weights: dict[str, float]) -> dict:
    spec = SPECS[label]
    templates, ids, handles, entities = pool_material(spec, pool)
    if label != "BENIGN" and rng.random() < PERSONAL_HANDLE_RATE:
        handles = PERSONAL_HANDLES_HOLDOUT if pool == "test_holdout" else PERSONAL_HANDLES_SEEN
    while True:  # re-draw until the narration is globally unique
        k = rng.randrange(len(templates))
        channel = _channel(rng, weights)
        remark = add_noise(fill(templates[k], spec, rng), rng)
        narration = wrap(channel, remark, rng.choice(handles), rng.choice(entities), rng)
        if narration not in seen:
            seen.add(narration)
            break
    lo, hi = spec.amount_range
    amount = round(min(hi, max(lo, rng.lognormvariate(0, 1) * (lo + hi) / 6)), 2)
    return {"channel": channel, "amount": f"{amount:.2f}", "narration": narration,
            "label": label, "template_id": ids[k]}


def generate_splits(seed: int) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    seen: set[str] = set()
    splits: dict[str, list[dict]] = {}
    novel_pool = {"train": None, "val": "val_only", "test": "test_holdout"}
    for split, counts in SPLIT_COUNTS.items():
        rows = []
        for label in LABELS:
            n = counts[label]
            n_novel = round(n * NOVEL_FRACTION) if novel_pool[split] else 0
            rows += [make_row(label, novel_pool[split] if i < n_novel else "train", rng, seen,
                              CHANNEL_WEIGHTS) for i in range(n)]
        rng.shuffle(rows)
        for i, row in enumerate(rows):
            row["txn_id"] = f"{split[:2].upper()}{seed:03d}{i:06d}"
        splits[split] = rows
    return splits


def generate_drift(seed: int) -> list[dict]:
    """A shifted batch: new scam family + new PSP handle suffixes + IMPS-heavy mix."""
    rng = random.Random(seed)
    seen: set[str] = set()
    rows = []
    spec = SPECS["IMPERSONATION_SCAM"]
    for i in range(DRIFT_ROWS):
        if rng.random() < 0.6:
            channel = _channel(rng, DRIFT_CHANNEL_WEIGHTS)
            while True:
                k = rng.randrange(len(DRIFT_TEMPLATES))
                remark = add_noise(fill(DRIFT_TEMPLATES[k], spec, rng), rng)
                narration = wrap(channel, remark, rng.choice(DRIFT_HANDLES), "NODAL OFFICER CELL", rng)
                if narration not in seen:
                    seen.add(narration)
                    break
            row = {"channel": channel, "amount": f"{rng.uniform(5000, 90000):.2f}",
                   "narration": narration, "label": "IMPERSONATION_SCAM",
                   "template_id": f"DRF_{k:02d}"}
        else:
            row = make_row("BENIGN", "train", rng, seen, DRIFT_CHANNEL_WEIGHTS)
        row["txn_id"] = f"DR{seed:03d}{i:06d}"
        rows.append(row)
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--drift", action="store_true", help="write a single shifted batch")
    args = parser.parse_args(argv)
    if args.drift:
        write_csv(generate_drift(args.seed), args.out / "drift.csv")
        print(f"wrote {DRIFT_ROWS} drifted rows to {args.out / 'drift.csv'}")
        return
    for split, rows in generate_splits(args.seed).items():
        write_csv(rows, args.out / f"{split}.csv")
        print(f"{split}: {len(rows)} rows -> {args.out / f'{split}.csv'}")


if __name__ == "__main__":
    main()

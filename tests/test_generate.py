import pandas as pd
import pytest

from remark.data import holdout_mask
from remark.generate import COLUMNS, main
from remark.labels import CHANNELS, LABELS, MAX_NARRATION_LEN, RISKY_LABELS


@pytest.fixture(scope="module")
def splits(tmp_path_factory):
    out = tmp_path_factory.mktemp("data")
    main(["--seed", "42", "--out", str(out)])
    return out, {s: pd.read_csv(out / f"{s}.csv") for s in ("train", "val", "test")}


def test_same_seed_gives_identical_files(splits, tmp_path):
    out, _ = splits
    main(["--seed", "42", "--out", str(tmp_path)])
    for s in ("train", "val", "test"):
        assert (out / f"{s}.csv").read_bytes() == (tmp_path / f"{s}.csv").read_bytes()


def test_different_seed_differs(splits, tmp_path):
    out, _ = splits
    main(["--seed", "43", "--out", str(tmp_path)])
    assert (out / "train.csv").read_bytes() != (tmp_path / "train.csv").read_bytes()


def test_schema_sizes_and_class_balance(splits):
    _, d = splits
    assert len(d["train"]) >= 1200 and len(d["val"]) >= 400 and len(d["test"]) >= 600
    for df in d.values():
        assert list(df.columns) == COLUMNS
        assert set(df.label) == set(LABELS)
        assert set(df.channel) <= set(CHANNELS)
        assert df.narration.str.len().max() <= MAX_NARRATION_LEN
        assert df.txn_id.is_unique
    everything = pd.concat(d.values())
    assert 0.70 <= (everything.label == "BENIGN").mean() <= 0.85
    counts = d["test"].label.value_counts()
    assert all(counts[label] >= 30 for label in RISKY_LABELS)


def test_no_exact_narration_shared_between_splits(splits):
    _, d = splits
    train, val, test = (set(d[s].narration) for s in ("train", "val", "test"))
    assert not train & test and not val & test and not train & val


def test_template_holdout_rule(splits):
    _, d = splits
    mask = holdout_mask(d["test"], [d["train"], d["val"]])
    assert mask.mean() >= 0.20
    # Held-out rows must also use payee handles never seen in train/val.
    seen_text = " ".join(pd.concat([d["train"], d["val"]]).narration)
    handles = d["test"][mask].narration.str.extract(r"([\w.]+@\w+)")[0].dropna().unique()
    assert len(handles) > 0 and not any(h in seen_text for h in handles)


def test_drift_batch_is_deterministic(tmp_path):
    main(["--seed", "7", "--drift", "--out", str(tmp_path / "a")])
    main(["--seed", "7", "--drift", "--out", str(tmp_path / "b")])
    a, b = (tmp_path / x / "drift.csv" for x in "ab")
    assert a.read_bytes() == b.read_bytes()
    assert list(pd.read_csv(a).columns) == COLUMNS

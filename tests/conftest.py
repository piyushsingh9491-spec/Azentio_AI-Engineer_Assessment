"""Shared fixtures: everything is generated and trained in tmp dirs, fully offline."""

import pytest
from fastapi.testclient import TestClient

from remark import train_baseline
from remark.api import Settings, create_app
from remark.data import load_split
from remark.generate import main as generate


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("data")
    generate(["--seed", "42", "--out", str(out)])
    return out


@pytest.fixture(scope="session")
def models_dir(data_dir, tmp_path_factory):
    models = tmp_path_factory.mktemp("models")
    reports = tmp_path_factory.mktemp("reports")
    train_baseline.main(["--data", str(data_dir), "--models", str(models), "--reports", str(reports)])
    return models


@pytest.fixture(scope="session")
def splits(data_dir):
    return {s: load_split(data_dir, s) for s in ("train", "val", "test")}


@pytest.fixture()
def client(data_dir, models_dir, tmp_path):
    settings = Settings(model="baseline", models_dir=models_dir, data_dir=data_dir,
                        labels_db=tmp_path / "labels.db")
    with TestClient(create_app(settings)) as c:
        yield c

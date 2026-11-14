import pytest

from remark.data import load_split
from remark.generate import main as generate


def test_review_queue_end_to_end(client):
    first = client.get("/v1/review-queue", params={"k": 20}).json()["items"]
    assert len(first) == 20
    scores = [item["uncertainty"] for item in first]
    assert scores == sorted(scores, reverse=True)
    labelled = [item["txn_id"] for item in first[:5]]
    for txn_id in labelled:
        r = client.post("/v1/labels", json={"txn_id": txn_id, "label": "BENIGN",
                                            "analyst_id": "analyst_01"})
        assert r.status_code == 201 and r.json()["created_at"]
    second = client.get("/v1/review-queue", params={"k": 20}).json()["items"]
    second_ids = {item["txn_id"] for item in second}
    assert len(second) == 20 and not second_ids & set(labelled)
    assert [i["txn_id"] for i in second[:15]] == [i["txn_id"] for i in first[5:]]


def test_review_queue_never_exposes_labels(client):
    item = client.get("/v1/review-queue", params={"k": 1}).json()["items"][0]
    assert "label" not in item


@pytest.mark.parametrize("body", [
    {"txn_id": "NOPE000001", "label": "BENIGN", "analyst_id": "a1"},
    {"txn_id": "TR042000000", "label": "FRAUD", "analyst_id": "a1"},
    {"txn_id": "TR042000000", "label": "BENIGN", "analyst_id": ""},
    {"txn_id": "TR042000000", "label": "BENIGN", "analyst_id": "a1; DROP TABLE labels"},
])
def test_bad_labels_return_422(client, body):
    assert client.post("/v1/labels", json=body).status_code == 422


def test_drift_detected_on_seeded_drift_batch(client, tmp_path):
    generate(["--seed", "7", "--drift", "--out", str(tmp_path)])
    narrations = load_split(tmp_path, "drift")["narration"].tolist()
    body = client.post("/v1/drift", json={"narrations": narrations}).json()
    assert body["drift"] is True
    assert body["psi"] > body["thresholds"]["psi"] or body["oov_rate"] > body["thresholds"]["oov_rate"]


def test_no_drift_on_held_back_in_distribution_batch(client, splits):
    narrations = splits["val"]["narration"].tolist()  # never used to fit the model
    body = client.post("/v1/drift", json={"narrations": narrations}).json()
    assert body["drift"] is False and set(body["thresholds"]) == {"psi", "oov_rate"}

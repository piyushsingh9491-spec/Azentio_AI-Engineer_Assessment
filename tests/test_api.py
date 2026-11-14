import pytest

from remark.labels import LABELS

VALID = {"narration": "UPI/123456789012/betking777@ybl/ipl satta bhai", "channel": "UPI",
         "amount": 500.0}


def test_health_reports_model_version(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_version"].startswith("baseline-")


def test_score_contract(client):
    r = client.post("/v1/score", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert set(body["probabilities"]) == set(LABELS)
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-4)
    assert body["risk_score"] == pytest.approx(1 - body["probabilities"]["BENIGN"], abs=1e-5)
    assert body["label"] in LABELS and body["model_version"].startswith("baseline-")
    assert 0 < body["threshold"] <= 1
    assert body["alert"] == (body["risk_score"] >= body["threshold"])


def test_obvious_cases_are_classified(client):
    assert client.post("/v1/score", json=VALID).json()["label"] == "GAMBLING_BETTING"
    benign = {**VALID, "narration": "UPI/123456789012/swiggy@icici/swiggy order 221"}
    assert client.post("/v1/score", json=benign).json()["label"] == "BENIGN"


@pytest.mark.parametrize("patch", [
    {"narration": ""},
    {"narration": "   "},
    {"narration": "x" * 201},
    {"channel": "RTGS"},
    {"channel": "upi"},
    {"amount": -1},
    {"narration": None},
])
def test_invalid_single_requests_return_422(client, patch):
    assert client.post("/v1/score", json={**VALID, **patch}).status_code == 422


def test_missing_field_and_extra_field_return_422(client):
    assert client.post("/v1/score", json={"narration": "rent", "channel": "UPI"}).status_code == 422
    assert client.post("/v1/score", json={**VALID, "debug": True}).status_code == 422


def test_batch_scores_in_order(client):
    items = [VALID, {**VALID, "narration": "NEFT/HDFCN12345678901/ACME TECH PVT LTD/salary nov"}]
    r = client.post("/v1/score/batch", json={"items": items})
    assert r.status_code == 200
    results = r.json()["results"]
    assert [x["label"] for x in results] == ["GAMBLING_BETTING", "BENIGN"]


def test_batch_limits_return_422(client):
    assert client.post("/v1/score/batch", json={"items": []}).status_code == 422
    assert client.post("/v1/score/batch", json={"items": [VALID] * 257}).status_code == 422
    assert client.post("/v1/score/batch", json={"items": [VALID] * 256}).status_code == 200
    bad = [VALID, {**VALID, "channel": "SWIFT"}]
    assert client.post("/v1/score/batch", json={"items": bad}).status_code == 422

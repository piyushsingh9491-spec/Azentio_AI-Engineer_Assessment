import numpy as np
import pytest

torch = pytest.importorskip("torch")

from remark.finetune import CharTokenizer, FinetunedPredictor, build_model, train_step  # noqa: E402
from remark.labels import LABELS  # noqa: E402

TEXTS = ["UPI/123456789012/betking777@ybl/ipl satta", "UPI/123456789012/swiggy@icici/order 22",
         "IMPS/P2A/123456789012/KYC SERVICES/kyc update fee", "usdt p2p order 51"]


def tiny_model(vocab_size):
    torch.manual_seed(0)
    return build_model(vocab_size, hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                       intermediate_size=64)


def test_tokenizer_keeps_end_of_long_narration_and_maps_unknown_chars():
    tok = CharTokenizer.fit(["abc"])
    ids, mask = tok.encode(["x" * 500 + "abc", "z"], max_len=8)
    assert ids.shape == (2, 8) and ids[0, 0] == 2  # [CLS]
    assert tok.vocab[int(ids[0, -1])] == "c"       # end of text kept
    assert int(ids[1, 1]) == 1 and int(mask[1].sum()) == 2  # [UNK], then padding


def test_training_steps_reduce_loss_on_fixed_batch():
    tok = CharTokenizer.fit(TEXTS)
    model = tiny_model(len(tok.vocab))
    ids, mask = tok.encode(TEXTS, max_len=64)
    labels = torch.tensor([1, 0, 4, 2])
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
    model.eval()
    with torch.no_grad():
        before = torch.nn.functional.cross_entropy(model(input_ids=ids, attention_mask=mask).logits,
                                                   labels).item()
    model.train()
    for _ in range(3):
        train_step(model, optimizer, ids, mask, labels)
    model.eval()
    with torch.no_grad():
        after = torch.nn.functional.cross_entropy(model(input_ids=ids, attention_mask=mask).logits,
                                                  labels).item()
    assert after < before


def test_predictor_roundtrip(tmp_path):
    tok = CharTokenizer.fit(TEXTS)
    p = FinetunedPredictor(tiny_model(len(tok.vocab)), tok, {"version": "finetuned-test",
                                                            "temperature": 2.0, "threshold": 0.5})
    p.save(tmp_path)
    loaded = FinetunedPredictor.load(tmp_path)
    np.testing.assert_allclose(loaded.raw_logits(TEXTS), p.raw_logits(TEXTS), atol=1e-5)
    probs = loaded.predict_proba(TEXTS)
    assert probs.shape == (4, len(LABELS)) and np.allclose(probs.sum(axis=1), 1.0)
    assert loaded.temperature == 2.0 and loaded.threshold == 0.5

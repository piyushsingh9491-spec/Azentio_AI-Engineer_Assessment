from remark.active_loop import labels_to_reach, run


def test_labels_to_reach():
    curve = [(60, 0.5), (90, 0.8), (120, 0.92)]
    assert labels_to_reach(curve, 0.9) == 120
    assert labels_to_reach(curve, 0.95) is None


def test_strategies_share_seed_set_and_grow_by_batch(splits):
    train, test = splits["train"], splits["test"]
    u = run("uncertainty", train, test, seed=0, seed_labels=60, batch=30, rounds=2)
    r = run("random", train, test, seed=0, seed_labels=60, batch=30, rounds=2)
    assert [n for n, _ in u] == [n for n, _ in r] == [60, 90, 120]
    assert u[0][1] == r[0][1]  # identical seed-label set -> identical round-0 score

"""Phase 5: the two-tower data pipeline must never leak the future, and the loss must match a
brute-force computation. Runs on CPU for determinism."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src import data_prep as dp
from src import evaluate as ev
from src import two_tower as tt

FIXTURES = Path(__file__).resolve().parent / "fixtures"
CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def phase1(spark, tmp_path_factory):
    out = tmp_path_factory.mktemp("phase1_tt")
    raw = out / "raw"
    raw.mkdir()
    (raw / "ratings.csv").write_text((FIXTURES / "tiny_ratings.csv").read_text())
    (raw / "movies.csv").write_text((FIXTURES / "tiny_movies.csv").read_text())
    dp.run(spark, raw, out, out / "stats.csv", with_genome=False)
    return out / "splits.parquet"


@pytest.fixture(scope="module")
def items():
    return ev.ItemIndex.from_movies_csv(FIXTURES / "tiny_movies.csv")


@pytest.fixture(scope="module")
def seq(phase1, items):
    return tt.load_sequences(phase1, items, ("train",))


@pytest.fixture(scope="module")
def splits(phase1):
    return pd.read_parquet(phase1)


def test_sequences_are_time_ordered_train_positives(seq, splits, items):
    pos = splits[(splits.split_user == "train") & (splits.rating >= 4)]
    pos = pos.sort_values(["userId", "timestamp", "movieId"])
    for u, g in pos.groupby("userId"):
        r = seq.rows([u])[0]
        got = seq.items[seq.offsets[r]:seq.offsets[r + 1]]
        assert list(got) == list(items.to_index(g.movieId.to_numpy())), u
    assert len(seq.items) == len(pos)
    assert set(seq.user_ids) == set(splits.userId)  # all users, even without positives


def test_pairs_histories_never_contain_the_future(seq):
    users, pos = tt.all_pairs(seq)
    assert len(users) == np.maximum(seq.lengths - 1, 0).sum()
    for L in (1, 3, 50):
        h, off = tt.history_bags(seq, users, pos, L)
        bounds = np.append(off, len(h))
        for i in range(len(users)):
            start = seq.offsets[users[i]]
            target = seq.items[start + pos[i]]
            hist = h[bounds[i]:bounds[i + 1]]
            expect = seq.items[start + max(0, pos[i] - L): start + pos[i]]  # strictly before t
            assert list(hist) == list(expect)
            assert 1 <= len(hist) <= L
            assert target not in hist


def test_epoch_sampling_caps_pairs_per_user(seq):
    users, pos = tt.all_pairs(seq)
    a = tt.sample_epoch(users, pos, 3, np.random.default_rng(0))
    b = tt.sample_epoch(users, pos, 3, np.random.default_rng(0))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    counts = np.bincount(a[0], minlength=len(seq.user_ids))
    full = np.bincount(users, minlength=len(seq.user_ids))
    assert (counts == np.minimum(full, 3)).all()
    allp = tt.sample_epoch(users, pos, None, np.random.default_rng(0))
    assert sorted(zip(*allp)) == sorted(zip(users, pos))


def test_loss_matches_brute_force_with_logq_dup_mask_and_uniform_negatives():
    g = torch.Generator().manual_seed(0)
    B, d, n = 6, 4, 10
    u = torch.nn.functional.normalize(torch.randn(B, d, generator=g), dim=-1)
    W = torch.nn.functional.normalize(torch.randn(n, d, generator=g), dim=-1)
    targets = torch.tensor([1, 3, 1, 5, 7, 3])  # duplicates: rows 0/2 and 1/5
    log_q = torch.log(torch.tensor([.05, .2, .05, .1, .05, .15, .05, .2, .1, .05]))
    neg = torch.tensor([2, 3, 9])
    tau, lq_neg = 0.1, float(np.log(1 / 10))
    got = tt.in_batch_loss(u, W[targets], targets, log_q, tau, neg, W[neg], lq_neg)
    total = 0.0
    for i in range(B):
        cands = []
        for j in range(B):
            if j != i and targets[j] == targets[i]:
                continue  # duplicate of my positive: masked
            cands.append((j == i, float(u[i] @ W[targets[j]]) / tau - float(log_q[targets[j]])))
        for k in neg:
            if k != targets[i]:
                cands.append((False, float(u[i] @ W[k]) / tau - lq_neg))
        logits = np.array([c[1] for c in cands])
        pos = [c[1] for c in cands if c[0]][0]
        total += -(pos - np.log(np.exp(logits).sum()))
    assert float(got) == pytest.approx(total / B, rel=1e-5)


def test_training_on_fixture_and_scoring(seq):
    cfg = {"dim": 8, "hidden": 16, "batch": 16, "hist_len": 5, "pairs_per_user": None,
           "max_epochs": 3, "tau": 0.1, "lr": 1e-2}
    a, info = tt.train_two_tower(seq, cfg, seed=1, fixed_epochs=3, log=lambda *_: None, dev=CPU)
    b, _ = tt.train_two_tower(seq, cfg, seed=1, fixed_epochs=3, log=lambda *_: None, dev=CPU)
    users = seq.user_ids[seq.lengths > 0]
    s = a.score(users)
    assert s.shape == (len(users), seq.n_items)
    trained = seq.trained_items()
    assert np.isneginf(s[:, ~trained]).all() and np.isfinite(s[:, trained]).all()
    assert np.allclose(s[:, trained], b.score(users)[:, trained])  # seeded, CPU: reproducible
    assert info["epochs"] == 3 and a.config["epochs"] == 3
    hist = info["history"]
    assert hist[-1]["loss"] < hist[0]["loss"]


def test_early_stopping_keeps_best_epoch(seq):
    vals = iter([0.1, 0.3, 0.2, 0.1, 0.05])
    cfg = {"dim": 8, "hidden": 16, "batch": 16, "hist_len": 5, "max_epochs": 5}
    sc, info = tt.train_two_tower(seq, cfg, seed=0, eval_fn=lambda s: next(vals),
                                  log=lambda *_: None, dev=CPU)
    assert info["epochs"] == 2           # best val at epoch 2
    assert len(info["history"]) == 4     # stopped after 2 epochs without improvement


def test_two_tower_through_harness(seq, phase1, items):
    r = ev.load_ratings(phase1, "user")
    d = ev.build_eval_data(r, items, "user", "val", n_negatives=5)
    sc, _ = tt.train_two_tower(seq, {"dim": 8, "hidden": 16, "batch": 16, "hist_len": 5},
                               seed=0, fixed_epochs=2, log=lambda *_: None, dev=CPU)
    s = ev.evaluate(sc, d).summary
    assert s["n_users"] == d.n_users and 0 <= s["recall@10"] <= 1



def test_strict_time_history_excludes_same_second_ratings(seq):
    users, pos = tt.all_pairs(seq)
    ends = tt.history_ends(seq, users, pos, strict_time=True)
    assert np.array_equal(tt.history_ends(seq, users, pos, strict_time=False), pos)
    n_burst = 0
    for u, p, e in zip(users, pos, ends):
        o = seq.offsets[u]
        ts = seq.timestamps[o:o + seq.lengths[u]]
        assert 0 <= e <= p
        assert (ts[:e] < ts[p]).all()                 # every history rating is strictly earlier
        assert (ts[e:p] == ts[p]).all()               # everything cut shares the target's second
        n_burst += e < p
    assert n_burst > 0  # the fixture has bursts, so the rule actually bites



def test_save_and_load_roundtrip(seq, tmp_path):
    sc, _ = tt.train_two_tower(seq, {"dim": 8, "hidden": 16, "batch": 16, "hist_len": 5},
                               seed=3, fixed_epochs=2, log=lambda *_: None, dev=CPU)
    path = tmp_path / "m" / "tt.pt"
    tt.save_scorer(sc, path, seed=3)
    back = tt.load_scorer(path, seq, dev=CPU)
    users = seq.user_ids[seq.lengths > 0]
    assert np.array_equal(sc.score(users), back.score(users))
    assert back.config == sc.config
    other = tt.Sequences(seq.user_ids, seq.items[:-1], seq.offsets, seq.n_items, seq.training)
    with pytest.raises(ValueError, match="different data"):
        tt.load_scorer(path, other, dev=CPU)

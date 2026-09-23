"""Phase 3: each baseline is checked against an independent brute-force implementation on the
fixture, and the tuner's grid extension is checked with a stub scorer."""
import json
from pathlib import Path

import numpy as np
import pytest

from src import baselines as bl
from src import data_prep as dp
from src import evaluate as ev
from src import tune_baselines as tb

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="module")
def phase1(spark, tmp_path_factory):
    out = tmp_path_factory.mktemp("phase1_bl")
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
def ratings(phase1):
    return ev.load_ratings(phase1, "user")


@pytest.fixture(scope="module")
def td(ratings, items):
    return bl.build_train_data(ratings, items, ("train",))


def dense(td):
    return td.X.toarray().astype(np.float64)


def test_train_data_is_train_positives_only(td, ratings, items):
    r = ratings.assign(split=ratings.split.astype(str))
    pos = r[(r.split == "train") & (r.rating >= 4)]
    X = dense(td)
    assert X.sum() == len(pos)
    for u, m in zip(pos.userId, pos.movieId):
        assert X[td.rows([u])[0], items.to_index([m])[0]] == 1
    # nothing from val/test, nothing below 4
    other = r[(r.split != "train") | (r.rating < 4)]
    for u, m in zip(other.userId, other.movieId):
        assert X[td.rows([u])[0], items.to_index([m])[0]] == 0
    tv = bl.build_train_data(ratings, items, ("train", "val"))
    assert tv.X.nnz == ((r.split.isin(["train", "val"])) & (r.rating >= 4)).sum()


def test_most_popular(td):
    s = bl.MostPopular(td).score(td.user_ids[:3])
    assert s.shape == (3, len(td.items))
    assert np.array_equal(s[0], dense(td).sum(axis=0))


def brute_knn_scores(X, keep, k, shrink, users_rows):
    Xk = X[:, keep]
    G = Xk.T @ Xk
    n = np.sqrt(np.diag(G))
    S = G / (np.outer(n, n) + shrink)
    np.fill_diagonal(S, 0)
    kk = min(k, S.shape[0] - 1)
    S2 = np.zeros_like(S)
    for j in range(S.shape[0]):
        cand = [i for i in np.argsort(-S[:, j], kind="stable") if i != j][:kk]  # ties: index
        S2[cand, j] = S[cand, j]
    return Xk[users_rows] @ S2


@pytest.mark.parametrize("k,shrink", [(1, 0.0), (3, 0.0), (5, 10.0), (100, 1.0)])
def test_item_knn_matches_brute_force(td, k, shrink):
    gram = bl.ItemGram(td, min_pos=2)
    m = bl.ItemKNN(td, gram, k, shrink)
    got = m.score(td.user_ids)
    ref = brute_knn_scores(dense(td), gram.keep, k, shrink, np.arange(len(td.user_ids)))
    assert np.allclose(got[:, gram.keep], ref, atol=1e-5)  # exact, ties included
    dropped = np.setdiff1d(np.arange(len(td.items)), gram.keep)
    assert np.isneginf(got[:, dropped]).all()


def test_item_knn_is_deterministic(td):
    gram = bl.ItemGram(td, min_pos=2)
    a = bl.ItemKNN(td, gram, 3, 0.0).S
    b = bl.ItemKNN(td, gram, 3, 0.0).S
    assert (a != b).nnz == 0


def test_ease_matches_per_column_least_squares(td):
    """EASE's closed form must equal, for every column j, the ridge solution
    B[-j, j] = (G[-j,-j] + lam I)^-1 G[-j, j] with B[j, j] = 0 (Steck 2019)."""
    lam = 3.0
    gram = bl.ItemGram(td, min_pos=2)
    m = bl.EASE(td, gram, lam)
    G = gram.G
    n = G.shape[0]
    ref = np.zeros((n, n))
    for j in range(n):
        o = np.array([i for i in range(n) if i != j])
        ref[o, j] = np.linalg.solve(G[np.ix_(o, o)] + lam * np.eye(n - 1), G[o, j])
    assert np.allclose(m.B, ref, atol=1e-4)
    assert np.all(np.diag(m.B) == 0)
    s = m.score(td.user_ids[:4])
    Xk = dense(td)[:, gram.keep][:4]
    assert np.allclose(s[:, gram.keep], Xk @ ref, atol=1e-4)


def test_implicit_als_on_fixture(spark, phase1, td):
    pos = bl.positives_for_spark(spark, phase1, "user", ("train",))
    a = bl.ImplicitALS(td, pos, rank=4, reg=0.1, alpha=5.0, max_iter=5, seed=1)
    b = bl.ImplicitALS(td, pos, rank=4, reg=0.1, alpha=5.0, max_iter=5, seed=1)
    s = a.score(td.user_ids)
    assert s.shape == (len(td.user_ids), len(td.items))
    no_pos = dense(td).sum(axis=0) == 0
    assert np.isneginf(s[:, no_pos]).all() and np.isfinite(s[:, ~no_pos]).all()
    assert np.allclose(s[:, ~no_pos], b.score(td.user_ids)[:, ~no_pos], atol=1e-5)  # seeded


def test_every_baseline_runs_through_harness(td, ratings, items, spark, phase1):
    d = ev.build_eval_data(ratings, items, "user", "val", n_negatives=5)
    gram = bl.ItemGram(td, min_pos=2)
    pos = bl.positives_for_spark(spark, phase1, "user", ("train",))
    for m in (bl.MostPopular(td), bl.ItemKNN(td, gram, 5, 1.0), bl.EASE(td, gram, 10.0),
              bl.ImplicitALS(td, pos, rank=4, reg=0.1, alpha=5.0, max_iter=3)):
        s = ev.evaluate(m, d).summary
        assert s["n_users"] == d.n_users
        assert 0 <= s["recall@10"] <= 1 and 0 <= s["auc"] <= 1


def test_tuner_extends_grid_past_edge(monkeypatch, tmp_path):
    monkeypatch.setattr(tb, "TRIALS_CSV", tmp_path / "trials.csv")  # never touch real results
    class Stub:
        def __init__(self, v):
            self.v = v

    def fake_eval(model, data):
        v = model.v  # peak at 1600, outside the initial grid
        return ev.EvalResult({"ndcg@10": -abs(np.log2(v / 1600)), "recall@10": 0, "auc": 0,
                              "coverage": 0, "n_users": 1}, None)

    monkeypatch.setattr(tb.ev, "evaluate", fake_eval)
    t = tb.Tuner("stub", None)
    best = t.search_1d(lambda v: Stub(v), [100.0, 200.0, 400.0], "lam", {})
    assert best == 1600.0
    assert [json.loads(r["config"])["lam"] for r in t.trials] == [100, 200, 400, 800, 1600, 3200]
    # interior optimum: no extension
    t2 = tb.Tuner("stub", None, resume=False)
    assert t2.search_1d(lambda v: Stub(v), [800.0, 1600.0, 3200.0], "lam", {}) == 1600.0
    assert len(t2.trials) == 3


def test_als_is_implicit_and_uses_only_positives(spark, phase1, td, monkeypatch):
    """Mistake #4 in the guide: ALS on explicit ratings. Capture what ALS is actually given."""
    import pyspark.ml.recommendation as rec
    seen = {}
    real_fit = rec.ALS.fit

    def spy_fit(self, df, *a, **k):  # read the settings off the estimator actually being fit
        seen["implicitPrefs"] = self.getImplicitPrefs()
        seen["labels"] = sorted({r.label for r in df.select("label").distinct().collect()})
        seen["rows"] = df.count()
        return real_fit(self, df, *a, **k)

    monkeypatch.setattr(rec.ALS, "fit", spy_fit)
    pos = bl.positives_for_spark(spark, phase1, "user", ("train",))
    bl.ImplicitALS(td, pos, rank=4, reg=0.1, alpha=5.0, max_iter=2)
    assert seen["implicitPrefs"] is True
    assert seen["labels"] == [1.0]
    assert seen["rows"] == td.X.nnz  # exactly the train positives



def test_tuner_resumes_from_logged_trials(monkeypatch, tmp_path):
    """An interrupted run must not redo finished trials, and must refit a logged best."""
    monkeypatch.setattr(tb, "TRIALS_CSV", tmp_path / "trials.csv")
    built = []

    class Stub:
        def __init__(self, v):
            built.append(v)
            self.v = v

    monkeypatch.setattr(tb.ev, "evaluate", lambda m, d: ev.EvalResult(
        {"ndcg@10": -abs(m.v - 3), "recall@10": 0, "auc": 0, "coverage": 0, "n_users": 1}, None))
    a = tb.Tuner("stub", None)
    a.search_1d(lambda v: Stub(v), [1, 2], "k", {}, max_extend=0)
    assert built == [1, 2]
    # "crash", then resume: logged configs are skipped, only new ones are fit
    b = tb.Tuner("stub", None)
    assert len(b.trials) == 2 and b.best[2] is None
    b.search_1d(lambda v: Stub(v), [1, 2, 3], "k", {}, max_extend=0)
    assert built == [1, 2, 3]
    assert [r["trial"] for r in b.trials] == [1, 2, 3]
    # best logged but not in memory -> refit on demand
    c = tb.Tuner("stub", None)
    m = c.best_model(lambda cfg: Stub(cfg["k"]))
    assert m.v == 3 and built == [1, 2, 3, 3]
    # --fresh ignores the log
    assert tb.Tuner("stub", None, resume=False).trials == []

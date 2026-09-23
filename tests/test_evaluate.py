"""Phase 2: the harness must reproduce hand-computed metrics, and its data must follow the
protocol (population, masking, negatives) exactly."""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from src import data_prep as dp
from src import evaluate as ev

FIXTURES = Path(__file__).resolve().parent / "fixtures"


# ---------------------------------------------------------------------------------------------
# Hand-computed toy example
# ---------------------------------------------------------------------------------------------

class FixedModel:
    def __init__(self, scores_by_user):
        self.s = scores_by_user

    def score(self, users):
        return np.array([self.s[u] for u in users], dtype=float)


def toy():
    """6 items (movieIds 10..60 -> index 0..5), 2 users, k = 3.

    User 1: seen {0}, relevant {2, 4}, scores 0.9 .. 0.4 (strictly decreasing).
        after masking item 0, top-3 = [1, 2, 3], hits = [0, 1, 0]
        recall = 1 / min(2, 3) = 0.5
        DCG = 1/log2(3);  IDCG = 1 + 1/log2(3);  NDCG = 0.63093 / 1.63093 = 0.386853
        AUC vs negatives {3, 5}: pos {0.7, 0.5} vs neg {0.6, 0.4} -> 3 of 4 pairs = 0.75
    User 2: seen {}, relevant {0}, scores all 0.5 except item 5 = 0.9 (ties!).
        top-3 by (-score, index) = [5, 0, 1], hits = [0, 1, 0]
        recall = 1 / min(1, 3) = 1.0
        NDCG = (1/log2(3)) / 1 = 0.630930
        AUC vs negatives {1, 5}: 0.5 vs 0.5 tie (0.5), 0.5 vs 0.9 loss (0) -> 0.25
    Means: recall 0.75, NDCG 0.508891, AUC 0.5. Recommended {0,1,2,3,5}: coverage 5/6.
    """
    items = ev.ItemIndex.from_ids([10, 20, 30, 40, 50, 60])
    csr = lambda rows: sp.csr_matrix(
        (np.ones(sum(map(len, rows)), dtype=np.int8),
         (np.repeat(np.arange(len(rows)), [len(r) for r in rows]), np.concatenate(rows))),
        shape=(len(rows), 6))
    data = ev.EvalData("toy", "val", items, np.array([1, 2]),
                       seen=csr([np.array([0]), np.array([], dtype=int)]),
                       relevant=csr([np.array([2, 4]), np.array([0])]),
                       negatives=np.array([[3, 5], [1, 5]]),
                       train_items=np.ones(6, dtype=bool),
                       counts={"users_no_train_positive": 0, "users_no_target_positive": 0})
    model = FixedModel({1: [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
                        2: [0.5, 0.5, 0.5, 0.5, 0.5, 0.9]})
    return data, model


def test_hand_computed_metrics():
    data, model = toy()
    res = ev.evaluate(model, data, k=3)
    s, pu = res.summary, res.per_user.set_index("userId")
    l3 = 1 / math.log2(3)
    assert pu.loc[1, "recall"] == pytest.approx(0.5)
    assert pu.loc[1, "ndcg"] == pytest.approx(l3 / (1 + l3))
    assert pu.loc[1, "auc"] == pytest.approx(0.75)
    assert pu.loc[2, "recall"] == pytest.approx(1.0)
    assert pu.loc[2, "ndcg"] == pytest.approx(l3)
    assert pu.loc[2, "auc"] == pytest.approx(0.25)
    assert s["recall@10"] == pytest.approx(0.75)
    assert s["ndcg@10"] == pytest.approx((l3 / (1 + l3) + l3) / 2)
    assert s["auc"] == pytest.approx(0.5)
    assert s["coverage"] == pytest.approx(5 / 6)
    assert s["n_users"] == 2 and s["skipped_all_seen"] == 0


def test_mean_not_median():
    # Three users with recall 0, 0, 1: mean 1/3, median would be 0.
    items = ev.ItemIndex.from_ids(range(5))
    rel = sp.csr_matrix(([1, 1, 1], ([0, 1, 2], [4, 4, 0])), shape=(3, 5), dtype=np.int8)
    data = ev.EvalData("t", "val", items, np.array([1, 2, 3]),
                       sp.csr_matrix((3, 5), dtype=np.int8), rel,
                       np.zeros((3, 1), dtype=int), np.ones(5, bool), {})
    m = FixedModel({u: [5, 4, 3, 2, 1] for u in (1, 2, 3)})
    assert ev.evaluate(m, data, k=1).summary["recall@10"] == pytest.approx(1 / 3)


def test_recall_is_capped_at_k():
    # 5 relevant items, k = 3, all top-3 relevant: capped recall = 3/min(5,3) = 1.0 (uncapped 0.6).
    # NDCG is 1.0 too: IDCG covers min(5, 3) = 3 positions.
    items = ev.ItemIndex.from_ids(range(8))
    rel = sp.csr_matrix((np.ones(5), ([0] * 5, [0, 1, 2, 3, 4])), shape=(1, 8), dtype=np.int8)
    data = ev.EvalData("t", "val", items, np.array([1]), sp.csr_matrix((1, 8), dtype=np.int8),
                       rel, np.array([[5, 6]]), np.ones(8, bool), {})
    res = ev.evaluate(FixedModel({1: [8, 7, 6, 5, 4, 3, 2, 1]}), data, k=3).summary
    assert res["recall@10"] == pytest.approx(1.0)
    assert res["ndcg@10"] == pytest.approx(1.0)
    # 2 of top-3 relevant: recall 2/3
    res = ev.evaluate(FixedModel({1: [8, 7, 1, 1, 1, 1, 1, 6.5]}), data, k=3).summary
    assert res["recall@10"] == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------------------------
# Ranking primitives
# ---------------------------------------------------------------------------------------------

def test_top_k_matches_stable_full_sort_with_heavy_ties():
    rng = np.random.default_rng(0)
    s = rng.integers(0, 4, size=(200, 50)).astype(float)  # lots of ties
    s[rng.random(s.shape) < 0.3] = -np.inf
    got = ev.top_k(s, 10)
    for i in range(len(s)):
        ref = [j for j in np.argsort(-s[i], kind="stable") if np.isfinite(s[i, j])][:10]
        assert list(got[i][got[i] >= 0]) == ref



@pytest.mark.parametrize("window", [0, 3, 64])
def test_top_k_huge_tie_groups_match_stable_sort(window):
    # Catalog much larger than k + window, so rows take the tie-group path; includes an
    # oracle-like row (one 1, rest 0), a row with ties straddling k, -inf blocks, and all -inf.
    rng = np.random.default_rng(1)
    n = 500
    s = rng.integers(0, 3, size=(40, n)).astype(float)
    s[rng.random(s.shape) < 0.2] = -np.inf
    s[0] = 0.0
    s[0, 321] = 1.0
    s[1] = -np.inf
    s[1, [7, 400]] = 5.0
    s[2] = -np.inf
    s[3, :] = 2.0
    s[4] = 0.0                      # huge tie group at the cutoff...
    s[4, [300, 20, 150]] = 3.0      # ...plus ties among the strictly-better items
    s[4, 10] = 4.0
    got = ev.top_k(s, 10, window=window)
    assert list(got[4][:6]) == [10, 20, 150, 300, 0, 1]
    for i in range(len(s)):
        ref = [j for j in np.argsort(-s[i], kind="stable") if np.isfinite(s[i, j])][:10]
        assert list(got[i][got[i] >= 0]) == ref, i

def test_top_k_fewer_finite_than_k_and_k_larger_than_catalog():
    s = np.array([[-np.inf, 2.0, -np.inf, 1.0]])
    assert list(ev.top_k(s, 3)[0]) == [1, 3, -1]
    assert list(ev.top_k(np.array([[1.0, 3.0]]), 10)[0][:3]) == [1, 0, -1]


def test_masked_items_never_recommended():
    data, model = toy()
    m = FixedModel({1: [100, 0, 0, 0, 0, 0], 2: [0] * 6})
    res = ev.evaluate(m, data, k=3)
    assert res.per_user.set_index("userId").loc[1, "hits"] <= 1


def test_nan_and_posinf_rejected():
    data, _ = toy()
    for bad in (np.nan, np.inf):
        m = FixedModel({1: [bad, 0, 0, 0, 0, 0], 2: [0] * 6})
        with pytest.raises(ValueError, match="NaN"):
            ev.evaluate(m, data, k=3)


def test_wrong_shape_rejected():
    data, _ = toy()
    m = FixedModel({1: [0] * 5, 2: [0] * 5})
    with pytest.raises(ValueError, match="expected"):
        ev.evaluate(m, data, k=3)


def test_model_that_cannot_score_user_gets_zero_not_skipped():
    data, _ = toy()
    m = FixedModel({1: [-np.inf] * 6, 2: [-np.inf] * 6})
    res = ev.evaluate(m, data, k=3)
    assert res.summary["n_users"] == 2  # a model can't dodge users by returning -inf
    assert res.summary["recall@10"] == 0.0
    assert res.summary["auc"] == pytest.approx(0.5)  # all ties


def test_user_with_whole_catalog_seen_is_skipped_and_counted():
    items = ev.ItemIndex.from_ids(range(3))
    seen = sp.csr_matrix(([1, 1, 1], ([0, 0, 0], [0, 1, 2])), shape=(2, 3), dtype=np.int8)
    rel = sp.csr_matrix(([1, 1], ([0, 1], [0, 1])), shape=(2, 3), dtype=np.int8)
    data = ev.EvalData("t", "val", items, np.array([1, 2]), seen, rel,
                       np.zeros((2, 1), dtype=int), np.ones(3, bool), {})
    res = ev.evaluate(FixedModel({1: [1, 2, 3], 2: [3, 2, 1]}), data, k=2)
    assert res.summary["skipped_all_seen"] == 1 and res.summary["n_users"] == 1


def test_batch_size_does_not_change_results():
    data, model = toy()
    a = ev.evaluate(model, data, k=3, batch_size=1).summary
    b = ev.evaluate(model, data, k=3, batch_size=100).summary
    assert a == b


def test_model_scores_not_mutated():
    data, _ = toy()
    arr = {1: np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4]), 2: np.full(6, 0.5)}

    class Shared:
        def score(self, users):
            return np.stack([arr[u] for u in users])

    ev.evaluate(Shared(), data, k=3)
    assert np.isfinite(arr[1]).all()


def test_item_index_rejects_unknown_movie():
    items = ev.ItemIndex.from_ids([3, 1, 2])
    assert list(items.to_index([1, 3])) == [0, 2]
    with pytest.raises(KeyError):
        items.to_index([4])


# ---------------------------------------------------------------------------------------------
# EvalData built from real Phase 1 output (on the fixture)
# ---------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fixture_splits(spark, tmp_path_factory):
    out = tmp_path_factory.mktemp("phase1")
    raw = out / "raw"
    raw.mkdir()
    (raw / "ratings.csv").write_text((FIXTURES / "tiny_ratings.csv").read_text())
    (raw / "movies.csv").write_text((FIXTURES / "tiny_movies.csv").read_text())
    dp.run(spark, raw, out, out / "stats.csv", with_genome=False)
    return out / "splits.parquet", out / "stats.csv"


@pytest.fixture(scope="module")
def items():
    return ev.ItemIndex.from_movies_csv(FIXTURES / "tiny_movies.csv")


@pytest.mark.parametrize("scheme", dp.SCHEMES)
@pytest.mark.parametrize("slice_", ev.SLICES)
def test_eval_data_follows_protocol(fixture_splits, items, scheme, slice_):
    splits_path, stats_path = fixture_splits
    r = ev.load_ratings(splits_path, scheme)
    d = ev.build_eval_data(r, items, scheme, slice_, n_negatives=5)
    r = r.assign(split=r.split.astype(str))
    training = ["train"] if slice_ == "val" else ["train", "val"]
    pos = r[r.rating >= 4]

    # population = >=1 train positive AND >=1 target positive
    expected = sorted(set(pos[pos.split == "train"].userId) & set(pos[pos.split == slice_].userId))
    assert list(d.users) == expected
    st = pd.read_csv(stats_path)
    stat = st[(st.scheme == scheme) & (st.stat == f"users_eligible_with_{slice_}_positive")]
    assert d.n_users == stat.value.iloc[0]  # agrees with Phase 1's data_stats.csv

    mid = items.movie_ids
    for i, u in enumerate(d.users):
        seen = set(mid[d.seen[i].indices])
        rel = set(mid[d.relevant[i].indices])
        assert seen == set(r[(r.userId == u) & r.split.isin(training)].movieId)
        assert rel == set(pos[(pos.userId == u) & (pos.split == slice_)].movieId)
        assert not seen & rel
        negs = d.negatives[i]
        assert len(set(negs)) == len(negs) == 5
        assert not set(mid[negs]) & set(r[r.userId == u].movieId)  # never rated, any slice
        assert d.train_items[negs].all()

    assert set(mid[d.train_items]) == set(r[r.split.isin(training)].movieId)


def test_negatives_are_deterministic(fixture_splits, items):
    r = ev.load_ratings(fixture_splits[0], "user")
    a = ev.build_eval_data(r, items, "user", "val", n_negatives=5)
    b = ev.build_eval_data(r, items, "user", "val", n_negatives=5)
    assert np.array_equal(a.negatives, b.negatives)


def test_oracle_and_random_on_fixture(fixture_splits, items):
    r = ev.load_ratings(fixture_splits[0], "user")
    d = ev.build_eval_data(r, items, "user", "test", n_negatives=5)
    o = ev.evaluate(ev.OracleModel(d), d).summary
    assert o["recall@10"] == o["ndcg@10"] == o["auc"] == 1.0
    rnd = ev.evaluate(ev.RandomModel(len(items)), d).summary
    assert rnd["recall@10"] < 1.0


def test_subsample_is_fixed_and_consistent(fixture_splits, items):
    r = ev.load_ratings(fixture_splits[0], "user")
    d = ev.build_eval_data(r, items, "user", "val", n_negatives=5)
    a, b = d.subsample(3, seed=1), d.subsample(3, seed=1)
    assert list(a.users) == list(b.users) and a.n_users == 3
    for i, u in enumerate(a.users):
        j = list(d.users).index(u)
        assert (a.relevant[i] != d.relevant[j]).nnz == 0
        assert np.array_equal(a.negatives[i], d.negatives[j])


# ---------------------------------------------------------------------------------------------
# Comparing and recording runs
# ---------------------------------------------------------------------------------------------

def test_paired_bootstrap():
    a = pd.DataFrame({"userId": range(500), "ndcg": np.linspace(0, 1, 500)})
    same = ev.paired_bootstrap(a, a, "ndcg")
    assert same["mean_diff"] == 0 and same["ci_low"] == 0 and same["ci_high"] == 0
    b = a.assign(ndcg=a.ndcg - 0.1)
    d = ev.paired_bootstrap(a, b, "ndcg")
    assert d["mean_diff"] == pytest.approx(0.1) and d["ci_low"] <= 0.1 <= d["ci_high"]
    with pytest.raises(ValueError):
        ev.paired_bootstrap(a, b.iloc[:-1], "ndcg")


def test_results_csv_roundtrip_and_seed_summary(tmp_path):
    data, model = toy()
    res = ev.evaluate(model, data, k=3)
    path = tmp_path / "r.csv"
    rows = [ev.result_row("toy", s, {"k": 3}, res, data) for s in (1, 2, 3)]
    ev.append_results(rows[:1], path)
    ev.append_results(rows[1:], path)  # appends, header written once
    df = pd.read_csv(path)
    assert list(df.columns) == ev.RESULT_COLUMNS and len(df) == 3
    summ = ev.summarize_seeds(df)
    assert summ.loc[0, "n_seeds"] == 3
    assert summ.loc[0, "recall@10_median"] == pytest.approx(0.75)

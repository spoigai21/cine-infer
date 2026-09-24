"""Phase 6: candidates, features and labels must each come from ONE training set, and the
two-stage plumbing must not change results on its own (identity ranker == retrieval only)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src import baselines as bl
from src import data_prep as dp
from src import evaluate as ev
from src import ranker as rk
from src import two_tower as tt

FIXTURES = Path(__file__).resolve().parent / "fixtures"
CPU = torch.device("cpu")
K = 5


@pytest.fixture(scope="module")
def env(spark, tmp_path_factory):
    out = tmp_path_factory.mktemp("phase1_rk")
    raw = out / "raw"
    raw.mkdir()
    (raw / "ratings.csv").write_text((FIXTURES / "tiny_ratings.csv").read_text())
    (raw / "movies.csv").write_text((FIXTURES / "tiny_movies.csv").read_text())
    dp.run(spark, raw, out, out / "stats.csv", with_genome=False)
    items = ev.ItemIndex.from_movies_csv(FIXTURES / "tiny_movies.csv")
    # synthetic tag genome for 25 of the 40 movies, 6 "tags"
    rng = np.random.default_rng(0)
    gm = items.movie_ids[:25]
    pd.DataFrame({"movieId": gm, "genome": list(rng.random((25, 6)))}).to_parquet(out / "genome.parquet")
    content = rk.ItemContent.load(FIXTURES / "tiny_movies.csv", out / "genome.parquet", items)
    fine = ev.load_ratings(out / "splits.parquet", "user", fine=True)
    return {"out": out, "items": items, "content": content, "fine": fine,
            "splits": pd.read_parquet(out / "splits.parquet")}


def make_ctx(env, name="train_core", training=("train_core",), use_ease=True, k=K):
    seq = tt.load_sequences(env["out"] / "splits.parquet", env["items"], training)
    retr, _ = tt.train_two_tower(seq, {"dim": 8, "hidden": 16, "batch": 16, "hist_len": 5},
                                 seed=0, fixed_epochs=2, log=lambda *_: None, dev=CPU)
    ease = None
    if use_ease:
        td = bl.build_train_data(env["fine"], env["items"], training)
        ease = bl.EASE(td, bl.ItemGram(td, 1), 5.0)
    return rk.TrainingSetContext.build(name, retr, ease, seq, env["fine"], training,
                                       env["out"] / "features", env["items"], env["content"], k)


@pytest.fixture(scope="module")
def ctx(env):
    return make_ctx(env)


@pytest.fixture(scope="module")
def ctx15(env):
    """More candidates, so some train_tail positives reach them on the tiny fixture."""
    return make_ctx(env, k=15)


def users_with_core(ctx):
    return ctx.seq.user_ids[ctx.seq.lengths > 0]


def test_candidates_mask_everything_rated_in_the_training_set(ctx, env):
    users = users_with_core(ctx)
    cand, s = rk.candidates(ctx, users)
    s_full = ctx.retriever.score(users)
    fine = env["fine"]
    for i, u in enumerate(users):
        rated = set(env["items"].to_index(fine[(fine.userId == u) & (fine.split == "train_core")].movieId))
        assert not rated & set(cand[i])                      # nothing rated in train_core
        allowed = [j for j in np.argsort(-s_full[i], kind="stable") if j not in rated
                   and np.isfinite(s_full[i, j])]
        assert list(cand[i]) == allowed[:K]                  # exactly the retriever's top-k
    assert (np.diff(s, axis=1) <= 0).all()


def test_features_come_from_the_training_set_only(ctx, env):
    users = users_with_core(ctx)[:6]
    cand, X = rk.build_features(ctx, users)
    f = {n: X[..., i] for i, n in enumerate(rk.FEATURES)}
    sp_ = env["splits"]
    core = sp_[(sp_.split_user == "train") & (sp_.train_part == "core")]
    mid = env["items"].movie_ids
    for i, u in enumerate(users):
        mine = core[core.userId == u]
        assert f["user_log_n"][i, 0] == pytest.approx(np.log1p(len(mine)))
        assert f["user_mean"][i, 0] == pytest.approx(mine.rating.mean())
        for j, c in enumerate(cand[i]):
            m = core[core.movieId == mid[c]]
            if len(m):  # counts are train_core counts: train_tail/val/test never leak in
                assert f["item_log_n"][i, j] == pytest.approx(np.log1p(len(m)))
                assert f["item_log_pos"][i, j] == pytest.approx(np.log1p((m.rating >= 4).sum()))
                assert f["item_days_since_last"][i, j] == pytest.approx(
                    (mine.timestamp.max() - m.timestamp.max()) / 86400, rel=1e-5)
            else:
                assert np.isnan(f["item_log_n"][i, j])
    assert np.array_equal(f["tt_rank"][0], np.arange(K))
    es = np.take_along_axis(ctx.ease.score(users), cand, axis=1)
    assert np.allclose(f["ease_score"], np.where(np.isfinite(es), es, np.nan), equal_nan=True)


def test_genre_and_genome_features_match_manual(ctx, env):
    users = users_with_core(ctx)[:5]
    cand, X = rk.build_features(ctx, users)
    f = {n: X[..., i] for i, n in enumerate(rk.FEATURES)}
    G, has = env["content"].genome, env["content"].has_genome
    genres = env["content"].genres.toarray()
    for i, u in enumerate(users):
        r = ctx.seq.rows([u])[0]
        hist = ctx.seq.items[ctx.seq.offsets[r]:ctx.seq.offsets[r + 1]]
        share = ctx.user_genres[r]
        for j, c in enumerate(cand[i]):
            assert f["genre_overlap"][i, j] == pytest.approx((genres[c] * share).sum(), abs=1e-6)
            for name, h in (("genome_sim_all", hist), ("genome_sim_recent", hist[-10:])):
                h = h[has[h]]
                if len(h) and has[c]:
                    p = G[h].sum(0)
                    ref = p @ G[c] / np.linalg.norm(p)
                    assert f[name][i, j] == pytest.approx(ref, abs=1e-5)
                else:
                    assert np.isnan(f[name][i, j])


def test_labelled_rows_use_label_slice_and_drop_users_without_signal(ctx15, env):
    ctx = ctx15
    users = users_with_core(ctx)
    fine = env["fine"]
    tail = fine[(fine.split == "train_tail") & (fine.rating >= 4)]
    rows = ctx.seq.rows(users)
    import scipy.sparse as sp
    L = sp.csr_matrix((np.ones(len(tail)), (ctx.seq.rows(tail.userId.to_numpy()),
                                             env["items"].to_index(tail.movieId.to_numpy()))),
                      shape=(len(ctx.seq.user_ids), len(env["items"])))
    X, y, groups, dropped = rk.labelled_rows(ctx, users, L, rows)
    assert len(X) == len(y) == groups.sum() and set(np.unique(y)) <= {0.0, 1.0}
    cand, _ = rk.candidates(ctx, users)
    expect_keep = [any(L[rows[i], c] for c in cand[i]) for i in range(len(users))]
    assert len(groups) == sum(expect_keep) and dropped == len(users) - sum(expect_keep)
    assert y.sum() == sum(sum(L[rows[i], c] for c in cand[i]) for i in range(len(users)))


def test_identity_ranker_reproduces_retrieval_only(env, tmp_path):
    """Candidates -> files -> PrecomputedScorer, with 'predictions' = the retriever's own score:
    the two-stage plumbing must reproduce retrieval-only metrics exactly."""
    ctx = make_ctx(env, "train", ("train_core", "train_tail"), k=10)
    r = ev.load_ratings(env["out"] / "splits.parquet", "user")
    d = ev.build_eval_data(r, env["items"], "user", "val", n_negatives=5)
    rk.write_eval_features(ctx, d.users, tmp_path, batch=3)  # small batch: exercises batching
    X = np.load(tmp_path / "eval_X.npy")
    cand = np.load(tmp_path / "eval_cand.npy")
    two = rk.PrecomputedScorer(np.load(tmp_path / "eval_users.npy"), cand,
                               X[..., rk.FEATURES.index("tt_score")], len(env["items"]))
    a = ev.evaluate(ctx.retriever, d).summary
    b = ev.evaluate(two, d).summary
    for m in ("recall@10", "ndcg@10", "coverage"):
        assert a[m] == pytest.approx(b[m]), m
    s = two.score(d.users[::-1])  # any user order
    assert (np.isfinite(s).sum(1) == (cand[::-1] >= 0).sum(1)).all()  # only candidates scored

    # With the retriever as fallback: the same top-k, every candidate above every
    # non-candidate, and non-candidates ordered by the retriever (so AUC isn't mostly ties).
    fb = rk.PrecomputedScorer(two.users, cand, two.pred, len(env["items"]), fallback=ctx.retriever)
    c = ev.evaluate(fb, d).summary
    for m in ("recall@10", "ndcg@10", "coverage"):
        assert b[m] == pytest.approx(c[m]), m
    assert c["auc"] == pytest.approx(a["auc"])  # identity ranker: same order as the retriever
    s, base = fb.score(d.users), ctx.retriever.score(d.users)
    for i in range(len(d.users)):
        ci = cand[i][cand[i] >= 0]
        non = np.setdiff1d(np.flatnonzero(np.isfinite(base[i])), ci)
        assert s[i, ci].min() > s[i, non].max()
        assert np.array_equal(s[i, non], base[i, non])


def test_lightgbm_runs_in_its_own_process_without_torch(ctx15, env, tmp_path):
    users = users_with_core(ctx15)
    fine = env["fine"]
    tail = fine[(fine.split == "train_tail") & (fine.rating >= 4)]
    import scipy.sparse as sp
    L = sp.csr_matrix((np.ones(len(tail)), (ctx15.seq.rows(tail.userId.to_numpy()),
                                             env["items"].to_index(tail.movieId.to_numpy()))),
                      shape=(len(ctx15.seq.user_ids), len(env["items"])))
    meta = rk.write_training_rows(ctx15, users, L, ctx15.seq.rows(users), tmp_path)
    assert meta["train_users_kept"] + meta["train_users_dropped_no_positive"] == len(users)
    rk.write_eval_features(ctx15, users, tmp_path)
    cfg = {"num_leaves": 4, "min_data_in_leaf": 1, "max_rounds": 20}
    a = rk.run_lgb(tmp_path, cfg, seed=0, fixed_rounds=5, pred_out=tmp_path / "a.npy")
    b = rk.run_lgb(tmp_path, cfg, seed=0, fixed_rounds=5, pred_out=tmp_path / "b.npy")
    assert a["torch_loaded"] is False                     # the whole point of the split
    assert a["rounds"] == 5
    pa, pb = np.load(tmp_path / "a.npy"), np.load(tmp_path / "b.npy")
    assert pa.shape == np.load(tmp_path / "eval_cand.npy").shape
    assert np.array_equal(pa, pb)                          # seeded + deterministic
    no_ease = [f for f in rk.FEATURES if f not in rk.EASE_FEATURES]
    c = rk.run_lgb(tmp_path, cfg, seed=0, features=no_ease, rows=[0, 2],
                   pred_out=tmp_path / "c.npy", save_model=tmp_path / "m.txt")
    assert c["features"] == no_ease and np.load(tmp_path / "c.npy").shape[0] == 2
    assert (tmp_path / "m.txt").exists() and 1 <= c["rounds"] <= 20   # early-stopped
    with pytest.raises(RuntimeError, match="lgb_ranker failed"):
        rk.run_lgb(tmp_path, cfg, seed=0, features=["no_such_feature"], pred_out=tmp_path / "d.npy")

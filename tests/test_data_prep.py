"""Phase 1 checks on the synthetic fixture. Spark results are compared against independent
pandas reference implementations, so a bug has to be made twice to slip through."""
from pathlib import Path

import pandas as pd
import pytest
from pyspark.sql import functions as F

from src import data_prep as dp

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="module")
def ratings(spark):
    return dp.read_csv(spark, FIXTURES / "tiny_ratings.csv", dp.RATINGS_SCHEMA)


@pytest.fixture(scope="module")
def movies(spark):
    return dp.read_csv(spark, FIXTURES / "tiny_movies.csv", dp.MOVIES_SCHEMA)


@pytest.fixture(scope="module")
def built(ratings):
    splits, t80, t90 = dp.build_splits(ratings)
    return splits.toPandas(), splits, t80, t90


@pytest.fixture(scope="module")
def pdf():
    return pd.read_csv(FIXTURES / "tiny_ratings.csv")


def reference_per_user(pdf):
    """Plain-pandas per-user split, written independently of the Spark code."""
    rows = []
    for _, g in pdf.sort_values(["userId", "timestamp", "movieId"]).groupby("userId"):
        n = len(g)
        n_train = n if n < 10 else n * 8 // 10
        n_val_end = n if n < 10 else n * 9 // 10
        n_core = n_train * 7 // 8
        for i, (_, r) in enumerate(g.iterrows(), start=1):
            split = "train" if i <= n_train else "val" if i <= n_val_end else "test"
            part = None if split != "train" else ("core" if i <= n_core else "tail")
            rows.append((r.userId, r.movieId, split, part))
    return pd.DataFrame(rows, columns=["userId", "movieId", "split_user", "train_part"])


def test_row_counts_reconcile(built, pdf):
    s, *_ = built
    assert len(s) == len(pdf)
    for scheme in dp.SCHEMES:
        assert s[f"split_{scheme}"].isin(["train", "val", "test"]).all()


def test_per_user_split_matches_reference(built, pdf):
    s, *_ = built
    got = s[["userId", "movieId", "split_user", "train_part"]]
    ref = reference_per_user(pdf)
    m = got.merge(ref, on=["userId", "movieId"], suffixes=("", "_ref"))
    assert len(m) == len(pdf)
    assert (m.split_user == m.split_user_ref).all()
    assert (m.train_part.fillna("-") == m.train_part_ref.fillna("-")).all()


def test_no_time_overlap_per_user(built):
    s, *_ = built
    for uid, g in s.groupby("userId"):
        by = g.groupby("split_user").timestamp
        lo, hi = by.min(), by.max()
        if "val" in lo:
            assert hi["train"] <= lo["val"], uid
            assert hi["val"] <= lo["test"], uid
        tr = g[g.split_user == "train"]
        if (tr.train_part == "tail").any():
            assert tr[tr.train_part == "core"].timestamp.max() <= \
                tr[tr.train_part == "tail"].timestamp.min(), uid


def test_small_user_is_train_only(built, pdf):
    s, *_ = built
    sizes = pdf.groupby("userId").size()
    small = sizes[sizes < 10].index
    assert len(small) > 0
    assert (s[s.userId.isin(small)].split_user == "train").all()


def test_exact_proportions_for_20_ratings(built, pdf):
    s, *_ = built
    sizes = pdf.groupby("userId").size()
    uid = sizes[sizes == 20].index[0]
    c = s[s.userId == uid].split_user.value_counts()
    assert (c["train"], c["val"], c["test"]) == (16, 2, 2)
    p = s[(s.userId == uid) & (s.split_user == "train")].train_part.value_counts()
    assert (p["core"], p["tail"]) == (14, 2)


def test_tie_broken_by_movie_id(built):
    s, *_ = built
    s = s.copy()
    order = {"train": 0, "val": 1, "test": 2}
    s["rank"] = s.split_user.map(order) * 10 + s.train_part.map({"core": 0, "tail": 1}).fillna(0)
    burst = s[s.duplicated(["userId", "timestamp"], keep=False)]
    assert len(burst) > 0
    for _, g in burst.groupby(["userId", "timestamp"]):
        # within one timestamp, later slices only ever get the larger movieIds
        g = g.sort_values("movieId")
        assert g["rank"].is_monotonic_increasing


def test_global_split_respects_cutoffs(built, pdf):
    s, _, t80, t90 = built
    assert (s[s.split_global == "train"].timestamp <= t80).all()
    v = s[s.split_global == "val"].timestamp
    assert ((v > t80) & (v <= t90)).all()
    assert (s[s.split_global == "test"].timestamp > t90).all()
    # cutoffs are exact percentiles (linear interpolation, then floored)
    assert t80 == int(pdf.timestamp.quantile(0.8))
    assert t90 == int(pdf.timestamp.quantile(0.9))


def test_random_split_is_partition_independent(ratings):
    a = dp.add_random_split(ratings.coalesce(1)).toPandas()
    b = dp.add_random_split(ratings.repartition(7)).toPandas()
    key = ["userId", "movieId"]
    m = a.merge(b, on=key)
    assert len(m) == len(a)
    assert (m.split_random_x == m.split_random_y).all()
    share_train = (a.split_random == "train").mean()
    assert 0.6 < share_train < 0.95


def test_split_is_deterministic(ratings):
    a, *_ = dp.build_splits(ratings)
    b, *_ = dp.build_splits(ratings.repartition(5))
    key = ["userId", "movieId"]
    pa = a.toPandas().sort_values(key).reset_index(drop=True)
    pb = b.toPandas().sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(pa, pb)


def test_check_splits_passes_and_catches_leaks(spark, built):
    s, splits, t80, t90 = built
    dp.check_splits(splits, len(s), t80, t90)  # clean split passes

    # Move one user's newest test rating into train: that's future leaking into training.
    uid = s[s.split_user == "test"].userId.iloc[0]
    last = s[(s.userId == uid)].timestamp.max()
    leaked = splits.withColumn(
        "split_user",
        F.when((F.col("userId") == int(uid)) & (F.col("timestamp") == int(last)), "train")
         .otherwise(F.col("split_user"))
    ).withColumn(
        "train_part",
        F.when((F.col("split_user") == "train") & F.col("train_part").isNull(), "tail")
         .otherwise(F.col("train_part")))
    with pytest.raises(AssertionError, match="overlap"):
        dp.check_splits(leaked, len(s), t80, t90)

    with pytest.raises(AssertionError, match="input rows"):
        dp.check_splits(splits.limit(len(s) - 1), len(s), t80, t90)


def test_eligible_user_counts(built, pdf):
    s, splits, *_ = built
    for scheme in dp.SCHEMES:
        st = dp.split_stats(splits, scheme)
        col = f"split_{scheme}"
        pos = s[s.rating >= 4]
        elig = set(pos[pos[col] == "train"].userId)
        assert st["users_eligible"] == len(elig), scheme
        assert st["users_excluded_no_train_positive"] == s.userId.nunique() - len(elig)
        test_pos = set(pos[pos[col] == "test"].userId) & elig
        assert st["users_eligible_with_test_positive"] == len(test_pos), scheme
    # the fixture's late starters must be excluded under the global cutoff
    assert dp.split_stats(splits, "global")["users_excluded_no_train_positive"] > 0


def test_burst_stat_matches_pandas(built):
    s, splits, *_ = built
    got = dp.burst_stats(splits)
    t = s[s.split_user == "test"].groupby("userId").timestamp
    span_h = (t.max() - t.min()) / 3600
    assert got["test_span_le_1h_share"] == pytest.approx((span_h <= 1).mean())
    assert got["test_span_median_hours"] == pytest.approx(span_h.median())


def test_features_use_train_only(built, movies, pdf):
    s, splits, *_ = built
    train = splits.filter(dp.TRAIN_SETS["train"])
    items = dp.item_features(train).toPandas().set_index("movieId")
    ref = s[s.split_user == "train"].groupby("movieId").rating
    assert set(items.index) == set(ref.count().index)  # no val/test-only movie gets features
    assert (items.n_ratings == ref.count().reindex(items.index)).all()
    assert items.mean_rating.values == pytest.approx(ref.mean().reindex(items.index).values)

    core = dp.item_features(splits.filter(dp.TRAIN_SETS["train_core"])).toPandas()
    assert core.n_ratings.sum() == (s.train_part == "core").sum()
    tv = dp.user_features(splits.filter(dp.TRAIN_SETS["train_val"])).toPandas()
    assert tv.n_ratings.sum() == s.split_user.isin(["train", "val"]).sum()


def test_genre_affinity(built, movies):
    s, splits, *_ = built
    aff = dp.user_genre_affinity(splits.filter(dp.TRAIN_SETS["train"]), movies).toPandas()
    assert (aff.share > 0).all() and (aff.share <= 1).all()
    assert "(no genres listed)" not in set(aff.genre)
    # recompute one user's shares in pandas
    mv = pd.read_csv(FIXTURES / "tiny_movies.csv")
    uid = aff.userId.iloc[0]
    pos = s[(s.userId == uid) & (s.split_user == "train") & (s.rating >= 4)]
    g = mv[mv.movieId.isin(pos.movieId)].genres.str.split("|").explode()
    g = g[g != "(no genres listed)"]
    ref = (g.value_counts() / len(pos)).sort_index()
    got = aff[aff.userId == uid].set_index("genre").share.sort_index()
    assert got.values == pytest.approx(ref.values)
    assert list(got.index) == list(ref.index)


def test_titles_with_commas_parse(movies):
    titles = {r.movieId: r.title for r in movies.collect()}
    assert titles[1] == "Amityville: A New Generation, The (1993)"
    assert titles[4] == '"Great Performances" Cats (1998)'


def test_genome_vectors_ordered_by_tag(spark):
    rows = [(1, 3, 0.3), (1, 1, 0.1), (1, 2, 0.2), (2, 2, 0.9), (2, 1, 0.8), (2, 3, 0.7)]
    g = spark.createDataFrame(rows, dp.GENOME_SCHEMA)
    v = {r.movieId: r.genome for r in dp.genome_vectors(g).collect()}
    assert v[1] == pytest.approx([0.1, 0.2, 0.3])
    assert v[2] == pytest.approx([0.8, 0.9, 0.7])


def test_end_to_end_run_writes_stats(spark, tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "ratings.csv").write_text((FIXTURES / "tiny_ratings.csv").read_text())
    (raw / "movies.csv").write_text((FIXTURES / "tiny_movies.csv").read_text())
    stats = tmp_path / "data_stats.csv"
    dp.run(spark, raw, tmp_path / "out", stats, with_genome=False)
    first = stats.read_text()
    st = pd.read_csv(stats)
    get = lambda sc, k: st[(st.scheme == sc) & (st.stat == k)].value.iloc[0]
    assert get("all", "ratings_total") == 200
    for sc in dp.SCHEMES:
        assert get(sc, "rows_train") + get(sc, "rows_val") + get(sc, "rows_test") == 200
    assert "test_span_le_1h_share" in set(st.stat)
    assert (tmp_path / "out" / "features" / "train_core" / "items.parquet").exists()
    dp.run(spark, raw, tmp_path / "out", stats, with_genome=False)
    assert stats.read_text() == first  # re-run is byte-identical

"""Phase 1: clean MovieLens, build the three splits, measure rating bursts, build train-only features.

Outputs (all git-ignored except the stats CSV):
  data/splits.parquet               one row per rating, with a column per split scheme
  data/features/<train_set>/...     item, user and user-genre features, one set per training set
  data/features/genome.parquet      tag-genome vector per movie
  results/data_stats.csv            every count and statistic below (committed)

Split columns on data/splits.parquet:
  split_user    per-user time split: train / val / test (80/10/10 of each user's ratings)
  train_part    within split_user == train: core (first 7/8) / tail (last 1/8, ranker labels)
  split_global  global time cutoff: ts <= t80 train, <= t90 val, else test
  split_random  hash-based random 80/10/10 (reproducible regardless of partitioning)

Run: `make prep` (or `python -m src.data_prep`).
"""
import argparse
import csv
import os
import time
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T

POSITIVE_THRESHOLD = 4.0   # a rating >= 4.0 is a positive (§1.6)
MIN_RATINGS_FOR_EVAL = 10  # users with fewer ratings go to train only
RANDOM_SEED = 42
SCHEMES = ("user", "global", "random")

RATINGS_SCHEMA = T.StructType([
    T.StructField("userId", T.IntegerType(), False),
    T.StructField("movieId", T.IntegerType(), False),
    T.StructField("rating", T.DoubleType(), False),
    T.StructField("timestamp", T.LongType(), False),
])
MOVIES_SCHEMA = T.StructType([
    T.StructField("movieId", T.IntegerType(), False),
    T.StructField("title", T.StringType(), False),
    T.StructField("genres", T.StringType(), False),
])
GENOME_SCHEMA = T.StructType([
    T.StructField("movieId", T.IntegerType(), False),
    T.StructField("tagId", T.IntegerType(), False),
    T.StructField("relevance", T.DoubleType(), False),
])


def get_spark(app="cineinfer", master="local[*]", driver_memory="8g", shuffle_partitions=64):
    return (SparkSession.builder
            .appName(app)
            .master(master)
            .config("spark.driver.memory", driver_memory)
            .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
            .config("spark.sql.session.timeZone", "UTC")
            .config("spark.ui.showConsoleProgress", "false")
            .getOrCreate())


def read_csv(spark, path, schema):
    # Explicit schema: inferSchema costs a full extra pass and could guess types wrong.
    # Titles contain commas and quotes; Spark's CSV reader handles RFC 4180 quoting.
    return spark.read.csv(str(path), header=True, schema=schema, escape='"', mode="FAILFAST")


# ---------------------------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------------------------

def add_per_user_split(ratings: DataFrame) -> DataFrame:
    """80/10/10 per user by (timestamp, movieId); train further cut 7/8 core, 1/8 tail.

    movieId breaks timestamp ties (users rate many movies in the same second), which makes the
    order, and so the split, deterministic. Cut points use integer arithmetic: floor(n * 0.8) in
    floating point is off by one for some n.
    """
    order = Window.partitionBy("userId").orderBy("timestamp", "movieId")
    r = (ratings
         .withColumn("rn", F.row_number().over(order))
         .withColumn("n", F.count("*").over(Window.partitionBy("userId")))
         .withColumn("n_train", F.expr(f"CASE WHEN n < {MIN_RATINGS_FOR_EVAL} THEN n "
                                       "ELSE (n * 8) div 10 END"))
         .withColumn("n_val_end", F.expr(f"CASE WHEN n < {MIN_RATINGS_FOR_EVAL} THEN n "
                                         "ELSE (n * 9) div 10 END"))
         .withColumn("n_core", F.expr("(n_train * 7) div 8")))
    r = (r.withColumn("split_user",
                      F.when(F.col("rn") <= F.col("n_train"), "train")
                       .when(F.col("rn") <= F.col("n_val_end"), "val")
                       .otherwise("test"))
          .withColumn("train_part",
                      F.when(F.col("split_user") != "train", F.lit(None).cast("string"))
                       .when(F.col("rn") <= F.col("n_core"), "core")
                       .otherwise("tail")))
    return r.drop("rn", "n", "n_train", "n_val_end", "n_core")


def global_cutoffs(ratings: DataFrame):
    """Exact 80th / 90th timestamp percentiles (deterministic, unlike approxQuantile)."""
    row = ratings.select(F.expr("percentile(timestamp, array(0.8, 0.9))").alias("q")).first()
    return int(row["q"][0]), int(row["q"][1])


def add_global_split(ratings: DataFrame, t80: int, t90: int) -> DataFrame:
    return ratings.withColumn("split_global",
                              F.when(F.col("timestamp") <= t80, "train")
                               .when(F.col("timestamp") <= t90, "val")
                               .otherwise("test"))


def add_random_split(ratings: DataFrame, seed: int = RANDOM_SEED) -> DataFrame:
    """Random 80/10/10 from a hash of (seed, userId, movieId).

    F.rand(seed) depends on how rows are partitioned, so it changes with core count or input
    layout. A hash of the row's key gives the same split on any machine.
    """
    u = F.pmod(F.xxhash64(F.lit(seed), "userId", "movieId"), F.lit(1_000_000)) / 1_000_000
    return ratings.withColumn("split_random",
                              F.when(u < 0.8, "train").when(u < 0.9, "val").otherwise("test"))


def build_splits(ratings: DataFrame):
    t80, t90 = global_cutoffs(ratings)
    s = add_random_split(add_global_split(add_per_user_split(ratings), t80, t90))
    cols = ["userId", "movieId", "rating", "timestamp",
            "split_user", "train_part", "split_global", "split_random"]
    return s.select(*cols), t80, t90


# ---------------------------------------------------------------------------------------------
# Checks (run on the real data every time; also exercised by tests on the fixture)
# ---------------------------------------------------------------------------------------------

def check_splits(splits: DataFrame, n_ratings: int, t80: int, t90: int):
    """Raise AssertionError if any split invariant is violated."""
    for scheme in SCHEMES:
        col = f"split_{scheme}"
        bad = splits.filter(~F.col(col).isin("train", "val", "test")).count()
        assert bad == 0, f"{col}: {bad} rows with an unknown split label"
    total = splits.count()
    assert total == n_ratings, f"split rows {total} != input rows {n_ratings}"

    # Per-user time order: train <= val <= test, core <= tail. Equal timestamps are allowed
    # at a boundary (a same-second burst cut by the movieId tiebreak).
    ts = F.col("timestamp")
    per_user = splits.groupBy("userId").agg(
        F.max(F.when(F.col("split_user") == "train", ts)).alias("train_max"),
        F.min(F.when(F.col("split_user") == "val", ts)).alias("val_min"),
        F.max(F.when(F.col("split_user") == "val", ts)).alias("val_max"),
        F.min(F.when(F.col("split_user") == "test", ts)).alias("test_min"),
        F.max(F.when(F.col("train_part") == "core", ts)).alias("core_max"),
        F.min(F.when(F.col("train_part") == "tail", ts)).alias("tail_min"),
        F.count("*").alias("n"),
        F.count(F.when(F.col("split_user") != "train", 1)).alias("n_eval"),
    )
    overlap = per_user.filter(
        (F.col("train_max") > F.col("val_min")) | (F.col("val_max") > F.col("test_min"))
        | (F.col("train_max") > F.col("test_min")) | (F.col("core_max") > F.col("tail_min"))
    ).count()
    assert overlap == 0, f"{overlap} users have train/val/test (or core/tail) time overlap"
    small = per_user.filter((F.col("n") < MIN_RATINGS_FOR_EVAL) & (F.col("n_eval") > 0)).count()
    assert small == 0, f"{small} users with < {MIN_RATINGS_FOR_EVAL} ratings have val/test rows"

    part_ok = splits.filter(
        ((F.col("split_user") == "train") & F.col("train_part").isNull())
        | ((F.col("split_user") != "train") & F.col("train_part").isNotNull())).count()
    assert part_ok == 0, f"{part_ok} rows with an inconsistent train_part"

    g = splits.groupBy("split_global").agg(F.min(ts).alias("lo"), F.max(ts).alias("hi"))
    bounds = {r["split_global"]: (r["lo"], r["hi"]) for r in g.collect()}
    assert bounds.get("train", (0, t80))[1] <= t80
    assert "val" not in bounds or t80 < bounds["val"][0] <= bounds["val"][1] <= t90
    assert "test" not in bounds or bounds["test"][0] > t90


# ---------------------------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------------------------

def split_stats(splits: DataFrame, scheme: str):
    """Row counts, evaluable users and a content fingerprint for one split scheme."""
    col = f"split_{scheme}"
    stats = {}
    counts = {r[col]: r["count"] for r in splits.groupBy(col).count().collect()}
    for s in ("train", "val", "test"):
        stats[f"rows_{s}"] = counts.get(s, 0)

    pos = F.col("rating") >= POSITIVE_THRESHOLD
    users = splits.groupBy("userId").agg(
        F.count(F.when((F.col(col) == "train") & pos, 1)).alias("train_pos"),
        F.count(F.when((F.col(col) == "val") & pos, 1)).alias("val_pos"),
        F.count(F.when((F.col(col) == "test") & pos, 1)).alias("test_pos"),
    ).cache()
    n_users = users.count()
    eligible = users.filter(F.col("train_pos") > 0)
    n_eligible = eligible.count()
    stats["users_total"] = n_users
    stats["users_eligible"] = n_eligible  # >= 1 train positive: the only users ever evaluated
    stats["users_excluded_no_train_positive"] = n_users - n_eligible
    stats["users_eligible_with_val_positive"] = eligible.filter(F.col("val_pos") > 0).count()
    stats["users_eligible_with_test_positive"] = eligible.filter(F.col("test_pos") > 0).count()
    users.unpersist()

    # Order-independent fingerprint of which rating landed in which slice: identical across
    # re-runs iff the split is identical.
    label = F.when(F.col(col) == "train", 1).when(F.col(col) == "val", 2).otherwise(3)
    h = F.pmod(F.xxhash64("userId", "movieId", label), F.lit(1_000_000_007))
    stats["fingerprint"] = splits.select(F.sum(h).alias("f")).first()["f"]
    return stats


def burst_stats(splits: DataFrame):
    """How often a user's test slice (and their train/val boundary) sits inside one sitting."""
    ts = F.col("timestamp")
    test = splits.filter(F.col("split_user") == "test")
    span = test.groupBy("userId").agg(((F.max(ts) - F.min(ts)) / 3600.0).alias("span_h")).cache()
    n = span.count()
    out = {
        "test_span_le_1h_share": span.filter(F.col("span_h") <= 1).count() / n,
        "test_span_le_24h_share": span.filter(F.col("span_h") <= 24).count() / n,
        "test_span_median_hours": span.select(F.expr("percentile(span_h, 0.5)")).first()[0],
    }
    span.unpersist()
    # Train/val cut inside a same-timestamp burst: order there is set by movieId, not by time.
    b = splits.groupBy("userId").agg(
        F.max(F.when(F.col("split_user") == "train", ts)).alias("train_max"),
        F.min(F.when(F.col("split_user") == "val", ts)).alias("val_min"))
    b = b.filter(F.col("val_min").isNotNull())
    out["users_train_val_cut_within_same_second"] = b.filter(
        F.col("train_max") == F.col("val_min")).count()
    out["users_train_val_cut_within_1h_share"] = (
        b.filter(F.col("val_min") - F.col("train_max") <= 3600).count() / b.count())
    return out


def catalog_stats(splits: DataFrame, movies: DataFrame, genome: DataFrame | None):
    train_items = splits.filter(F.col("split_user") == "train").select("movieId").distinct()
    later_items = splits.filter(F.col("split_user") != "train").select("movieId").distinct()
    out = {
        "movies_total": movies.count(),
        "movies_rated": splits.select("movieId").distinct().count(),
        # Coverage denominator (fixed for the whole project): movies with >= 1 train rating.
        "movies_with_train_rating": train_items.count(),
        "movies_only_in_val_or_test": later_items.join(train_items, "movieId", "left_anti").count(),
    }
    if genome is not None:
        out["movies_with_genome"] = genome.select("movieId").distinct().count()
    return out


# ---------------------------------------------------------------------------------------------
# Features: computed from ONE training set at a time, never from val/test rows
# ---------------------------------------------------------------------------------------------

TRAIN_SETS = {
    # name -> SQL filter over data/splits.parquet. See §1.5 / §2.1 for which set feeds which step.
    # (SQL strings, not Columns: Columns can't be built before a SparkSession exists.)
    "train_core": "train_part = 'core'",               # ranker training
    "train": "split_user = 'train'",                   # validation scoring
    "train_val": "split_user IN ('train', 'val')",     # final test scoring
}


def item_features(train: DataFrame) -> DataFrame:
    pos = F.col("rating") >= POSITIVE_THRESHOLD
    return train.groupBy("movieId").agg(
        F.count("*").alias("n_ratings"),
        F.count(F.when(pos, 1)).alias("n_positives"),
        F.avg("rating").alias("mean_rating"),
        F.min("timestamp").alias("first_rated_ts"),
        F.max("timestamp").alias("last_rated_ts"),
    )


def user_features(train: DataFrame) -> DataFrame:
    pos = F.col("rating") >= POSITIVE_THRESHOLD
    return train.groupBy("userId").agg(
        F.count("*").alias("n_ratings"),
        F.count(F.when(pos, 1)).alias("n_positives"),
        F.avg("rating").alias("mean_rating"),
        F.max("timestamp").alias("last_rated_ts"),
    )


def user_genre_affinity(train: DataFrame, movies: DataFrame) -> DataFrame:
    """Share of each user's train positives that carry each genre (long format)."""
    genres = (movies
              .select("movieId", F.explode(F.split("genres", r"\|")).alias("genre"))
              .filter(F.col("genre") != "(no genres listed)"))
    pos = train.filter(F.col("rating") >= POSITIVE_THRESHOLD).select("userId", "movieId")
    n_pos = pos.groupBy("userId").agg(F.count("*").alias("n_pos"))
    return (pos.join(genres, "movieId")
               .groupBy("userId", "genre").agg(F.count("*").alias("n"))
               .join(n_pos, "userId")
               .select("userId", "genre", (F.col("n") / F.col("n_pos")).alias("share")))


def genome_vectors(genome: DataFrame) -> DataFrame:
    """movieId -> array of 1,128 relevance scores, ordered by tagId.

    Note: GroupLens computed the genome in 2019 from all tags and ratings, so it carries some
    information from after any per-user cutoff. It is item content, not user behaviour, and is
    used as-is; this is stated as a limitation in the write-up.
    """
    return (genome.groupBy("movieId")
                  .agg(F.sort_array(F.collect_list(F.struct("tagId", "relevance"))).alias("t"))
                  .select("movieId", F.col("t.relevance").alias("genome")))


def build_features(splits: DataFrame, movies: DataFrame, out_dir: Path):
    for name, cond in TRAIN_SETS.items():
        train = splits.filter(cond)
        d = out_dir / name
        item_features(train).write.mode("overwrite").parquet(str(d / "items.parquet"))
        user_features(train).write.mode("overwrite").parquet(str(d / "users.parquet"))
        user_genre_affinity(train, movies).write.mode("overwrite").parquet(
            str(d / "user_genres.parquet"))


# ---------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------

def write_stats(rows, path: Path):
    """rows: iterable of (scheme, stat, value). Sorted and fixed-format so re-runs diff cleanly."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["scheme", "stat", "value"])
        for scheme, stat, value in sorted(rows, key=lambda r: (r[0], r[1])):
            w.writerow([scheme, stat, f"{value:.6f}" if isinstance(value, float) else value])


def run(spark, raw_dir: Path, out_dir: Path, stats_path: Path, with_genome=True):
    t0 = time.time()
    ratings = read_csv(spark, raw_dir / "ratings.csv", RATINGS_SCHEMA)
    movies = read_csv(spark, raw_dir / "movies.csv", MOVIES_SCHEMA)
    genome = (read_csv(spark, raw_dir / "genome-scores.csv", GENOME_SCHEMA)
              if with_genome else None)
    n_ratings = ratings.count()
    null_rows = ratings.filter(F.col("userId").isNull() | F.col("movieId").isNull()
                               | F.col("rating").isNull() | F.col("timestamp").isNull()).count()
    assert null_rows == 0, f"{null_rows} ratings rows with nulls"

    splits, t80, t90 = build_splits(ratings)
    splits_path = out_dir / "splits.parquet"
    splits.write.mode("overwrite").parquet(str(splits_path))
    splits = spark.read.parquet(str(splits_path)).cache()
    print(f"splits written ({time.time() - t0:.0f}s)")

    check_splits(splits, n_ratings, t80, t90)
    print("split checks passed")

    rows = [("all", "ratings_total", n_ratings),
            ("global", "t80_timestamp", t80), ("global", "t90_timestamp", t90),
            ("user", "users_lt10_train_only",
             splits.groupBy("userId").count().filter(F.col("count") < MIN_RATINGS_FOR_EVAL).count()),
            ("user", "rows_train_core", splits.filter(F.col("train_part") == "core").count()),
            ("user", "rows_train_tail", splits.filter(F.col("train_part") == "tail").count())]
    for scheme in SCHEMES:
        rows += [(scheme, k, v) for k, v in split_stats(splits, scheme).items()]
    rows += [("user", k, v) for k, v in burst_stats(splits).items()]
    rows += [("all", k, v) for k, v in catalog_stats(splits, movies, genome).items()]
    for scheme in SCHEMES:
        s = {k: v for sc, k, v in rows if sc == scheme}
        assert s["rows_train"] + s["rows_val"] + s["rows_test"] == n_ratings, scheme
    write_stats(rows, stats_path)
    print(f"stats written to {stats_path} ({time.time() - t0:.0f}s)")

    build_features(splits, movies, out_dir / "features")
    if genome is not None:
        genome_vectors(genome).write.mode("overwrite").parquet(
            str(out_dir / "features" / "genome.parquet"))
    print(f"features written ({time.time() - t0:.0f}s)")
    splits.unpersist()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--raw", default="data/ml-25m", type=Path)
    p.add_argument("--out", default="data", type=Path)
    p.add_argument("--stats", default="results/data_stats.csv", type=Path)
    p.add_argument("--driver-memory", default=os.environ.get("SPARK_DRIVER_MEMORY", "8g"))
    args = p.parse_args()
    spark = get_spark(driver_memory=args.driver_memory)
    spark.sparkContext.setLogLevel("ERROR")
    try:
        run(spark, args.raw, args.out, args.stats)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

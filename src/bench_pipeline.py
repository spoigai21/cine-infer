"""Phase 9: the Phase 1 feature pipeline in four engines (pandas, Spark, Polars, DuckDB).

The workload (identical outputs in every engine; tests/test_bench_pipeline.py checks it):
  splits       per-user time split: order by (timestamp, movieId); n < 10 -> all train; else
               train = first (n*8)//10, val up to (n*9)//10, test the rest; train further cut
               into core = first (n_train*7)//8 and tail (Phase 1 §1.2, §1.3a)
  items        on train: n_ratings, n_positives (rating >= 4), mean_rating, first/last ts
  users        on train: n_ratings, n_positives, mean_rating, last ts
  user_genres  on train positives: share of each user's positives carrying each genre
               ("(no genres listed)" excluded)
All four tables are written to Parquet under --out. Spark runs the actual Phase 1 functions
(src/data_prep.py).

Timing, one fresh process per (engine, size, repeat), printed as one JSON line:
  startup_s        import the engine (+ JVM and SparkSession for Spark)
  cold_compute_s   first pipeline run: read input -> write all outputs
  warm_compute_s   second run in the same process (JVM warmed up, caches populated)
Prediction #5 compares "with startup" = startup + cold compute, and "without startup" = warm
compute (the most Spark-favourable reading). Defined before any timing was run.

Threads: Spark local[N]; Polars and DuckDB capped at N threads; pandas is single-threaded.
Usage: python -m src.bench_pipeline --engine {pandas,spark,polars,duckdb} --input ratings.parquet
          --movies movies.csv --out DIR [--threads 6]
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

POS = 4.0
NO_GENRE = "(no genres listed)"


# ---------------------------------------------------------------------------------------------
# pandas
# ---------------------------------------------------------------------------------------------

def run_pandas(inp, movies_csv, out):
    import numpy as np
    import pandas as pd
    r = pd.read_parquet(inp, columns=["userId", "movieId", "rating", "timestamp"])
    r = r.sort_values(["userId", "timestamp", "movieId"], kind="stable", ignore_index=True)
    g = r.groupby("userId", sort=False)
    rn = g.cumcount().to_numpy() + 1
    n = g["userId"].transform("size").to_numpy()
    n_train = np.where(n < 10, n, (n * 8) // 10)
    n_val_end = np.where(n < 10, n, (n * 9) // 10)
    n_core = (n_train * 7) // 8
    split = np.where(rn <= n_train, "train", np.where(rn <= n_val_end, "val", "test"))
    part = np.where(split != "train", None, np.where(rn <= n_core, "core", "tail"))
    r["split_user"], r["train_part"] = split, part
    train = r[r.split_user == "train"]
    pos = train.rating >= POS
    items = train.assign(p=pos).groupby("movieId").agg(
        n_ratings=("rating", "size"), n_positives=("p", "sum"), mean_rating=("rating", "mean"),
        first_rated_ts=("timestamp", "min"), last_rated_ts=("timestamp", "max")).reset_index()
    users = train.assign(p=pos).groupby("userId").agg(
        n_ratings=("rating", "size"), n_positives=("p", "sum"), mean_rating=("rating", "mean"),
        last_rated_ts=("timestamp", "max")).reset_index()
    m = pd.read_csv(movies_csv, usecols=["movieId", "genres"])
    genres = m.assign(genre=m.genres.str.split("|")).explode("genre")[["movieId", "genre"]]
    genres = genres[genres.genre != NO_GENRE]
    tp = train.loc[pos, ["userId", "movieId"]]
    n_pos = tp.groupby("userId").size().rename("n_pos")
    ug = (tp.merge(genres, on="movieId").groupby(["userId", "genre"]).size().rename("n")
          .reset_index().join(n_pos, on="userId"))
    ug["share"] = ug.n / ug.n_pos
    out.mkdir(parents=True, exist_ok=True)
    r.to_parquet(out / "splits.parquet", index=False)
    items.to_parquet(out / "items.parquet", index=False)
    users.to_parquet(out / "users.parquet", index=False)
    ug[["userId", "genre", "share"]].to_parquet(out / "user_genres.parquet", index=False)


# ---------------------------------------------------------------------------------------------
# Polars
# ---------------------------------------------------------------------------------------------

def run_polars(inp, movies_csv, out):
    import polars as pl
    r = (pl.scan_parquet(inp).select("userId", "movieId", "rating", "timestamp")
         .sort(["userId", "timestamp", "movieId"])
         .with_columns(rn=pl.int_range(1, pl.len() + 1).over("userId"),
                       n=pl.len().over("userId"))
         .with_columns(n_train=pl.when(pl.col("n") < 10).then(pl.col("n")).otherwise(pl.col("n") * 8 // 10),
                       n_val_end=pl.when(pl.col("n") < 10).then(pl.col("n")).otherwise(pl.col("n") * 9 // 10))
         .with_columns(n_core=pl.col("n_train") * 7 // 8)
         .with_columns(split_user=pl.when(pl.col("rn") <= pl.col("n_train")).then(pl.lit("train"))
                       .when(pl.col("rn") <= pl.col("n_val_end")).then(pl.lit("val"))
                       .otherwise(pl.lit("test")))
         .with_columns(train_part=pl.when(pl.col("split_user") != "train").then(pl.lit(None, pl.Utf8))
                       .when(pl.col("rn") <= pl.col("n_core")).then(pl.lit("core"))
                       .otherwise(pl.lit("tail")))
         .select("userId", "movieId", "rating", "timestamp", "split_user", "train_part")
         .collect())
    train = r.lazy().filter(pl.col("split_user") == "train")
    pos = pl.col("rating") >= POS
    items = train.group_by("movieId").agg(
        n_ratings=pl.len(), n_positives=pos.sum(), mean_rating=pl.col("rating").mean(),
        first_rated_ts=pl.col("timestamp").min(), last_rated_ts=pl.col("timestamp").max())
    users = train.group_by("userId").agg(
        n_ratings=pl.len(), n_positives=pos.sum(), mean_rating=pl.col("rating").mean(),
        last_rated_ts=pl.col("timestamp").max())
    genres = (pl.scan_csv(movies_csv).select("movieId", "genres")
              .with_columns(genre=pl.col("genres").str.split("|")).explode("genre")
              .filter(pl.col("genre") != NO_GENRE).select("movieId", "genre"))
    tp = train.filter(pos).select("userId", "movieId")
    n_pos = tp.group_by("userId").agg(n_pos=pl.len())
    ug = (tp.join(genres, on="movieId").group_by("userId", "genre").agg(n=pl.len())
          .join(n_pos, on="userId").select("userId", "genre", share=pl.col("n") / pl.col("n_pos")))
    out.mkdir(parents=True, exist_ok=True)
    r.write_parquet(out / "splits.parquet")
    items_df, users_df, ug_df = pl.collect_all([items, users, ug])
    items_df.write_parquet(out / "items.parquet")
    users_df.write_parquet(out / "users.parquet")
    ug_df.write_parquet(out / "user_genres.parquet")


# ---------------------------------------------------------------------------------------------
# DuckDB
# ---------------------------------------------------------------------------------------------

def run_duckdb(inp, movies_csv, out, con):
    out.mkdir(parents=True, exist_ok=True)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE splits AS
        WITH o AS (
          SELECT userId, movieId, rating, timestamp,
                 row_number() OVER (PARTITION BY userId ORDER BY timestamp, movieId) AS rn,
                 count(*) OVER (PARTITION BY userId) AS n
          FROM read_parquet('{inp}')),
        c AS (
          SELECT *, CASE WHEN n < 10 THEN n ELSE (n * 8) // 10 END AS n_train,
                    CASE WHEN n < 10 THEN n ELSE (n * 9) // 10 END AS n_val_end FROM o)
        SELECT userId, movieId, rating, timestamp,
               CASE WHEN rn <= n_train THEN 'train' WHEN rn <= n_val_end THEN 'val' ELSE 'test' END
                 AS split_user,
               CASE WHEN rn > n_train THEN NULL WHEN rn <= (n_train * 7) // 8 THEN 'core'
                    ELSE 'tail' END AS train_part
        FROM c""")
    con.execute(f"COPY splits TO '{out / 'splits.parquet'}' (FORMAT PARQUET)")
    con.execute(f"""COPY (SELECT movieId, count(*) AS n_ratings,
                                 count(*) FILTER (rating >= {POS}) AS n_positives,
                                 avg(rating) AS mean_rating, min(timestamp) AS first_rated_ts,
                                 max(timestamp) AS last_rated_ts
                          FROM splits WHERE split_user = 'train' GROUP BY movieId)
                    TO '{out / 'items.parquet'}' (FORMAT PARQUET)""")
    con.execute(f"""COPY (SELECT userId, count(*) AS n_ratings,
                                 count(*) FILTER (rating >= {POS}) AS n_positives,
                                 avg(rating) AS mean_rating, max(timestamp) AS last_rated_ts
                          FROM splits WHERE split_user = 'train' GROUP BY userId)
                    TO '{out / 'users.parquet'}' (FORMAT PARQUET)""")
    con.execute(f"""COPY (
        WITH g AS (SELECT movieId, unnest(string_split(genres, '|')) AS genre
                   FROM read_csv('{movies_csv}', header = true, quote = '"', escape = '"')),
             tp AS (SELECT userId, movieId FROM splits
                    WHERE split_user = 'train' AND rating >= {POS}),
             np AS (SELECT userId, count(*) AS n_pos FROM tp GROUP BY userId)
        SELECT tp.userId, g.genre, count(*)::DOUBLE / any_value(np.n_pos) AS share
        FROM tp JOIN g USING (movieId) JOIN np USING (userId)
        WHERE g.genre != '{NO_GENRE}' GROUP BY tp.userId, g.genre)
        TO '{out / 'user_genres.parquet'}' (FORMAT PARQUET)""")


# ---------------------------------------------------------------------------------------------
# Spark (the Phase 1 functions)
# ---------------------------------------------------------------------------------------------

def run_spark(inp, movies_csv, out, spark):
    from pyspark.sql import functions as F
    from src import data_prep as dp
    r = spark.read.parquet(str(inp)).select("userId", "movieId", "rating", "timestamp")
    s = dp.add_per_user_split(r).cache()
    train = s.filter(F.col("split_user") == "train")
    movies = dp.read_csv(spark, movies_csv, dp.MOVIES_SCHEMA)
    s.write.mode("overwrite").parquet(str(out / "splits.parquet"))
    dp.item_features(train).write.mode("overwrite").parquet(str(out / "items.parquet"))
    dp.user_features(train).write.mode("overwrite").parquet(str(out / "users.parquet"))
    dp.user_genre_affinity(train, movies).write.mode("overwrite").parquet(
        str(out / "user_genres.parquet"))
    s.unpersist()


# ---------------------------------------------------------------------------------------------
# Output comparison (tests, and the benchmark's correctness check on real data)
# ---------------------------------------------------------------------------------------------

TABLE_KEYS = {"splits": ["userId", "movieId"], "items": ["movieId"], "users": ["userId"],
              "user_genres": ["userId", "genre"]}


def load_outputs(d):
    """Engine outputs as sorted pandas frames with canonical dtypes."""
    import pandas as pd
    out = {}
    for t, keys in TABLE_KEYS.items():
        df = pd.read_parquet(Path(d) / f"{t}.parquet")
        df = df[sorted(df.columns)].sort_values(keys, ignore_index=True)
        for c in df.columns:
            if c in ("split_user", "train_part", "genre"):
                df[c] = df[c].astype(object).where(df[c].notna(), None)
            elif df[c].dtype.kind in "iu":
                df[c] = df[c].astype("int64")
            elif df[c].dtype.kind == "f":
                df[c] = df[c].astype("float64")
        out[t] = df
    return out


def compare_outputs(a, b, rtol=1e-9):
    """Raise AssertionError if two engines' outputs differ (floats compared with rtol)."""
    import numpy as np
    A, B = load_outputs(a), load_outputs(b)
    for t in TABLE_KEYS:
        x, y = A[t], B[t]
        assert list(x.columns) == list(y.columns), (t, list(x.columns), list(y.columns))
        assert len(x) == len(y), (t, len(x), len(y))
        for c in x.columns:
            if x[c].dtype.kind == "f" or y[c].dtype.kind == "f":
                assert np.allclose(x[c].to_numpy(float), y[c].to_numpy(float), rtol=rtol,
                                   atol=0, equal_nan=True), (t, c)
            else:
                assert (x[c].fillna("<NA>").to_numpy() == y[c].fillna("<NA>").to_numpy()).all(), (t, c)


# ---------------------------------------------------------------------------------------------
# Timing harness
# ---------------------------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--engine", required=True, choices=["pandas", "spark", "polars", "duckdb"])
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--movies", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--threads", type=int, default=6)
    a = p.parse_args(argv)
    inp, movies, out = a.input.resolve(), a.movies.resolve(), a.out.resolve()
    if a.engine == "polars":
        os.environ["POLARS_MAX_THREADS"] = str(a.threads)  # must be set before the import

    t0 = time.perf_counter()
    if a.engine == "pandas":
        import pandas  # noqa: F401
        fn = lambda o: run_pandas(inp, movies, o)
    elif a.engine == "polars":
        import polars  # noqa: F401
        fn = lambda o: run_polars(inp, movies, o)
    elif a.engine == "duckdb":
        import duckdb
        con = duckdb.connect()
        con.execute(f"SET threads = {a.threads}")
        fn = lambda o: run_duckdb(inp, movies, o, con)
    else:
        from src.data_prep import get_spark
        spark = (get_spark(app="cineinfer-bench", master=f"local[{a.threads}]"))
        spark.conf.set("spark.sql.files.maxPartitionBytes", str(16 * 1024 * 1024))
        spark.sparkContext.setLogLevel("ERROR")
        fn = lambda o: run_spark(inp, movies, o, spark)
    startup = time.perf_counter() - t0

    times = []
    for run in ("cold", "warm"):
        o = out / run
        if o.exists():
            shutil.rmtree(o)
        t = time.perf_counter()
        fn(o)
        times.append(time.perf_counter() - t)
    if a.engine == "spark":
        spark.stop()
    print(json.dumps({"engine": a.engine, "input": str(a.input), "threads": a.threads,
                      "startup_s": round(startup, 4), "cold_compute_s": round(times[0], 4),
                      "warm_compute_s": round(times[1], 4), "python": sys.version.split()[0]}))


if __name__ == "__main__":
    main()

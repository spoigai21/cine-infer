# CineInfer — Implementation Guide

Step-by-step build instructions for the plan in `cineinfer.md`. Each step has: what you're doing,
the commands, a code sketch, **how you know it worked**, and the edge cases that bite here.

> **Status of the code below:** Phase 0 is built and verified; its section describes the actual repo.
> From Phase 1 on, snippets are starting points written from the plan, **not yet executed**.
> Treat every snippet as a draft to verify, and trust the "how you know it worked" checks over the code.

**Golden rules for the whole project**

1. **Never tune on the test slice.** Validation is for choices; test is read once, at the end.
2. **Never commit the dataset** (MovieLens license) or any model weights over ~50 MB.
3. **Every number in the README comes from a results file**, never typed by hand.
4. **A step isn't done until its check passes.** No moving on with "probably fine".
5. **Set a seed everywhere** (`random`, `numpy`, `torch`) and log it with every result.

---

## Phase 0 — Setup

**Status: built and verified** (`make data`, `make check-java` and `make test` all pass). This
section describes what's actually in the repo.

### 0.1 Prerequisites (macOS)

- **Python 3.11** (`python3.11`). PySpark 3.5 and every dependency below support it.
- **Java 17 or 11 for Spark.** Spark 3.5 does **not** support Java 21, which is what SDKMAN and
  Homebrew install by default. Install a native arm64 JDK 17 with
  `/opt/homebrew/bin/brew install openjdk@17`. An x86_64 JDK works under Rosetta, but it's slower
  and will skew the Phase 9 timings.
- **`make` from the Command Line Tools.** If `make` fails with an `xcodebuild` error (on this
  machine, Xcode 27 doesn't run on macOS 26.6), point the developer tools at the Command Line
  Tools: `sudo xcode-select -s /Library/Developer/CommandLineTools`.
- **Docker** (Docker Desktop), only needed for the Compose stack.

### 0.2 Install dependencies

```bash
make install        # creates .venv (python3.11) and installs requirements.txt
make check-java     # starts a local Spark session with the selected JDK
```

`requirements.txt` pins `pyspark==3.5.*` and **`numpy<2`**, because PySpark 3.5's MLlib isn't
fully numpy-2 compatible. The API image has its own smaller `docker/requirements-api.txt`.

**Choosing the JDK.** The Makefile **ignores the `JAVA_HOME` in your shell** (SDKMAN sets it to
Java 21) and uses the first of these that exists: native Homebrew `openjdk@17`, SDKMAN
`17.0.5-tem`, Intel Homebrew `openjdk@17`, Intel Homebrew `openjdk@11`. Override with
`make SPARK_JAVA_HOME=/path/to/jdk <target>`. Any script that starts Spark outside `make` must set
`JAVA_HOME` the same way, or Spark fails with a confusing gateway error.

### 0.3 Folder layout

```
cine-infer/
  data/                     # MovieLens download + derived parquet (git-ignored)
  src/
    data_prep.py            # Phase 1: Spark cleaning + splits
    evaluate.py             # Phase 2: metrics, full-catalog ranking
    baselines.py            # Phase 3: popular, item-kNN, ALS, EASE
    two_tower.py            # Phase 5: PyTorch retrieval model
    ranker.py               # Phase 6: second-stage model
    serve.py                # FastAPI (Phase 0: /health only; /recommend in Phase 7)
  scripts/
    get_data.sh             # download + verify MovieLens 25M
    ml-25m.zip.sha256       # recorded checksum (committed; the data is not)
    make_fixture.py         # generates the synthetic test fixture
  tests/
    fixtures/tiny_ratings.csv, tiny_movies.csv   # synthetic, 200 ratings, committed
    test_setup.py           # Phase 0 checks
  dags/                     # Airflow DAGs (Phase 8)
  docker/                   # api.Dockerfile, requirements-api.txt
  results/                  # CSVs — committed, the source of truth for every number
  .github/workflows/ci.yml  # pytest on the fixture
  docker-compose.yml, Makefile, requirements.txt, pytest.ini, .env.example
```

Only `src/serve.py` exists so far; the other `src/` modules arrive in their phases.

### 0.4 .gitignore

Written before the first code commit. It ignores `data/`, `models/`, `*.pt`, `*.parquet`, `.venv/`,
`.env`, Python, pytest and Spark caches, and Airflow's runtime state (`airflow/logs/`,
`airflow/db/`). `tests/test_setup.py` asserts that `data/` stays ignored.

### 0.5 Download script (`make data`)

`scripts/get_data.sh` is safe to re-run:

- It skips the download if `data/ml-25m.zip` exists and passes both checksums. The **MD5 is the
  one GroupLens publishes** (`ml-25m.zip.md5`). The **SHA-256** was recorded on the first download
  in `scripts/ml-25m.zip.sha256`, which is committed.
- It downloads to `ml-25m.zip.part` and renames only after the download finishes, so an
  interrupted run never leaves a truncated zip that looks complete. A zip that fails its checksum
  is deleted and downloaded again.
- It re-extracts, then checks that `ratings.csv` has **25,000,096** lines and that `movies.csv`,
  `genome-scores.csv`, `genome-tags.csv`, `tags.csv` and `links.csv` exist.

### 0.6 Test fixture (`make fixture`)

`tests/fixtures/` is **synthetic, not a MovieLens sample**, because the license forbids
redistribution. It has the same schema. `scripts/make_fixture.py` (seeded, byte-identical on
re-run) builds 200 ratings from 9 users over 40 movies, with the edge cases later phases must
handle:

- a user with 8 ratings (Phase 1 routes them to train only)
- bursts of ratings sharing one timestamp (tests the `movieId` tiebreak)
- a user whose last 6 ratings are all below 4 (no relevant val/test items)
- users who start late in the timeline (cold users under the global-cutoff split)
- a movie with no ratings
- titles with commas and quotes

CI regenerates the fixture and fails if it differs from the committed copy.

### 0.7 Docker Compose (`make up` / `make down`)

| Service | Image | Port |
|---|---|---|
| `spark-master`, `spark-worker` | `apache/spark:3.5.7` (standalone cluster, 4 cores / 4 GB worker) | 8080 (UI), 7077 |
| `airflow` | `apache/airflow:2.10.5-python3.11`, `standalone` mode (SQLite) | **8081** (8080 is Spark's) |
| `api` | built from `docker/api.Dockerfile` | 8000 |

Development runs Spark in local mode from `.venv`. The cluster exists so the pipeline can run
unchanged against `spark://spark-master:7077`. `data/` is bind-mounted, never copied into an image
(`.dockerignore` excludes it).

**Airflow login.** Copy `.env.example` to `.env` (git-ignored) and set a password. `make up` and
`docker compose` refuse to start without it; there is no default password. `airflow standalone`
ignores the usual `_AIRFLOW_WWW_USER_*` variables and makes up a random password, so the service
creates the `.env` user itself before starting, and recreates it on every start so `.env` always
wins. The SQLite DB lives in `airflow/db/` on the bind mount, so it survives container rebuilds.
The REST API accepts basic auth (`curl -u user:pass localhost:8081/api/v1/dags`) so Phase 8
scripts can trigger DAG runs.

### 0.8 CI

`.github/workflows/ci.yml` runs on every push and PR: Python 3.11, Temurin Java 17, CPU-only torch
(skips the multi-GB CUDA wheels), fixture freshness check, then `pytest`. The real dataset is
never downloaded in CI.

**Done when:** `make data` reproduces `data/ml-25m/` from scratch with 25,000,096 lines in
`ratings.csv`, `git status` shows nothing under `data/`, and `make test` passes.

**Edge cases**
- `git status` must show no `data/` files. If it does, your `.gitignore` came too late — fix before committing.
- Files you'll use: `ratings.csv` (userId, movieId, rating, timestamp), `movies.csv`, `genome-scores.csv`, `genome-tags.csv`.
- Spark startup is about 5 s before any work. Phase 9 times it separately.

---

## Phase 1 — Data preparation and splits

This is the phase that decides whether your results mean anything. Slow down here.

**Status: built and verified.** `make prep` runs `src/data_prep.py` on the full data in about
100 s. All four "done when" checks pass: the split checks run on the real data every time, and
re-running produces a byte-identical `results/data_stats.csv`. `tests/test_data_prep.py`
(17 tests) compares every step against an independent pandas implementation on the fixture. The
snippets below are the original sketches; the code differs from them where noted here:

- **Output layout.** One table, `data/splits.parquet`, with one row per rating and one column per
  scheme: `split_user`, `train_part` (`core`/`tail`, only for train rows), `split_global`,
  `split_random`. Features go to `data/features/{train_core,train,train_val}/` as
  `items.parquet`, `users.parquet` and `user_genres.parquet` (long format: userId, genre, share),
  plus `data/features/genome.parquet` (movieId → 1,128 relevances ordered by tagId).
- **Cut points use integer arithmetic,** `(n * 8) div 10`, not `floor(n * 0.8)`. Floating-point
  error makes the latter off by one for some `n`.
- **Global cutoffs are exact percentiles** (Spark SQL `percentile`), not `approxQuantile`, so
  they're deterministic. They're still written to `data_stats.csv`.
- **The random split hashes `(seed, userId, movieId)`** instead of using `F.rand(seed)`, which
  changes with the partition layout and so with the machine's core count.
- **Extra statistics:** how close each user's train/val boundary falls in time, and how many
  movies appear only in val/test (cold items).
- **All three splits report eligible users** (≥1 train positive) and how many of them have a
  positive in val and in test. That's the population each metric is averaged over.
- **Tag genome caveat:** GroupLens computed it in 2019 from all the data, so it carries some
  post-cutoff information. It's item content and is used as-is, and this is listed as a limitation.

### 1.1 Start Spark (local mode)

```python
from pyspark.sql import SparkSession
spark = (SparkSession.builder
         .appName("cineinfer")
         .master("local[*]")                        # all CPU cores
         .config("spark.driver.memory", "8g")       # raise if you see OOM
         .config("spark.sql.shuffle.partitions", "64")  # default 200 is too many locally
         .getOrCreate())
ratings = spark.read.csv("data/ml-25m/ratings.csv", header=True, inferSchema=True)
```

### 1.2 Make the three time-ordered slices per user

For each user, sort their ratings oldest → newest, then cut **80% train / 10% validation / 10% test**.

```python
from pyspark.sql import functions as F, Window

w = Window.partitionBy("userId").orderBy("timestamp", "movieId")  # movieId breaks ties
r = (ratings
     .withColumn("rn", F.row_number().over(w))
     .withColumn("n", F.count("*").over(Window.partitionBy("userId"))))

r = r.withColumn("split",
      F.when(F.col("rn") <= F.floor(F.col("n") * 0.8), "train")
       .when(F.col("rn") <= F.floor(F.col("n") * 0.9), "val")
       .otherwise("test"))
r.write.mode("overwrite").parquet("data/splits_per_user")
```

**Why `movieId` in the ordering:** thousands of ratings share the exact same timestamp. Without a
tiebreaker the split is non-deterministic and your results won't reproduce.

### 1.3 Also build the comparison splits (needed for prediction #7)

```python
# global time cutoff: last 10% of the whole timeline is test, previous 10% is val
qs = ratings.approxQuantile("timestamp", [0.8, 0.9], 0.001)
t80, t90 = qs[0], qs[1]
glob = ratings.withColumn("split",
        F.when(F.col("timestamp") <= t80, "train")
         .when(F.col("timestamp") <= t90, "val").otherwise("test"))

# random split, for showing how much it inflates metrics
rand = ratings.withColumn("u", F.rand(seed=42)).withColumn("split",
        F.when(F.col("u") < 0.8, "train").when(F.col("u") < 0.9, "val").otherwise("test"))
```

Write `t80` and `t90` into `results/data_stats.csv` so the global split can be rebuilt exactly.
(As built: exact percentiles instead of `approxQuantile`, so recomputing gives the same cutoffs.)

**Cold-start users in the global split.** Anyone whose first rating is after `t80` has no train
data, and anyone whose first rating is after `t90` has neither train nor val data. Rule, applied to **all three
splits**: a user is evaluated only if they have at least one train positive (a rating ≥4, §1.6). Drop the others from that split's evaluation and **write the excluded count per
split to `results/data_stats.csv`**. The evaluated users therefore differ between splits; say so
next to the split-comparison table in the README, because it's part of why the numbers differ.

### 1.3a Ranker-label slice (inside train)

The ranker (Phase 6) needs its own labels, and they can't come from validation, because validation
is needed to tune the ranker. Carve the ranker's labels out of the **end of train**:

- `train_core` = the first 7/8 of each user's train rows (≈70% of the user's ratings)
- `train_tail` = the last 1/8 of each user's train rows (≈10%)

Use the same `(timestamp, movieId)` ordering as §1.2. `train_tail` exists only to train the ranker
(Phase 6). Every other model trains on the full train slice.

### 1.4 The rating-burst check

MovieLens timestamps are when someone *rated*, not when they *watched*. Many users rate in one
sitting, so their "most recent" ratings may be one undifferentiated batch.

```python
test = r.filter(F.col("split") == "test")
span = test.groupBy("userId").agg(
        ((F.max("timestamp") - F.min("timestamp")) / 3600).alias("span_hours"))
same_session = span.filter(F.col("span_hours") <= 1).count() / span.count()
```

Write `same_session` into `results/data_stats.csv` and mention it in the README.

### 1.5 Features (used later by the ranker)

Compute from the **train slice only**: movie popularity (rating count, mean rating), per-user genre
affinity, and join the tag genome (`genome-scores.csv`) for item features.

**Computing any feature over all splits leaks the future into training.** This is the single
easiest way to ruin the project without noticing.

Features are computed **once per training set**. Ranker training uses features from `train_core`,
validation scoring uses features from full train, and final test scoring uses features from
train + val (§2.1).

### 1.6 What counts as a positive (fixed for every model)

A rating **≥ 4.0 is a positive; ratings below 4.0 are dropped** from all model inputs: the item-kNN
and EASE matrices, the ALS labels, and the two-tower histories. They are not negatives or zeros.
Most-popular counts only ratings ≥ 4 as well. Every model sees the same binary interaction matrix,
so no model gets extra signal the others don't. (Low ratings still count as "seen" and are masked
at evaluation; see Phase 2.)

**Done when (all four must pass):**
1. `train + val + test` row counts sum to 25,000,095.
2. For every user: `max(train.timestamp) <= min(val.timestamp)` and `max(val.timestamp) <= min(test.timestamp)`. Write this as a pytest over the fixture and as a Spark assertion over the real data.
3. `results/data_stats.csv` exists with the burst number.
4. Re-running produces byte-identical split counts.

**Edge cases**
- **Users with few ratings:** every ML-25M user has ≥20, so 80/10/10 leaves ≥2 per slice. Still assert `n >= 10` and route anything smaller to train-only.
- **Empty test sets:** a user whose test slice has no rating ≥4 contributes nothing to Recall. Exclude them from the metric and **report how many you excluded**.
- **Movies only in val/test:** the model has never seen them. Keep them in the catalog but expect them to never be recommended; that's a real cold-item limitation to mention.
- **Memory:** if Spark dies with OOM, raise `spark.driver.memory` or run on a 1M-row sample while developing.
- **Titles contain commas** ("Amityville: A New Generation, The (1993)"). They're quoted; use a real CSV parser, never `line.split(",")`.

---

## Phase 2 — Evaluation harness (write this before any model)

```python
def top_k(scores, k):
    """Top-k item indices, ordered by (-score, item_id). Deterministic under ties."""
    k = min(k, len(scores))
    order = np.lexsort((np.arange(len(scores)), -scores))  # last key is primary
    return order[:k]

def evaluate(score_fn, eval_items_by_user, seen_items_by_user, k=10):
    """
    score_fn(user) -> array of scores over ALL items.
    eval_items_by_user: user -> set of held-out items rated >= 4 (val or test slice).
    seen_items_by_user: user -> every item the user rated (any rating) in the data the model
        was trained on: train when scoring val, train + val when scoring test (§2.1).
    """
    recalls, ndcgs, skipped_no_relevant, skipped_all_seen = [], [], 0, 0
    for user, relevant in eval_items_by_user.items():
        if not relevant:
            skipped_no_relevant += 1
            continue
        scores = score_fn(user).astype(np.float64)   # copy; never mutate the model's array
        scores[list(seen_items_by_user[user])] = -np.inf   # never recommend seen items
        if not np.isfinite(scores).any():
            skipped_all_seen += 1
            continue
        top = top_k(scores, k)
        top = top[np.isfinite(scores[top])]          # never "recommend" a masked item
        hits = [1 if i in relevant else 0 for i in top]
        recalls.append(sum(hits) / min(len(relevant), k))
        dcg = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
        ndcgs.append(dcg / idcg)
    return {"recall@10": np.mean(recalls), "ndcg@10": np.mean(ndcgs),
            "n_users": len(recalls),
            "skipped_no_relevant": skipped_no_relevant,
            "skipped_all_seen": skipped_all_seen}
```

(A full `lexsort` over 62k items per user is fine for correctness. If it's too slow, take the top
~k+100 with `argpartition` and `lexsort` only those, extending the window when the boundary score
ties.)

**Rules baked in**
- **Average per-user metrics with the mean, not the median.** Most users have 0–2 hits in the top
  10, so a per-user median is often exactly 0 for every model. Medians are for **across seeds**
  (§3 of the plan): compute the mean over users for each seed, then report the median and spread
  of those values over seeds.
- **Recall@10 is capped:** the denominator is `min(|relevant|, 10)`, so a user with 30 liked test
  movies can still reach 1.0. Call it "Recall@10 (capped)" in the README, since some papers
  divide by `|relevant|` and their numbers aren't directly comparable.
- Score **all** items, not a sample of 100. Sampled negatives can flip which model looks better (Krichene & Rendle, 2020).
- Mask every item the user rated (any rating) in the model's training data: train when scoring
  validation, train + val when scoring test. Pass that as `seen_items_by_user`.
- **AUC** = probability a held-out liked item outranks a random unrated item. Sample ~100 unrated items per user for this metric only, and say so.
- **Coverage** = distinct items appearing in anyone's top 10, divided by items with ≥1 train rating. State the denominator.

**Edge cases**
- `np.argpartition` with `k >= len(scores)` throws. `top_k` clamps `k`; keep the guard if you switch to the argpartition fast path.
- All-`-inf` scores (user has rated everything in the filtered catalog): skip the user and count it (`skipped_all_seen`).
- Fewer than k unmasked items: the `isfinite` filter drops masked items instead of padding the list with them.
- Ties in scores make ordering arbitrary. `top_k` sorts by `(-score, item_id)` so runs reproduce. Most-popular and EASE produce many ties, so this matters there.
- Report `skipped_no_relevant` and `skipped_all_seen` alongside every metric row.

### 2.1 Final test protocol (same for every model)

1. Tune hyperparameters on **validation**. The model trains on train, and evaluation masks train.
   Record the chosen config, including the epoch/iteration count for iterative models.
2. **Retrain on train + val** with that config fixed (no early stopping, same epoch count), and
   recompute train-only features on train + val.
3. Score the **test** slice once, masking train + val.

Every model, baselines included, follows these steps exactly. Retraining uses the most recent
behavior, which matters most under a time-ordered split. It's also what production does, and it's
the protocol used in the iALS re-evaluation (Rendle et al.).

---

## Phase 3 — Baselines (tuned, not token)

All four are scored the same way, ranking against the full catalog.

### 3.1 Most popular

Count positives (ratings ≥4, §1.6) per movie in train, and recommend the top 10 the user hasn't
rated. This is the floor.

### 3.2 Item-to-item similarity

Cosine similarity between movies over the user × item matrix (binarized: rated ≥4 = 1). Score a
user by summing similarities to the movies they liked.

### 3.3 Implicit ALS (Spark MLlib)

```python
from pyspark.ml.recommendation import ALS
als = ALS(userCol="userId", itemCol="movieId", ratingCol="label",
          implicitPrefs=True,      # NOT the default; the default optimizes rating prediction
          rank=64, regParam=0.05, alpha=20, maxIter=15,
          coldStartStrategy="drop", seed=42)
```

Input rows are **only the positives** (ratings ≥4), each with `label = 1.0`. Ratings below 4 are
dropped, not passed as `label = 0` (§1.6). Tune `rank`, `regParam`, `alpha` on **validation** only.

`coldStartStrategy="drop"` only affects `transform()`. For full-catalog scoring, compute
`userFactors @ itemFactors.T` yourself. Movies with no train positives have no factor, so give
them a score of `-inf`.

### 3.4 EASE (closed form, ~20 lines, very strong here)

```python
import numpy as np
def ease(X, lam=250.0):          # X: users × items, 1 = rated ≥4 (§1.6), scipy sparse
    G = (X.T @ X).toarray()
    G[np.diag_indices(G.shape[0])] += lam
    P = np.linalg.inv(G)
    B = -P / np.diag(P)
    B[np.diag_indices(B.shape[0])] = 0.0
    return B                      # scores = X @ B
```

**Edge case:** `G` is items × items. At 62k items that's 62k² floats ≈ **30 GB**, which will not
fit. Restrict to items with at least N ratings in train (N≈20 keeps roughly 10–15k items and makes
the matrix ~1 GB), and state the cutoff in the README. The dropped items are all but unrecommendable
anyway.

**Done when:** `results/baselines.csv` holds Recall@10, NDCG@10, AUC and coverage for all four,
each with its tuned hyperparameters and the validation score that chose them.

---

## Phase 4 — Commit predictions

Fill in the blanks in `cineinfer.md` §4 using the baseline numbers, then:

```bash
git add cineinfer.md && git commit -m "Predictions before neural training"
```

**Done when:** `git log` shows this commit **before** any two-tower training commit. That timestamp
is the entire point.

---

## Phase 5 — Two-tower retrieval model

### 5.1 Shape of it

- **Item tower:** `nn.Embedding(n_items, 64)`.
- **User tower:** mean of the embeddings of the movies that user liked in train, then a small MLP.
  History-based, **not** a user-ID embedding, so new history works without retraining.
- **Score:** dot product of the two vectors.

### 5.2 Training loop (in-batch negatives)

Each batch is a set of (user history, liked movie) pairs. The other movies in the batch act as
negatives. Cross-entropy over the batch, with **logQ correction**: subtract `log(sampling_prob)` of
each item from its logit, or popular movies get punished for being popular.

```python
# user_vecs: B × d, item_vecs: B × d, target_ids: B (movie index of each row's positive)
logits = user_vecs @ item_vecs.T                       # B × B
logits = logits - torch.log(item_prior[target_ids])    # logQ per column; item_prior from train counts
# The same movie can appear twice in a batch. That copy is a false negative, so mask it out
# (keeping each row's own diagonal entry).
dup = target_ids[:, None] == target_ids[None, :]
dup.fill_diagonal_(False)
logits = logits.masked_fill(dup, float("-inf"))
loss = F.cross_entropy(logits, torch.arange(len(logits), device=logits.device))
```

**Building the training pairs: history comes strictly before the target.** For a user's positive
at position t (in the `(timestamp, movieId)` order of §1.2), the history is that user's positives
at positions < t. The target must be excluded, and so must anything rated after it. Using the whole
train history minus the target lets the model learn from future ratings. The eval-time user vector
then comes from a history that has no future in it, so training and serving no longer match.
Truncate the history to the most recent N positives (N is a hyperparameter, e.g. 50) and skip
pairs with an empty history. Because of the rating bursts in §1.4, items sharing the target's
timestamp are "before" it only because of the movieId tiebreak. Keep that rule consistent and
mention it.

### 5.3 Practical settings

- Batch 1024+ (in-batch negatives need a big batch to be useful).
- Adam, lr 1e-3 to start; tune on validation.
- Device: `torch.device("mps" if torch.backends.mps.is_available() else "cpu")`.
- Seed everything; run ≥3 seeds and report the median.

**Done when:** it beats most-popular on validation NDCG@10 across 3 seeds, and
`results/two_tower.csv` records per-seed numbers plus the config.

**Edge cases**
- **Loss goes to 0 immediately:** you leaked the target into the input. Check that the user history excludes the target movie.
- **MPS numerical oddities:** if results look wrong on MPS, rerun on CPU to confirm; note the discrepancy if any.
- **Users whose train history is empty after filtering to ratings ≥4:** skip them in training and use the popular fallback at serving.
- **Embedding table size:** 62k × 64 floats ≈ 16 MB. Fine in memory. Keep all weights out of git anyway (they're regenerated by the pipeline); save to `models/` (git-ignored).

---

## Phase 6 — Ranker

Take the top ~200 candidates from the two-tower model, then re-score with richer features: tag
genome vector, movie popularity and mean rating, genre overlap with the user's history, recency.
A small MLP or gradient-boosted trees is enough.

### 6.1 Where the ranker's labels come from

The ranker **can't train on validation**, because validation is needed to tune it and to compare it
against retrieval alone. It trains on the `train_tail` slice from §1.3a:

1. Train a two-tower model on **`train_core` only**, with the tuned retrieval config.
2. For each user, retrieve the top ~200 from that model, masking `train_core` items.
3. Label each candidate 1 if it's a `train_tail` positive, else 0. Features come from
   `train_core` only (§1.5).
4. Fit the ranker on those rows, using a pointwise or pairwise loss.

The candidates in step 2 have to come from a model that **never saw `train_tail`**. If you use the
full-train retriever, the tail positives it was trained on rank artificially high, and the ranker
learns a distribution it will never see at serving time.

**Evaluation on validation:** full-train retriever → top ~200 (masking train) → ranker with
full-train features → top 10. **Test:** repeat §2.1: retrain the retriever on train + val and the
ranker on its shifted equivalent (retriever trained on train, labels from val), then score test
once. Report the recall ceiling (share of relevant items that land in the top 200) next to the
ablation, since the ranker can't recover what retrieval missed.

**Done when:** `results/ablation.csv` compares retrieval-only vs retrieval + ranking on validation,
then on test **once**. This is the clean stopping point.

**Expect a small gain.** MovieLens has few features; the tag genome is your best one. A small or
zero gain, reported honestly, is a fine result and matches prediction #3 in the plan.

---

## Phase 7 — Serving

```python
@app.get("/recommend/{user_id}")
def recommend(user_id: int, k: int = 10):
    if user_id not in user_index:          # cold start
        return {"items": popular_top_k(k), "strategy": "popularity_fallback"}
    ...
```

Measure p50/p99 over ≥1000 requests and write `results/latency.csv`.

**Start with a brute-force dot product** over 62k items (well under a millisecond). Only add FAISS
if you measure that it matters, and report both numbers.

---

## Phase 8 — Airflow

DAG: `prep → train → evaluate → publish_if_better`. `publish_if_better` compares the new model's
validation NDCG@10 to the live model's and refuses to publish if it's worse.

**Done when:** you deliberately publish a worse model (train 1 epoch) and show the DAG rejecting it.
That test is the deliverable, not the DAG itself.

---

## Phase 9 — pandas vs Spark vs Polars/DuckDB

Same feature pipeline, four implementations, at 1M / 5M / 25M rows.

**Time two things separately:**
- **startup** (Spark JVM boot, session creation) — several seconds, and it's not compute
- **compute** (the actual job)

Report both. Without that split, pandas looks better than it deserves at small sizes, and an
interviewer who knows Spark will catch it.

---

## Phase 10 — Write-up

README sections: what it is, how to run it, results table, predictions (confirmed / refuted), the
split comparison, limitations. Every number read from `results/*.csv` by a small script with a
`--check` mode that fails if the README and the CSVs disagree.

**State these limitations plainly:** offline metrics only, no A/B test; MovieLens timestamps are
rating time; no cold-start users exist in this data; the EASE item cutoff.

---

## Phase 11 — Narrated demo video (2–4 minutes)

Record with QuickTime (screen + microphone) or OBS. Suggested beats:

1. **(20s)** What it does: "given a user, top 10 movies they'd rate highly, trained on 25M ratings."
2. **(40s)** One command runs the pipeline; show the Airflow DAG or `make all` running.
3. **(40s)** Hit the API live, show the 10 movies and the latency number.
4. **(60s)** The results table: your model vs all four baselines, and what the comparison means.
5. **(30s)** One honest limitation out loud, e.g. "EASE is within X% of the neural model" or "offline only, no A/B test."

**Rules:** no cuts that hide a failure. If something breaks, fix it and re-record. Link the video at
the top of the README.

---

## Order of work, and where you can stop

```
0 → 1 → 2 → 3 → 4 → 5 → 6 → [clean stop] → 7 → 8 → 9 → 10 → 11
```

Phases 0–6 plus the write-up stand on their own: data pipeline, tuned baselines, a model trained
from scratch, and an honest comparison. If interest fades, stop there **with the write-up finished**
rather than leaving a half-built pipeline.

---

## The five mistakes that would sink this

1. **Tuning on test.** Everything downstream becomes a story instead of a measurement.
2. **Computing features over all splits.** Silent leakage; the numbers look great and mean nothing.
3. **Sampled-negative evaluation.** Can reverse the model ordering and invalidate the comparison.
4. **Untuned baselines.** "Beat ALS" is worthless if ALS ran on defaults with explicit feedback.
5. **Hand-typed numbers in the README.** They drift, and one wrong number discredits the rest.

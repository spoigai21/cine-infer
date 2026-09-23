# CineInfer — Plan

A movie recommender trained on 25 million real ratings. It learns what each user likes from their
past ratings and returns the 10 movies they're most likely to rate highly. Every claim comes from a
measured number, and every model has to beat a tuned simple baseline before it counts.

**Status:** Phases 0–2 built and verified (setup, data splits, features, evaluation harness). Phases 3–11 not started.
**Build guide:** step-by-step instructions in `cineinfer-implementation.md`.

---

## 1. What gets built

1. **Data prep (PySpark).** Turn raw ratings into training data and features: user history, movie
   popularity, genre affinity, recency.
2. **Baselines first, tuned as hard as the neural models:**
   - most-popular
   - item-to-item similarity ("people who liked X also liked Y")
   - **implicit-feedback ALS** (Spark MLlib, `implicitPrefs=True`)
   - **EASE** (~20 lines, closed form, very strong on MovieLens)
3. **Neural models (PyTorch, trained from scratch):**
   - **Two-tower retrieval model.** Picks ~200 candidates from 62k movies fast.
   - **Ranking model.** Re-orders those candidates using richer features (the tag genome is the
     strongest item feature available).
   The two-stage retrieve-then-rank design is what YouTube, TikTok and Pinterest run.
4. **Evaluation.** Tune on a validation slice, report once on a test slice, rank against the whole
   catalog, and compare every model on identical data.
5. **Serving.** A FastAPI endpoint: send a user ID, get back the top 10 movies. Latency measured,
   cold-start path defined.
6. **Orchestration.** An Airflow DAG reruns prep → train → evaluate → publish, and a model is only
   published if it beats the current one.
7. **Demo.** A short narrated walkthrough of the working system.

---

## 2. Dataset

**MovieLens 25M** (GroupLens, University of Minnesota), a free research dataset:

- 25,000,095 ratings from 162,541 users on 62,423 movies, spanning 1995–2019
- ratings from 0.5 to 5 stars, with timestamps
- movie titles and genres
- user-applied tags and a **tag genome** of movie traits (the main item-feature source for the ranker)

**Why this one:** real user behavior, big enough that data handling matters, and standard in
recommender research, so results can be compared against published numbers.

**Known properties to handle explicitly (all belong in the write-up):**

- **Licensing: never commit the data.** The MovieLens license forbids redistribution. Keep the
  download-script + checksum approach, and keep the files out of git.
- **Timestamps are rating time, not watch time.** Many users rate dozens of films in one sitting, so
  a user's "last N ratings" can all come from a single burst where internal order means little.
  **Measure how often this happens in Phase 1** and report it.
- **No cold-start users exist here.** Every user has ≥20 ratings. The API must still define what
  happens for an unknown user ID (fall back to most-popular).
- **Not every movie has ratings.** State whether catalog coverage divides by all 62,423 movies or
  only the rated subset, and keep that definition fixed.

**On Spark:** 25M rows (about 1 GB in memory) *can* fit in pandas on one machine, so "why Spark?" is
a fair question. Answer it with a measurement rather than a claim (Phase 9), and design the pipeline
so it runs unchanged on a larger dataset (e.g. Amazon Reviews 2023) as a stretch step.

---

## 3. Evaluation protocol (fixed BEFORE any training)

**Three time-ordered slices per user: train → validation → test.**
Every hyperparameter (ALS rank/regularization, EASE λ, embedding size, learning rate, number of
negatives) is tuned **only on validation**. The test slice is touched **once**, at the end. Tuning
against test is what turns "beats the baseline" into a meaningless claim.

**Split choice, named on purpose.** The per-user split stops a user's own future leaking into their
training data, but user A's 2008 test ratings are still predicted by a model that saw user B's 2015
ratings, including popularity signals that didn't exist in 2008. A **global time cutoff** (everything
after date T is test) is stricter. The plan uses the per-user split as primary and **measures all
three — per-user, global cutoff, random — to report how much each inflates metrics.** Under the
global cutoff, users who start rating after the cutoff have no train data. On every split, a user
is evaluated only if they have ≥1 train positive. The excluded count is reported per split, and the
comparison table notes that the evaluated users differ between splits.

**Final test run.** After tuning on validation, every model (baselines included) is **retrained on
train + validation** with its chosen config frozen, then scored once on test, masking train +
validation items.

**Ranker labels come from inside train.** The last ~1/8 of each user's train slice (`train_tail`)
is held out as the ranker's training labels. Its candidates come from a retriever trained only on
the earlier part of train, so validation stays free for tuning the ranker.

**One definition of a positive for all models.** A rating ≥4 is a positive. Ratings below 4 are
dropped from model inputs (not treated as negatives) but still count as "seen" for masking. ALS,
EASE, item-kNN, most-popular and the two-tower model all train on the same binary matrix.

**Ranking is against the full catalog.** Score every movie the user has not rated in training, and
filter already-rated movies out of the recommendations. Sampled negatives (ranking the true item
against 100 random ones) can reverse which model looks better — Krichene & Rendle (2020). At 62k
items, full-catalog scoring is cheap.

**Metrics:**

| Metric | What it answers |
|---|---|
| Recall@10 (capped) | of the movies the user actually liked, how many made the top 10; denominator is min(#liked, 10), so it's labeled "capped" and not compared directly to papers that divide by #liked |
| NDCG@10 | did the liked ones land near the top of the 10 |
| **AUC** | **ranking held-out liked items (rating ≥ 4) above unrated items sampled from the catalog** — not vs held-out low ratings |
| Catalog coverage | share of the catalog ever recommended (catches "just recommend blockbusters"); denominator defined in §2 |
| p50 / p99 serving latency | how fast one recommendation request is answered |

**Rules:**

- "Relevant" = a held-out rating of **4.0 or higher**.
- Every neural model is reported next to **all four baselines** on the same test slice.
- A model that doesn't beat the best tuned baseline is reported as a failure, not dropped.
- Per-user metrics are averaged with the **mean** over users (a per-user median is usually 0).
  Across seeds, report the **median and spread** of those means, not a single best run.
- Ties in scores are broken by `(-score, movieId)` so every run reproduces.
- Users with no relevant held-out items, or with nothing left to recommend, are skipped and
  counted, and the counts are reported.

---

## 4. Predictions to commit to git before training

Written down and committed first, then confirmed or refuted in the write-up. Fill in numeric ranges
after the baselines run, **before** the neural models train.

1. The two-tower model beats most-popular on NDCG@10.
2. **The best tuned baseline (EASE or implicit ALS) lands within ___% of the two-tower model on
   Recall@10.** Re-evaluations (Dacrema et al. 2019; Rendle et al. on iALS) found well-tuned simple
   models often match or beat neural ones on MovieLens. An honest "the neural model barely helped"
   is a real result, and tuning the baselines properly is what makes it credible.
3. Adding the ranking stage improves NDCG@10 over retrieval alone by ___ (may be small: MovieLens
   has few rich features, so the tag genome carries this stage).
4. Most-popular has the lowest catalog coverage of all models.
5. pandas beats PySpark on the feature pipeline below ___M rows; Spark wins above it.
6. Serving p99 latency stays under ___ ms for top-10 on a laptop CPU.
7. The random split overstates NDCG@10 versus the global time cutoff by ___%.

---

## 5. Build phases

| Phase | Output | Done when |
|---|---|---|
| 0. Setup | repo, `make` targets, Docker Compose (Spark, Airflow, API), dataset download script with checksum, data git-ignored | `make data` reproduces the dataset from scratch |
| 1. Data + splits | PySpark cleaning; train/validation/test slices per user (train further split into `train_core` / `train_tail` for the ranker), plus global-cutoff and random splits for comparison; rating-burst measurement; tests on a small fixture | row counts reconcile to 25,000,095; **zero train/val/test time overlap per user, asserted in a test**; burst statistic reported |
| 2. Evaluation harness | metric code (Recall@10, NDCG@10, AUC, coverage) with mean-over-users aggregation, full-catalog ranking, seen-item masking (train for val; train + val for test), deterministic `(-score, movieId)` tie-breaking | metrics match hand-computed values on a toy example, in a unit test |
| 3. Baselines | most-popular, item-kNN, implicit ALS, EASE | `results/baselines.csv` generated from code, not hand-typed; each baseline tuned on validation with its budget logged |
| 4. Predictions | fill in §4, commit | commit timestamped before any neural training |
| 5. Two-tower | PyTorch model; user tower built from rating history (pooled item embeddings), not a user-ID embedding; history for each training pair uses only positives rated before the target; in-batch negatives with logQ correction (Yi et al. 2019) and duplicate-item masking | beats most-popular; results over 3+ seeds; tuned only on validation |
| 6. Ranker | second-stage model over retrieved candidates, using tag-genome features; trained on `train_tail` labels with candidates from a `train_core`-only retriever | ablation: retrieval-only vs retrieval + ranking. **Clean stopping point: phases 0–6 plus the write-up stand on their own.** |
| 7. Serving | FastAPI + precomputed embeddings; brute-force dot product vs FAISS measured; cold-start fallback for unknown user IDs | load test reports p50/p99; unknown-user path returns popular items |
| 8. Orchestration | Airflow DAG: prep → train → evaluate → publish-if-better | a deliberately worse model is refused publication |
| 9. pandas vs Spark benchmark | timing at 1M / 5M / 25M, JVM startup and local-mode overhead timed separately from compute; one of Polars or DuckDB included | crossover measured and charted, with and without startup cost |
| 10. Write-up | README with every number generated from result files; predictions confirmed / refuted; split-comparison table | `--check` mode fails if the README drifts from the CSVs |
| 11. Demo | 2–4 minute narrated screen recording: one command runs the pipeline, the API returns recommendations with latency shown, the metrics table is read out against the baselines, one limitation stated out loud | the video plays start to finish with no cuts hiding failures, and the README links it |

**Stretch (only after 0–11):** run the unchanged pipeline on a larger dataset; containerize serving
on Kubernetes (kind) and load-test it.

---

## 6. Tech stack

| Area | Tools |
|---|---|
| Language | Python |
| Data | PySpark (local mode, then unchanged on a cluster); pandas + Polars or DuckDB for the benchmark |
| ML | PyTorch (two-tower, ranker), Spark MLlib ALS (implicit), EASE (numpy), scikit-learn (metrics) |
| Serving | FastAPI; brute-force dot product first, FAISS only if measured to matter at 62k items |
| Orchestration | Airflow |
| Infra | Docker Compose, Make, GitHub Actions CI on the fixture dataset |
| Testing | pytest; metric code checked against hand-computed examples; split-leakage assertions |
| Demo | screen recording with voice-over |

**Cost:** everything runs locally. Two-tower models at this size train fine on CPU or Apple MPS.
Rent a GPU only if a phase proves it's needed, and record the cost.

---

## 7. Known limitations to state in the write-up

- Offline metrics only. No A/B test, because that needs real users.
- MovieLens timestamps are rating time, not watch time.
- No cold-start users exist in this dataset; the cold-start path is designed but untested on real traffic.
- EASE runs on an item subset (items with ≥20 train ratings) because the full item×item matrix is too large.
- Results are specific to MovieLens; they do not automatically transfer to other domains.
- The tag genome was computed by GroupLens in 2019 from the full dataset, so ranker item features
  carry some information from after each user's train cutoff.
- Under the per-user split, most users' validation and test ratings come from the same rating
  session as the end of their training data (measured in `results/data_stats.csv`), so
  "predicting the future" is often "predicting the rest of one sitting".

# CineInfer

A movie recommender trained on {{data.ratings}} MovieLens ratings: given a user, it returns the 10
movies they're most likely to rate 4 stars or higher. It's built as a two-stage system (a PyTorch
two-tower retriever picks 200 candidates from {{data.movies}} movies; a LightGBM ranker re-orders
them) and served by a FastAPI endpoint.

The point of the project is honest measurement. Every model is compared with four baselines
tuned as hard as the neural models, the test slice was scored exactly once, and
[{{predictions.total_word}} predictions](#predictions-committed-before-training) were committed and pushed
([`predictions` tag](https://github.com/spoigai21/cine-infer/tree/predictions)) before any neural model was trained. {{predictions.refuted_Word}} of them
turned out wrong. **Every result in this README is generated from `results/*.csv`**
(`make readme`), and CI fails if they drift apart (`make readme-check`).

> Demo video: *coming in Phase 11.*

## Results (test slice, scored once)

{{test.users}} users, each ranking against the full catalog with everything they'd already rated
masked. A movie counts as relevant if the user rated it ≥ 4 in their held-out test period.

{{table:headline}}

- **The two-stage system beats every baseline**: {{gap.two_stage.ease.ndcg.rel}} NDCG@10 over
  the best baseline, EASE ({{gap.two_stage.ease.ndcg.diff}}, 95% CI {{gap.two_stage.ease.ndcg.ci}},
  paired bootstrap over users).
- **The ranker adds {{gap.two_stage.two_tower.ndcg.rel}}** over retrieval alone
  ({{gap.two_stage.two_tower.ndcg.diff}}, CI {{gap.two_stage.two_tower.ndcg.ci}}). It can only
  re-order what retrieval found: {{test.recall_ceiling_200}} of relevant test movies are in the
  200 candidates.
- **The two-tower model beats EASE by {{gap.two_tower.ease.ndcg.rel}}**
  ({{gap.two_tower.ease.ndcg.diff}}, CI {{gap.two_tower.ease.ndcg.ci}}) and recommends
  {{test.coverage_ratio.two_tower.ease}}× as much of the catalog ({{test.two_tower.coverage}} vs {{test.ease.coverage}} of the
  {{test.coverage_denominator}} movies with training data).
- Paired bootstrap on NDCG@10, {{gaps.chain}} (`results/test_gaps.csv`).

### Where the gains come from

MovieLens timestamps are *rating* time, and users rate in bursts: {{data.burst_1h}} of users rated
their whole test slice within one hour (median span {{data.burst_median_min}} minutes). So under a
per-user time split, the "future" is usually the rest of the same sitting. Breaking test results
down by the gap between a user's last training rating and their first test rating:

{{table:boundary}}

The two-tower model uses only a user's last 10 liked movies, so it's excellent at "what comes next
in this session" ({{boundary.two_tower.same_second}} vs EASE's {{boundary.ease.same_second}}) and
**worse than EASE when the test period starts more than an hour later**
({{boundary.two_tower.over_1h}} vs {{boundary.ease.over_1h}}). That isn't just a short input
window: feeding EASE only the last 10 movies (the control row) doesn't reproduce it. The ranker
sees both scores and learns when to trust each, so the two-stage system beats EASE in every
bucket, including over 1 h ({{boundary.two_stage.over_1h}}).

### Ranker ablations

{{table:ablation}}

- **EASE's score is the ranker's most valuable feature.** Without it the ranker keeps only
  {{ablation.no_ease_share}} of its gain over retrieval.
- **Two time features are excluded** even though they help. They compare a user's last rating
  with the last time *anyone* rated a movie, and under a per-user split "anyone" includes other
  users' ratings from after this user's cutoff: information a live system wouldn't have.

### Removing the cross-user time leak (validation, after the fact)

The headline's item features (a movie's popularity and mean rating) are computed over the whole
training set, which under a per-user split includes other users' ratings from *after* this user's
cutoff. A follow-up rebuilt them **point in time**: only ratings made strictly before the user's
last training rating, by anyone (`make pit-experiment`). The recomputed headline reproduces
Phase 6 exactly for every seed, so the comparison is like for like:

| Ranker features (validation, 3 seeds) | NDCG@10 | vs headline (95% CI) |
|---|---|---|
| Headline (full-train item stats) | {{pit.headline.ndcg}} | — |
| Point-in-time item stats | {{pit.pit.ndcg}} | {{pit.pit.diff}} {{pit.pit.ci}} |
| Point-in-time item stats + point-in-time recency | {{pit.pit_time.ndcg}} | {{pit.pit_time.diff}} {{pit.pit_time.ci}} |

The leak wasn't propping the headline up. Popularity as of the user's own moment is *more*
informative than popularity over the whole training period, and the leak-free recency features
recover what the excluded time features offered. This was designed after the test results were
known, so it's reported on validation only: scoring it on test would be a second look.

## Predictions (committed before training)

{{predictions.confirmed}} of {{predictions.total}} confirmed, {{predictions.refuted}} refuted.
Each was written with a "refuted if" condition, committed, tagged and pushed before any neural
model existed, and is settled by code from the result files, never by hand.

{{table:predictions}}

What the refutations say:
- **#2, #3:** I expected a well-tuned EASE to match the neural model and the ranker to add little.
  Both gains are real, but the boundary table shows much of the two-tower's comes from predicting
  the rest of a rating session.
- **#5:** local-mode Spark's overhead dominates at 1M and 5M rows, but single-threaded pandas scales
  worse, and Spark wins at 25M. DuckDB and Polars beat both at every size.
- **#6:** the typical request was fast, but the p99 missed the target by {{latency.p6.p99_miss}} ms.
- **#7:** the random split inflates NDCG@10 even more than predicted, but the global cutoff scored
  *above* the per-user split: its {{split.global.users}} test users are the heavy raters still
  active after {{split.global_cutoff_date}}, a different population.

## How the split choice changes the numbers

Same model (EASE, same config) and same metric, three ways to split the data:

{{table:splits}}

A random split lets the model train on a user's *later* ratings and test on earlier ones; it
reports {{split.ratio}}× (95% CI {{split.ratio_ci}}) the global cutoff's NDCG@10. This project
uses the per-user time split for everything else: no user's own future leaks into their
training data, and it keeps {{split.user.users}} users evaluable (the global cutoff keeps
{{split.global.users}}).

## Serving

`GET /recommend/{user_id}?k=10` retrieves 200 candidates by brute-force dot product over all
{{data.movies}} movies, builds 15 features, ranks with LightGBM and returns titles, scores and
per-stage timings. Unknown users get the most-popular list. The server is NumPy + LightGBM only
(no PyTorch) and loads the model bundle in {{latency.load_s}} s.

- **No training/serving skew:** on {{parity.users}} test users, the server's top-10 matches the
  batch pipeline's for {{parity.top10}}.
- **Latency ({{latency.p6.requests}} sequential HTTP requests, laptop CPU):** the run that settled prediction #6
  measured p50 {{latency.p6.p50}} ms and p99 {{latency.p6.p99}} ms (target < 25 ms: refuted).
  Afterwards, and reported separately, single-threaded serving removed contention between the
  libraries' thread pools: p50 {{latency.threads1.p50}} ms, p99 {{latency.threads1.p99_range}} ms
  (default threading: p99 {{latency.default.p99_range}} ms). The popularity fallback takes
  {{latency.fallback.p50}} ms.

## Retraining with Airflow

An Airflow DAG runs prep → train → evaluate → **publish only if better** than the live model:
validation NDCG@10 must beat it by more than {{airflow.margin}}, just above the measured run-to-run
noise, so a plain retrain can't win by chance. Publishing refits on train + val, rebuilds the
ranker, exports and smoke-tests a bundle, then swaps the served bundle atomically.

- **Refusal (the deliverable):** a deliberately worse {{airflow.epochs}}-epoch model (validation
  NDCG@10 {{airflow.candidate}} vs live {{airflow.live}}) was rejected, and publish was skipped
  (`results/airflow_reject_demo.txt`).
- **Publish, end to end on real data:** in a sandbox registry whose live model is that 1-epoch
  model, a tuned candidate ({{airflow.pub.candidate}} vs {{airflow.pub.live}}) was published and
  served, with the production registry left byte-identical (`results/airflow_publish_demo.txt`).

## pandas vs Spark vs Polars vs DuckDB

The Phase 1 data pipeline (per-user split and features), implemented four ways with identical
outputs, timed in fresh processes ({{bench.runs}} runs, on AC power, load ≤ {{bench.max_load}}).
Each cell is *with startup / without startup* (warm compute), median of 3:

{{table:benchmark}}

![benchmark](results/benchmark.png)

## How it was built

| Phase | What |
|---|---|
| 0–1 | Download with checksums; PySpark per-user time split (checked for zero leakage); features from train only |
| 2 | Evaluation harness: full-catalog ranking, seen-item masking, deterministic ties, bootstrap CIs |
| 3 | Four baselines tuned on validation (grids extend past edges; every trial logged) |
| 4 | Predictions committed, tagged and pushed |
| 5–6 | Two-tower retriever (PyTorch, Apple GPU) and LightGBM ranker, tuned on validation, 3 seeds |
| 6b | Every model refit on train + val and scored on test **once** (sealed results) |
| 7–9 | Serving, Airflow retraining gate, engine benchmark |

Tuning, on {{data.val_users}} validation users (test was never used for any choice):

{{table:tuning}}

## Running it

Requirements: macOS or Linux, Python 3.11, **Java 17** for Spark, `brew install libomp` on macOS
for LightGBM, ~15 GB free disk.

```bash
make install        # .venv with everything
make data           # download + verify MovieLens 25M (not redistributable, never committed)
make prep           # Phase 1: splits + features (~2 min)
make baselines      # Phase 3: tune the baselines on validation (~1.5 h, mostly ALS)
make two-tower      # Phase 5: tune the retriever (~3 h on an Apple GPU)
make ranker         # Phase 6: two-stage system on validation (~45 min)
make final-test     # Phase 6b: refit everything, score test once (~1.5 h)
make export serve   # Phase 7: build the serving bundle, run the API on :8000
make airflow-install airflow-reject-demo   # Phase 8
make benchmark      # Phase 9 (plug in, idle machine)
make split-comparison test-analysis readme  # Phase 10
make test           # the test suite, on a synthetic fixture (also in CI)
```

Times are rough guides for one laptop. Run one pipeline at a time: several steps use most of the CPU, and timing steps check the load.

## Limitations

- **Offline metrics only.** There's no A/B test, and no real users.
- **Timestamps are rating time, not watch time**, and most "future" ratings belong to the same
  sitting as the training data (the boundary table). The headline gains are largest there.
- **No cold-start users:** every MovieLens user has ≥ 20 ratings. The fallback path is designed
  and tested, not validated on real traffic. {{data.movies_cold}} movies appear only in val/test and
  can't be recommended by any model.
- **Cross-user time leakage under the per-user split:** a 2008 test rating is predicted by models
  that saw other users' 2015 ratings (the models themselves, and the headline ranker's popularity
  features). Point-in-time features remove it from the ranker and score higher on validation
  (above); the headline test numbers still use the original features.
- **The tag genome was computed by GroupLens in 2019 from all the data**, so ranker features built
  from it carry some post-cutoff information (only {{data.movies_genome}} movies have one).
- **Compute caps:** ALS rank and the two-tower embedding size were capped at 256; EASE and
  item-kNN use only movies with ≥ 20 training positives.
- **One machine:** latency is one client on a laptop; the Spark benchmark is local mode.
- **Results are specific to MovieLens.**

## Repository map

`src/` pipeline code (data prep, evaluation harness, models, serving, retraining) · `tests/` pytest
on a synthetic fixture · `results/` every number above, as CSV · `dags/` Airflow ·
`cineinfer.md` the plan and predictions · `cineinfer-implementation.md` the build log, phase by
phase. Data: [MovieLens 25M](https://grouplens.org/datasets/movielens/25m/) (GroupLens; downloaded
by `make data`, not redistributed).

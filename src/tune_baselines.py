"""Phase 3: tune the four baselines on VALIDATION and record the winners.

Protocol (the test slice is never read here):
  1. Every config is fit on train positives and scored on a fixed 20k-user subsample of the
     validation population (seeded). The selection metric is NDCG@10.
  2. Grids extend automatically when the best value sits on a grid edge, so a baseline is never
     cut off just short of its optimum.
  3. The winning config is scored on the FULL validation population (151,597 users). ALS, the
     only seeded baseline, is re-fit with 3 seeds; the rest are deterministic.
  4. results/tuning/baselines_trials.csv logs every trial (the tuning budget);
     results/baselines.csv holds each winner's full-validation metrics, config, the subsample
     score that chose it, the trial count and the tuning time.

Test-slice numbers come later, in one final run of every model with train + val refits (§2.1).

Resumable: each trial is appended to the trials CSV as soon as it finishes, and a re-run skips any
config already there (its logged score is reused), so an interrupted run loses at most the trial
in progress. `--fresh` ignores logged trials. Spark runs on `--spark-cores` cores (default 6,
not all 10: ALS at full load heats a laptop quickly).

Usage: python -m src.tune_baselines --models most_popular,item_knn,ease,implicit_als
"""
import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import baselines as bl
from src import evaluate as ev

SPLITS = Path("data/splits.parquet")
MOVIES = Path("data/ml-25m/movies.csv")
BASELINES_CSV = Path("results/baselines.csv")
TRIALS_CSV = Path("results/tuning/baselines_trials.csv")
PER_USER_DIR = Path("data/eval_runs")
TUNE_USERS, TUNE_SEED = 20_000, 0
SELECT = "ndcg@10"
ALS_SEEDS = (42, 43, 44)

TRIAL_COLUMNS = ["model", "trial", "config", "ndcg@10", "recall@10", "auc", "coverage",
                 "n_users", "fit_seconds", "eval_seconds", "source"]
BASELINE_COLUMNS = ev.RESULT_COLUMNS + ["selection_metric", "selection_value_tune_subsample",
                                        "n_trials", "tuning_seconds"]


class Tuner:
    """Runs trials for one model, logs each to disk immediately, remembers the best."""

    def __init__(self, name, tune_data, resume=True):
        self.name, self.tune, self.trials = name, tune_data, []
        self.best = None  # (score, config, fitted model or None if it came from the log)
        self.t0 = time.time()
        if resume and TRIALS_CSV.exists():
            logged = pd.read_csv(TRIALS_CSV)
            for r in logged[logged.model == name].to_dict("records"):
                self.trials.append(r)
                if self.best is None or r[SELECT] > self.best[0]:
                    self.best = (r[SELECT], json.loads(r["config"]), None)
            if self.trials:
                print(f"  resuming {name}: {len(self.trials)} logged trials", flush=True)

    def trial(self, build, config):
        key = json.dumps(config, sort_keys=True)
        for t in self.trials:
            if t["config"] == key:
                return t[SELECT]
        t = time.time()
        model = build()
        fit_s = time.time() - t
        t = time.time()
        s = ev.evaluate(model, self.tune).summary
        eval_s = time.time() - t
        row = {"model": self.name, "trial": len(self.trials) + 1, "config": key,
               **{m: s[m] for m in ("ndcg@10", "recall@10", "auc", "coverage", "n_users")},
               "fit_seconds": round(fit_s, 1), "eval_seconds": round(eval_s, 1),
               "source": "run"}
        self.trials.append(row)
        ev.replace_model_rows(self.trials, TRIALS_CSV, self.name, TRIAL_COLUMNS)
        print(f"  [{self.name} #{row['trial']}] {key}  ndcg={s['ndcg@10']:.5f} "
              f"recall={s['recall@10']:.5f} auc={s['auc']:.4f} cov={s['coverage']:.4f} "
              f"(fit {fit_s:.0f}s, eval {eval_s:.0f}s)", flush=True)
        if self.best is None or s[SELECT] > self.best[0]:
            self.best = (s[SELECT], config, model)
        else:
            del model
            gc.collect()
        return s[SELECT]

    def best_model(self, build_from_config):
        """The best fitted model, refitting it if the best trial came from the log."""
        score, config, model = self.best
        if model is None:
            print(f"  refitting best logged config {config}", flush=True)
            model = build_from_config(config)
            self.best = (score, config, model)
        return model

    def search_1d(self, build_for, values, key, fixed, grow=2.0, max_extend=4):
        """Try `values` for `key`; if the best is on an edge, extend geometrically."""
        values = sorted(_clean(v) for v in values)
        scores = {v: self.trial(lambda v=v: build_for(v), {**fixed, key: v}) for v in values}
        for _ in range(max_extend):
            best_v = max(scores, key=scores.get)
            vs = sorted(scores)
            if best_v == vs[-1]:
                nxt = vs[-1] * grow
            elif best_v == vs[0] and vs[0] / grow >= self._floor(key):
                nxt = vs[0] / grow
            else:
                break
            nxt = _clean(type(values[0])(nxt))
            if nxt in scores:
                break
            scores[nxt] = self.trial(lambda v=nxt: build_for(v), {**fixed, key: nxt})
        return max(scores, key=scores.get)

    @staticmethod
    def _floor(key):
        return {"k": 5, "rank": 8, "lam": 1.0, "alpha": 0.1, "reg": 1e-4}.get(key, 0)

    def seconds(self):
        """Tuning budget: fit + eval time of every trial, across interrupted and resumed runs."""
        return float(sum(t["fit_seconds"] + t["eval_seconds"] for t in self.trials))


def _clean(v):
    """Round grid values to 6 significant digits: 0.1 * 3 is logged as 0.3, not 0.30000000000000004."""
    return v if isinstance(v, int) else float(f"{v:.6g}")


def record(tuner, full_val, models_by_seed):
    """Score the winner(s) on full validation and write the result rows."""
    rows = []
    for seed, model in models_by_seed:
        t = time.time()
        res = ev.evaluate(model, full_val)
        print(f"  full validation (seed {seed}): ndcg={res.summary['ndcg@10']:.5f} "
              f"recall={res.summary['recall@10']:.5f} auc={res.summary['auc']:.4f} "
              f"cov={res.summary['coverage']:.4f} ({time.time() - t:.0f}s)", flush=True)
        ev.save_per_user(res, PER_USER_DIR / f"{tuner.name}_user_val_seed{seed}.parquet")
        row = ev.result_row(tuner.name, seed, model.config, res, full_val)
        row.update({"selection_metric": SELECT, "selection_value_tune_subsample": tuner.best[0],
                    "n_trials": len(tuner.trials), "tuning_seconds": round(tuner.seconds(), 1)})
        rows.append(row)
    ev.replace_model_rows(rows, BASELINES_CSV, tuner.name, BASELINE_COLUMNS)
    ev.replace_model_rows(tuner.trials, TRIALS_CSV, tuner.name, TRIAL_COLUMNS)


def tune_most_popular(td, tune, full, resume=True):
    t = Tuner("most_popular", tune, resume)
    t.trial(lambda: bl.MostPopular(td), {})
    record(t, full, [(0, t.best_model(lambda c: bl.MostPopular(td)))])


def tune_item_knn(td, tune, full, resume=True, min_pos=20):
    t = Tuner("item_knn", tune, resume)
    gram = bl.ItemGram(td, min_pos)
    best_shrink, best_k = 0.0, 100
    for _ in range(2):  # coordinate descent: k, then shrink, twice
        best_k = t.search_1d(lambda k: bl.ItemKNN(td, gram, k, best_shrink),
                             [25, 50, 100, 200, 400], "k", {"shrink": best_shrink,
                                                            "min_pos": min_pos})
        best_shrink = t.search_1d(lambda s: bl.ItemKNN(td, gram, best_k, s),
                                  [0.0, 1.0, 10.0, 100.0], "shrink", {"k": best_k,
                                                                      "min_pos": min_pos})
    model = t.best_model(lambda c: bl.ItemKNN(td, gram, c["k"], c["shrink"]))
    del gram
    record(t, full, [(0, model)])


def tune_ease(td, tune, full, resume=True):
    t = Tuner("ease", tune, resume)
    for min_pos in (20, 10):
        gram = bl.ItemGram(td, min_pos)
        t.search_1d(lambda lam: bl.EASE(td, gram, lam),
                    [100.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0], "lam", {"min_pos": min_pos})
        del gram
        gc.collect()
    model = t.best_model(lambda c: bl.EASE(td, bl.ItemGram(td, c["min_pos"]), c["lam"]))
    record(t, full, [(0, model)])


def tune_implicit_als(td, tune, full, resume=True, spark_cores=6):
    from src.data_prep import get_spark
    spark = get_spark(master=f"local[{spark_cores}]", driver_memory="6g")
    spark.sparkContext.setLogLevel("ERROR")
    spark.sparkContext.setCheckpointDir("/tmp/cineinfer_als_checkpoints")
    pos = bl.positives_for_spark(spark, SPLITS, "user", td.training).cache()
    pos.count()
    t = Tuner("implicit_als", tune, resume)
    build = lambda **c: bl.ImplicitALS(td, pos, **c)
    # Coordinate descent at rank 64 (reg, then alpha, twice), then rank at the best pair, then a
    # final finer reg check at the best (rank, alpha): the best reg moved with alpha earlier.
    rank, alpha, reg = 64, 10.0, 0.1
    for _ in range(2):
        reg = t.search_1d(lambda v: build(rank=rank, reg=v, alpha=alpha),
                          [0.01, 0.1, 1.0], "reg", {"rank": rank, "alpha": alpha},
                          grow=10.0, max_extend=2)
        alpha = t.search_1d(lambda v: build(rank=rank, reg=reg, alpha=v),
                            [1.0, 5.0, 20.0, 50.0], "alpha", {"rank": rank, "reg": reg},
                            max_extend=2)
    rank = t.search_1d(lambda v: build(rank=v, reg=reg, alpha=alpha),
                       [32, 64, 128, 256], "rank", {"reg": reg, "alpha": alpha}, max_extend=0)
    if rank == 256:
        print("  NOTE: best rank is the largest tried (256); rank is capped by compute budget",
              flush=True)
    reg = t.search_1d(lambda v: build(rank=rank, reg=v, alpha=alpha),
                      [reg / 3, reg, reg * 3], "reg", {"rank": rank, "alpha": alpha},
                      grow=3.0, max_extend=2)
    cfg = {"rank": rank, "reg": reg, "alpha": alpha}
    print(f"  best ALS config {cfg}; refitting with seeds {ALS_SEEDS}", flush=True)
    models = [(s, build(**cfg, seed=s)) for s in ALS_SEEDS]
    record(t, full, models)
    spark.stop()


TUNERS = {"most_popular": tune_most_popular, "item_knn": tune_item_knn, "ease": tune_ease,
          "implicit_als": tune_implicit_als}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", default=",".join(TUNERS))
    p.add_argument("--fresh", action="store_true", help="ignore logged trials")
    p.add_argument("--spark-cores", type=int, default=6)
    args = p.parse_args()
    t = time.time()
    items = ev.ItemIndex.from_movies_csv(MOVIES)
    ratings = ev.load_ratings(SPLITS, "user")
    td = bl.build_train_data(ratings, items, ("train",))
    full = ev.build_eval_data(ratings, items, "user", "val")
    tune = full.subsample(TUNE_USERS, TUNE_SEED)
    del ratings
    gc.collect()
    print(f"data ready ({time.time() - t:.0f}s): {td.X.nnz} train positives, "
          f"{full.n_users} val users, tuning on {tune.n_users}", flush=True)
    for name in args.models.split(","):
        print(f"\n== {name} ==", flush=True)
        kw = {"spark_cores": args.spark_cores} if name == "implicit_als" else {}
        TUNERS[name](td, tune, full, resume=not args.fresh, **kw)
        gc.collect()


if __name__ == "__main__":
    main()

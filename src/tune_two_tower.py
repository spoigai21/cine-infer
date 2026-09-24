"""Phase 5: tune the two-tower model on VALIDATION, with the same protocol as the baselines.

- Every config trains on train positives and is early-stopped on the fixed 20k-user validation
  subsample (NDCG@10, patience 2). The best epoch count becomes part of its config.
- Coordinate descent over tau, lr, hist_len, pairs_per_user, dim, then single trials of uniform
  negatives and a larger batch. Grids extend when the best value sits on an edge.
- The winner is refit with seeds 42/43/44 for exactly its chosen epoch count (no early stopping:
  the same rule as the §2.1 train + val refit), then scored on the FULL validation population.
- strict_time is fixed to True by design, not tuned. The within-second movieId order is a split
  artifact that validation rewards (see the Phase 5 notes in the guide), so tuning would pick it
  for the wrong reason.
- The test slice is never read here.

Outputs: results/two_tower.csv (per-seed full-validation rows), results/tuning/two_tower_trials.csv
(every trial, resumable), models/two_tower_seed{42,43,44}.pt (weights, git-ignored). Usage: `make two-tower`.
"""
import argparse
import json
import time
from pathlib import Path

from src import evaluate as ev
from src import two_tower as tt
from src.tune_baselines import TRIAL_COLUMNS, TUNE_SEED, TUNE_USERS, Tuner, record

SPLITS = Path("data/splits.parquet")
MOVIES = Path("data/ml-25m/movies.csv")
OUT_CSV = Path("results/two_tower.csv")
TRIALS_CSV = Path("results/tuning/two_tower_trials.csv")
MODELS_DIR = Path("models")  # git-ignored
SEEDS = (42, 43, 44)
TUNE_SEED_MODEL = 0
BASE = {"strict_time": True}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fresh", action="store_true")
    args = p.parse_args()
    t0 = time.time()
    items = ev.ItemIndex.from_movies_csv(MOVIES)
    seq = tt.load_sequences(SPLITS, items, ("train",))
    ratings = ev.load_ratings(SPLITS, "user")
    full = ev.build_eval_data(ratings, items, "user", "val")
    tune = full.subsample(TUNE_USERS, TUNE_SEED)
    del ratings
    print(f"data ready ({time.time() - t0:.0f}s), device {tt.device()}", flush=True)

    early_stop_metric = lambda sc: ev.evaluate(sc, tune).summary["ndcg@10"]

    def fit(cfg):
        sc, info = tt.train_two_tower(seq, cfg, TUNE_SEED_MODEL, eval_fn=early_stop_metric,
                                      log=lambda m: print(m, flush=True))
        sc.config["device"] = info["device"]
        return sc

    defaults = {k: v for k, v in tt.DEFAULTS.items() if k != "max_epochs"}
    t = Tuner("two_tower", tune, resume=not args.fresh, trials_csv=TRIALS_CSV,
              columns=TRIAL_COLUMNS + ["epochs"], canon=lambda c: {**defaults, **c})
    cfg = {**BASE, "tau": 0.05, "lr": 1e-3, "hist_len": 50, "pairs_per_user": 100, "dim": 64}

    def search(key, values, **kw):
        cfg[key] = t.search_1d(lambda v: fit({**cfg, key: v}), values, key,
                               {k: v for k, v in cfg.items() if k != key}, **kw)

    search("tau", [0.02, 0.05, 0.1])
    search("lr", [3e-4, 1e-3, 3e-3], grow=3.0)
    search("hist_len", [20, 50, 100])
    search("pairs_per_user", [50, 100, 200])
    search("dim", [64, 128], max_extend=1)
    search("n_uniform", [0, 1024], max_extend=0)
    search("batch", [4096, 8192], max_extend=0)

    best_score, best_cfg, best_model = t.best
    if best_model is not None:
        epochs = best_model.config["epochs"]
    else:  # best trial came from the log: its early-stopped epoch count is logged too
        key = json.dumps(t.canon(best_cfg), sort_keys=True)
        row = next(r for r in t.trials
                   if json.dumps(t.canon(json.loads(r["config"])), sort_keys=True) == key)
        epochs = int(row["epochs"])
    print(f"  best config {best_cfg}, {epochs} epochs; refitting seeds {SEEDS}", flush=True)
    models = []
    for s in SEEDS:
        sc, info = tt.train_two_tower(seq, best_cfg, s, fixed_epochs=epochs,
                                      log=lambda m: print(m, flush=True))
        sc.config["device"] = info["device"]
        tt.save_scorer(sc, MODELS_DIR / f"two_tower_seed{s}.pt", s)
        models.append((s, sc))
    record(t, full, models, out_csv=OUT_CSV)
    print(f"done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

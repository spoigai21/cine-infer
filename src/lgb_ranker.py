"""Phase 6: the LightGBM ranker, run as its own process.

Never imports torch, directly or through other src modules. PyTorch and LightGBM each bring
their own OpenMP runtime on macOS, and in one process LightGBM's multithreaded training
segfaults. src/ranker.py (the PyTorch side) writes features to disk; this process trains,
predicts, and writes predictions back.

Inputs (in --data, written by ranker.write_training_rows / write_eval_features):
  train_X.npy, train_y.npy, train_groups.npy, meta.json     ranker-training rows (label slice)
  eval_X.npy (users x k x features), eval_cand.npy          candidates to score
Outputs: --pred-out (eval users x k predictions, float32); optional --save-model; and one JSON
line on stdout: rounds used, timing, and whether torch was (wrongly) loaded.

Early stopping: without --fixed-rounds, 10% of the ranker-training users (by seed) are held out
and training stops after 30 rounds without NDCG@10 improvement on them. This is never the real
validation slice. With --fixed-rounds, trains exactly that many rounds (seed refits, §2.1).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

DEFAULTS = {"num_leaves": 63, "learning_rate": 0.05, "min_data_in_leaf": 100,
            "feature_fraction": 0.8, "max_rounds": 2000}


def split_groups(groups, frac, seed):
    """Row masks for a user-level (whole query group) train/holdout split."""
    rng = np.random.default_rng(seed)
    hold = rng.random(len(groups)) < frac
    row_hold = np.repeat(hold, groups)
    return ~row_hold, row_hold, groups[~hold], groups[hold]


def main(argv=None):
    import lightgbm as lgb
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--config", default="{}")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--features", default=None, help="comma list; default all in meta.json")
    p.add_argument("--rows", default=None, type=Path, help="npy of eval rows to score")
    p.add_argument("--fixed-rounds", type=int, default=None)
    p.add_argument("--save-model", default=None, type=Path)
    p.add_argument("--pred-out", required=True, type=Path)
    p.add_argument("--threads", type=int, default=6)
    a = p.parse_args(argv)
    t0 = time.time()
    cfg = {**DEFAULTS, **json.loads(a.config)}
    meta = json.loads((a.data / "meta.json").read_text())
    names = meta["features"]
    feats = a.features.split(",") if a.features else names
    cols = [names.index(f) for f in feats]

    X = np.load(a.data / "train_X.npy", mmap_mode="r")[:, cols]
    y = np.load(a.data / "train_y.npy")
    g = np.load(a.data / "train_groups.npy")
    params = {"objective": "lambdarank", "metric": "ndcg", "eval_at": [10],
              "num_leaves": cfg["num_leaves"], "learning_rate": cfg["learning_rate"],
              "min_data_in_leaf": cfg["min_data_in_leaf"],
              "feature_fraction": cfg["feature_fraction"], "bagging_fraction": 0.8,
              "bagging_freq": 1, "seed": a.seed, "deterministic": True, "force_row_wise": True,
              "num_threads": a.threads, "verbose": -1}
    if a.fixed_rounds is None:
        tr, ho, gtr, gho = split_groups(g, 0.1, a.seed)
        dtr = lgb.Dataset(X[tr], y[tr], group=gtr, feature_name=feats)
        dho = lgb.Dataset(X[ho], y[ho], group=gho, reference=dtr)
        booster = lgb.train(params, dtr, cfg["max_rounds"], valid_sets=[dho],
                            callbacks=[lgb.early_stopping(30, verbose=False)])
        rounds = booster.best_iteration
    else:
        booster = lgb.train(params, lgb.Dataset(X, y, group=g, feature_name=feats),
                            a.fixed_rounds)
        rounds = a.fixed_rounds
    fit_s = time.time() - t0

    E = np.load(a.data / "eval_X.npy", mmap_mode="r")
    rows = np.load(a.rows) if a.rows else np.arange(len(E))
    pred = np.empty((len(rows), E.shape[1]), dtype=np.float32)
    for b in range(0, len(rows), 4096):
        chunk = np.asarray(E[rows[b:b + 4096]])[..., cols]
        n, k, f = chunk.shape
        pred[b:b + n] = booster.predict(chunk.reshape(n * k, f), num_iteration=rounds).reshape(n, k)
    a.pred_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(a.pred_out, pred)
    if a.save_model:
        a.save_model.parent.mkdir(parents=True, exist_ok=True)
        booster.save_model(str(a.save_model), num_iteration=rounds)
    print(json.dumps({"rounds": int(rounds), "fit_seconds": round(fit_s, 1),
                      "total_seconds": round(time.time() - t0, 1), "features": feats,
                      "torch_loaded": "torch" in sys.modules}))


if __name__ == "__main__":
    main()

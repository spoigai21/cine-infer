"""Loose end: how much did the cross-user time leak help the ranker? (VALIDATION only.)

The headline ranker's item features (popularity, mean rating) are computed over the whole
training set. Under the per-user split that includes other users' ratings from after this user's
cutoff. This rebuilds the Phase 6 validation pipeline with point-in-time versions
(ranker.ItemTimeline: only ratings made strictly before the user's last training rating, by
anyone) and compares, with the headline's frozen LightGBM config and fixed round count:

  headline   the headline's 15 features, recomputed here (must reproduce results/two_stage.csv
             exactly: same candidates, same features, same training users, deterministic LightGBM)
  pit        headline with item_log_n / item_log_pos / item_mean replaced by point-in-time versions
  pit_time   pit + point-in-time recency (days since the movie's last rating, days since its
             first), which the headline had to exclude because the full-train versions leak

Validation only: the test slice was scored once in Phase 6b, and these variants were designed
after seeing test results, so evaluating them on test would be a second look.
Outputs: results/pit_experiment.csv (per seed + paired bootstrap vs headline). Uses the Phase 6
train_core / full-train retrievers (models/two_tower_core_seed*.pt, two_tower_seed*.pt).
Usage: `make pit-experiment` (~45 min).
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src import baselines as bl
from src import evaluate as ev
from src import ranker as rk
from src import two_tower as tt
from src.tune_ranker import HEADLINE

SPLITS = Path("data/splits.parquet")
MOVIES = Path("data/ml-25m/movies.csv")
FEATURES_DIR = Path("data/features")
WORK = Path("data/ranker_pit")
MODELS = Path("models")
RUNS = Path("data/eval_runs")
OUT = Path("results/pit_experiment.csv")
SEEDS = (42, 43, 44)
RANKER_TRAIN_USERS = 30_000
SWAP = {"item_log_n": "pit_item_log_n", "item_log_pos": "pit_item_log_pos", "item_mean": "pit_item_mean"}
VARIANTS = {"headline": list(HEADLINE),
            "pit": [SWAP.get(f, f) for f in HEADLINE],
            "pit_time": [SWAP.get(f, f) for f in HEADLINE] + ["pit_days_since_last", "pit_age_days"]}


def log(m):
    print(m, flush=True)


def ease_on(fine, items, training, cfg):
    td = bl.build_train_data(fine, items, training)
    return bl.EASE(td, bl.ItemGram(td, cfg["min_pos"]), cfg["lam"])


def stage_a(seed, items, content, fine, timeline, val, ease_cfg):
    d = WORK / f"seed{seed}"
    if (d / "meta.json").exists() and (d / "eval_users.npy").exists():
        log(f"  seed {seed}: features exist, skipping")
        return
    t = time.time()
    seq_core = tt.load_sequences(SPLITS, items, ("train_core",))
    ctx = rk.TrainingSetContext.build(
        "train_core", tt.load_scorer(MODELS / f"two_tower_core_seed{seed}.pt", seq_core),
        ease_on(fine, items, ("train_core",), ease_cfg), seq_core, fine, ("train_core",),
        FEATURES_DIR, items, content)
    ctx.timeline = timeline
    tail = fine[(fine.split == "train_tail") & (fine.rating >= 4)]
    L = sp.csr_matrix((np.ones(len(tail), dtype=np.int8),
                       (seq_core.rows(tail.userId.to_numpy()), items.to_index(tail.movieId.to_numpy()))),
                      shape=(len(seq_core.user_ids), len(items)))
    eligible = seq_core.user_ids[(seq_core.lengths > 0) & (np.diff(L.indptr) > 0)]
    users = np.sort(np.random.default_rng(seed).choice(eligible, RANKER_TRAIN_USERS, replace=False))
    rk.write_training_rows(ctx, users, L, seq_core.rows(users), d)  # same users as Phase 6
    del ctx
    seq = tt.load_sequences(SPLITS, items, ("train",))
    ctx = rk.TrainingSetContext.build(
        "train", tt.load_scorer(MODELS / f"two_tower_seed{seed}.pt", seq),
        ease_on(fine, items, ("train_core", "train_tail"), ease_cfg), seq, fine,
        ("train_core", "train_tail"), FEATURES_DIR, items, content)
    ctx.timeline = timeline
    rk.write_eval_features(ctx, val.users, d)
    log(f"  seed {seed}: features built ({time.time() - t:.0f}s)")


def main():
    t0 = time.time()
    items = ev.ItemIndex.from_movies_csv(MOVIES)
    content = rk.ItemContent.load(MOVIES, FEATURES_DIR / "genome.parquet", items)
    fine = ev.load_ratings(SPLITS, "user", fine=True)
    allr = pd.read_parquet(SPLITS, columns=["userId", "movieId", "rating", "timestamp"])
    timeline = rk.ItemTimeline.from_ratings(allr, items)
    del allr
    val = ev.build_eval_data(ev.load_ratings(SPLITS, "user"), items, "user", "val")
    ease_cfg = json.loads(pd.read_csv("results/baselines.csv").query("model == 'ease'").iloc[0].config)
    ts = pd.read_csv("results/two_stage.csv")
    head = ts[ts.model == "two_stage"].set_index("seed")
    rcfg = json.loads(head.config.iloc[0])
    rounds = int(rcfg["rounds"])
    lgb_cfg = {k: v for k, v in rcfg.items() if k not in ("features", "rounds")}
    log(f"data ready ({time.time() - t0:.0f}s); frozen ranker config {lgb_cfg}, {rounds} rounds")
    seq_train = tt.load_sequences(SPLITS, items, ("train",))
    rows = []
    for s in SEEDS:
        stage_a(s, items, content, fine, timeline, val, ease_cfg)
        d = WORK / f"seed{s}"
        users = np.load(d / "eval_users.npy")
        cand = np.load(d / "eval_cand.npy", mmap_mode="r")
        fallback = tt.load_scorer(MODELS / f"two_tower_seed{s}.pt", seq_train)
        for name, feats in VARIANTS.items():
            rep = rk.run_lgb(d, lgb_cfg, s, features=feats, fixed_rounds=rounds,
                             pred_out=d / f"pred_{name}.npy")
            sc = rk.PrecomputedScorer(users, cand, np.load(rep["pred_path"]), len(items),
                                      {**lgb_cfg, "rounds": rounds, "features": name}, fallback=fallback)
            res = ev.evaluate(sc, val)
            ev.save_per_user(res, RUNS / f"pit_{name}_user_val_seed{s}.parquet")
            r = {"variant": name, "seed": s, **{k: res.summary[k] for k in ("ndcg@10", "recall@10", "auc", "coverage")}}
            rows.append(r)
            log(f"  seed {s} {name:<9} ndcg={r['ndcg@10']:.5f} recall={r['recall@10']:.5f}")
        got = next(r["ndcg@10"] for r in rows if r["seed"] == s and r["variant"] == "headline")
        ref = head.loc[s, "ndcg@10"]
        if abs(got - ref) > 1e-6:
            raise AssertionError(f"seed {s}: recomputed headline {got:.6f} != Phase 6 {ref:.6f}; "
                                 f"the comparison would not be like for like")
        log(f"  seed {s}: headline reproduces Phase 6 exactly ({got:.6f})")
    df = pd.DataFrame(rows)
    pu = {v: pd.concat(pd.read_parquet(RUNS / f"pit_{v}_user_val_seed{s}.parquet") for s in SEEDS)
          .groupby("userId")[["ndcg", "recall"]].mean().reset_index() for v in VARIANTS}
    for v in ("pit", "pit_time"):
        b = ev.paired_bootstrap(pu[v], pu["headline"], "ndcg", n_boot=2000, seed=0)
        df.loc[df.variant == v, ["ndcg_diff_vs_headline", "ci_low", "ci_high"]] = \
            [b["mean_diff"], b["ci_low"], b["ci_high"]]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, float_format="%.6f")
    log(df.groupby("variant")[["ndcg@10", "recall@10", "ndcg_diff_vs_headline", "ci_low", "ci_high"]]
        .median().round(5).to_string())
    log(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

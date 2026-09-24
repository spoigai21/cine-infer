"""Phase 6: build, tune and evaluate the two-stage system (two-tower -> LightGBM ranker) on
VALIDATION. The test slice is never read here (Phase 6b does that, once).

Per seed s in 42/43/44 (steps are skipped when their outputs already exist, so it resumes):
  A1  two-tower on train_core only (tuned config, fixed epochs) -> models/two_tower_core_seed{s}.pt
  A2  ranker-training rows: 30k users with train_core and train_tail positives; candidates from the
      train_core retriever, features from train_core, labels = train_tail positives
      -> data/ranker/seed{s}/train_*.npy
  A3  validation candidates + features: the saved full-train two-tower (seed s), EASE and Phase 1
      features on train, for the full validation population -> data/ranker/seed{s}/eval_*.npy
  A4  recall ceiling of the validation candidates at k = 50/100/200/500
Tuning (seed 42): LightGBM configs, each scored on the fixed 20k validation-user subsample
  (NDCG@10), rounds early-stopped on held-out ranker-training users (never validation).
Final (each seed): best config, fixed rounds, scored on the full validation population, plus the
  two ablations: without the EASE features, and with the time features added back. The headline
  ranker excludes the time features (see HEADLINE). Paired bootstrap vs retrieval only.

Outputs: results/two_stage.csv, results/tuning/ranker_trials.csv, results/ablation.csv,
  models/ranker_seed{s}.txt (+ _no_ease). LightGBM always runs in its own process
  (src/lgb_ranker.py); see ranker.py for why. Usage: `make ranker`.
"""
import argparse
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
from src.tune_baselines import (BASELINE_COLUMNS, SELECT, TRIAL_COLUMNS, TUNE_SEED, TUNE_USERS,
                                Tuner, record)

SPLITS = Path("data/splits.parquet")
MOVIES = Path("data/ml-25m/movies.csv")
FEATURES_DIR = Path("data/features")
GENOME = FEATURES_DIR / "genome.parquet"
WORK = Path("data/ranker")
MODELS = Path("models")
OUT_CSV = Path("results/two_stage.csv")
TRIALS_CSV = Path("results/tuning/ranker_trials.csv")
ABLATION_CSV = Path("results/ablation.csv")
SEEDS = (42, 43, 44)
RANKER_TRAIN_USERS = 30_000
# Time features compare the user's last training rating with the movie's first/last rating by
# ANYONE in the training set. Under the per-user split, that set includes other users' ratings
# from after this user's cutoff (§3), so these features can say "still being rated after your
# cutoff" or "released around your cutoff": information a live system wouldn't have.
TIME_FEATURES = ["item_days_since_last", "item_age_days"]
# The headline two-stage system leaves the time features out (decided after measuring them:
# they carried ~18% of the ranker's gain on validation, the part a live system couldn't have).
HEADLINE = [f for f in rk.FEATURES if f not in TIME_FEATURES]
ABLATIONS = {"no_ease": ([f for f in HEADLINE if f not in rk.EASE_FEATURES],
                         "headline minus EASE features"),
             "with_time": (list(rk.FEATURES),
                           "headline plus time features (cross-user future leak, not used)")}


def log(msg):
    print(msg, flush=True)


def tuned(csv, model):
    b = pd.read_csv(csv)
    return json.loads(b[b.model == model].iloc[0].config)


def ease_for(fine, items, training, cfg):
    td = bl.build_train_data(fine, items, training)
    return bl.EASE(td, bl.ItemGram(td, cfg["min_pos"]), cfg["lam"])


def stage_a(seed, items, content, fine, val, tt_cfg, ease_cfg):
    d = WORK / f"seed{seed}"
    done = (d / "meta.json").exists() and (d / "eval_users.npy").exists() \
        and (d / "recall_ceiling.json").exists()
    if done:
        log(f"  seed {seed}: stage A outputs exist, skipping")
        return
    t0 = time.time()
    # A1 train_core retriever
    seq_core = tt.load_sequences(SPLITS, items, ("train_core",))
    core_path = MODELS / f"two_tower_core_seed{seed}.pt"
    if core_path.exists():
        retr_core = tt.load_scorer(core_path, seq_core)
    else:
        log(f"  seed {seed}: training two-tower on train_core ({tt_cfg['epochs']} epochs)")
        retr_core, info = tt.train_two_tower(seq_core, tt_cfg, seed, fixed_epochs=tt_cfg["epochs"],
                                             log=log)
        retr_core.config["device"] = info["device"]
        tt.save_scorer(retr_core, core_path, seed)
    # A2 ranker-training rows (labels from train_tail)
    if not (d / "meta.json").exists():
        ctx_core = rk.TrainingSetContext.build(
            "train_core", retr_core, ease_for(fine, items, ("train_core",), ease_cfg), seq_core,
            fine, ("train_core",), FEATURES_DIR, items, content)
        tail = fine[(fine.split == "train_tail") & (fine.rating >= 4)]
        L = sp.csr_matrix((np.ones(len(tail), dtype=np.int8),
                           (seq_core.rows(tail.userId.to_numpy()),
                            items.to_index(tail.movieId.to_numpy()))),
                          shape=(len(seq_core.user_ids), len(items)))
        eligible = seq_core.user_ids[(seq_core.lengths > 0) & (np.diff(L.indptr) > 0)]
        users = np.sort(np.random.default_rng(seed).choice(eligible, RANKER_TRAIN_USERS,
                                                           replace=False))
        meta = rk.write_training_rows(ctx_core, users, L, seq_core.rows(users), d)
        meta["eligible_users"] = int(len(eligible))
        (d / "meta.json").write_text(json.dumps(meta, indent=1))
        log(f"  seed {seed}: ranker rows {meta['train_rows']} from {meta['train_users_kept']} users "
            f"({meta['train_users_dropped_no_positive']} dropped: no tail positive among candidates)")
        del ctx_core
    # A3 validation candidates + features (full-train retriever, train features)
    seq = tt.load_sequences(SPLITS, items, ("train",))
    ctx = rk.TrainingSetContext.build(
        "train", tt.load_scorer(MODELS / f"two_tower_seed{seed}.pt", seq),
        ease_for(fine, items, ("train_core", "train_tail"), ease_cfg), seq, fine,
        ("train_core", "train_tail"), FEATURES_DIR, items, content)
    if not (d / "eval_users.npy").exists():
        rk.write_eval_features(ctx, val.users, d)
    # A4 recall ceiling
    rc = rk.recall_ceiling(ctx, val)
    (d / "recall_ceiling.json").write_text(json.dumps(rc))
    log(f"  seed {seed}: stage A done ({time.time() - t0:.0f}s); recall ceiling {rc}")


N_ITEMS = None  # set in main(): catalog size
SEQ_TRAIN = None  # set in main(): train sequences, for the full-train retrievers
_RETRIEVERS = {}


def retriever(seed):
    """The full-train two-tower that produced this seed's validation candidates."""
    if seed not in _RETRIEVERS:
        _RETRIEVERS[seed] = tt.load_scorer(MODELS / f"two_tower_seed{seed}.pt", SEQ_TRAIN)
    return _RETRIEVERS[seed]


def scorer_from(seed, report, rows, config):
    d = WORK / f"seed{seed}"
    users = np.load(d / "eval_users.npy")
    cand = np.load(d / "eval_cand.npy", mmap_mode="r")
    rows = np.arange(len(users)) if rows is None else rows
    pred = np.load(report["pred_path"])
    # Non-candidates keep the retriever's score (below every candidate), so AUC measures ranking
    # rather than the top-200 cut-off; top-10 metrics are unchanged (see PrecomputedScorer).
    return rk.PrecomputedScorer(users[rows], cand[rows], pred, N_ITEMS,
                                {**config, "rounds": report["rounds"]}, fallback=retriever(seed))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--only-ablations", default=None,
                   help="comma list of ablation variants to (re)run; reuses the main fits")
    args = p.parse_args()
    t0 = time.time()
    global N_ITEMS, SEQ_TRAIN
    items = ev.ItemIndex.from_movies_csv(MOVIES)
    N_ITEMS = len(items)
    content = rk.ItemContent.load(MOVIES, GENOME, items)
    fine = ev.load_ratings(SPLITS, "user", fine=True)
    val = ev.build_eval_data(ev.load_ratings(SPLITS, "user"), items, "user", "val")
    tune = val.subsample(TUNE_USERS, TUNE_SEED)
    tt_cfg = tuned("results/two_tower.csv", "two_tower")
    ease_cfg = tuned("results/baselines.csv", "ease")
    log(f"data ready ({time.time() - t0:.0f}s); two-tower {tt_cfg}; EASE {ease_cfg}")

    for s in SEEDS:
        stage_a(s, items, content, fine, val, tt_cfg, ease_cfg)
    SEQ_TRAIN = tt.load_sequences(SPLITS, items, ("train",))

    # Tuning on seed 42's features, scored on the 20k tuning users
    s0 = SEEDS[0]
    users0 = np.load(WORK / f"seed{s0}" / "eval_users.npy")
    tune_rows = np.searchsorted(users0, tune.users)
    assert np.array_equal(users0[tune_rows], tune.users)
    defaults = {k: v for k, v in rk_defaults().items() if k != "max_rounds"}
    t = Tuner("two_stage", tune, resume=not args.fresh, trials_csv=TRIALS_CSV,
              columns=TRIAL_COLUMNS + ["rounds"], canon=lambda c: {**defaults, **c})

    def fit(cfg):
        lgb_cfg = {k: v for k, v in cfg.items() if k != "features"}
        rep = rk.run_lgb(WORK / f"seed{s0}", lgb_cfg, s0, features=HEADLINE, rows=tune_rows,
                         pred_out=WORK / f"seed{s0}" / "pred_tune.npy")
        return scorer_from(s0, rep, tune_rows, cfg)

    # "features" is part of the config so logged trials of other feature sets never match
    cfg = {"features": "headline", "learning_rate": 0.05, "num_leaves": 63,
           "min_data_in_leaf": 100}

    def search(key, values, **kw):
        cfg[key] = t.search_1d(lambda v: fit({**cfg, key: v}), values, key,
                               {k: v for k, v in cfg.items() if k != key}, **kw)

    search("learning_rate", [0.02, 0.05, 0.1], grow=2.5, max_extend=2)
    search("num_leaves", [31, 63, 127], max_extend=2)
    search("min_data_in_leaf", [25, 100, 400], grow=4.0, max_extend=2)

    best_cfg = {k: v for k, v in t.best[1].items() if k != "features"}
    key = json.dumps(t.canon(t.best[1]), sort_keys=True)
    best_row = next(r for r in t.trials
                    if json.dumps(t.canon(json.loads(r["config"])), sort_keys=True) == key)
    rounds = int(best_row["rounds"])
    log(f"  best ranker config {best_cfg}, {rounds} rounds; refitting seeds {SEEDS}")

    # Final: every seed, full validation; then each ablation variant with the SAME config and the
    # SAME fixed round count. (Letting variants early-stop on their own stopped them at 20-112
    # rounds vs 235, so part of each gap would have been under-training, not the missing features.)
    if not args.only_ablations:
        full_models = []
        for s in SEEDS:
            d = WORK / f"seed{s}"
            rep = rk.run_lgb(d, best_cfg, s, features=HEADLINE, fixed_rounds=rounds,
                             save_model=MODELS / f"ranker_seed{s}.txt", pred_out=d / "pred_full.npy")
            full_models.append((s, scorer_from(s, rep, None, {**best_cfg, "features": "headline"})))
        record(t, val, full_models, out_csv=OUT_CSV)
    variants = args.only_ablations.split(",") if args.only_ablations else list(ABLATIONS)
    for v in variants:
        feats, why = ABLATIONS[v]
        name, rows = f"two_stage_{v}", []
        for s in SEEDS:
            d = WORK / f"seed{s}"
            rep = rk.run_lgb(d, best_cfg, s, features=feats, fixed_rounds=rounds,
                             save_model=MODELS / f"ranker_{v}_seed{s}.txt",
                             pred_out=d / f"pred_full_{v}.npy")
            sc = scorer_from(s, rep, None, {**best_cfg, "features": v})
            res = ev.evaluate(sc, val)
            ev.save_per_user(res, Path("data/eval_runs") / f"{name}_user_val_seed{s}.parquet")
            row = ev.result_row(name, s, sc.config, res, val)
            row["selection_metric"] = f"n/a (ablation: best two_stage config, {why})"
            rows.append(row)
            log(f"  seed {s}: {name} ndcg={res.summary['ndcg@10']:.5f} ({rep['rounds']} rounds)")
        ev.replace_model_rows(rows, OUT_CSV, name, BASELINE_COLUMNS)
    write_ablation()
    log(f"done ({time.time() - t0:.0f}s)")


def rk_defaults():
    from src.lgb_ranker import DEFAULTS  # plain dict; lgb_ranker imports lightgbm only inside main()
    return DEFAULTS


def write_ablation():
    """Retrieval only vs two-stage (with / without EASE features), per seed, full validation,
    with the recall ceiling and a paired bootstrap against retrieval only."""
    runs = Path("data/eval_runs")
    tw = pd.read_csv("results/two_tower.csv").set_index("seed")
    ts = pd.read_csv(OUT_CSV)
    rows = []
    for s in SEEDS:
        rc = json.loads((WORK / f"seed{s}" / "recall_ceiling.json").read_text())
        base = pd.read_parquet(runs / f"two_tower_user_val_seed{s}.parquet")
        rows.append({"system": "retrieval_only", "seed": s, "ndcg@10": tw.loc[s, "ndcg@10"],
                     "recall@10": tw.loc[s, "recall@10"], "coverage": tw.loc[s, "coverage"],
                     **{f"recall_ceiling@{k}": v for k, v in rc.items()}})
        for model in ["two_stage"] + [f"two_stage_{v}" for v in ABLATIONS
                                      if (ts.model == f"two_stage_{v}").any()]:
            r = ts[(ts.model == model) & (ts.seed == s)].iloc[0]
            pu = pd.read_parquet(runs / f"{model}_user_val_seed{s}.parquet")
            bs = ev.paired_bootstrap(pu, base, "ndcg")
            rows.append({"system": model, "seed": s, "ndcg@10": r["ndcg@10"],
                         "recall@10": r["recall@10"], "coverage": r["coverage"],
                         **{f"recall_ceiling@{k}": v for k, v in rc.items()},
                         "ndcg_diff_vs_retrieval": bs["mean_diff"],
                         "ndcg_diff_ci_low": bs["ci_low"], "ndcg_diff_ci_high": bs["ci_high"]})
    pd.DataFrame(rows).to_csv(ABLATION_CSV, index=False, float_format="%.6f")
    log(pd.DataFrame(rows).groupby("system")[["ndcg@10", "recall@10"]].median().round(4).to_string())


if __name__ == "__main__":
    main()

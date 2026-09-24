"""Phase 6b: the one-time test run.

Every model is refit on train + val with the config frozen on validation (read from the
results files, never typed in), then scored ONCE on the test slice of the per-user split:
masking train + val, population = users with >= 1 train positive and >= 1 test positive (§2.1).

Sealing: each (model, seed) is scored once and saved to data/test_runs/<model>_seed<s>.json
(+ per-user metrics in data/eval_runs/). A re-run skips every (model, seed) already scored and
never overwrites one, so an interruption can't turn into a second look at test.

Two-stage on test (§6.1, shifted by one slice): the ranker trains on candidates from the
full-train two-tower (saved in Phase 5) with train features and VALIDATION positives as labels.
It then ranks candidates from the train + val two-tower with train + val features. Same frozen
config and fixed round count as on validation; ablations too.

Outputs: results/test.csv (one row per model x seed), results/predictions_status.csv
(predictions #1-#4 settled). Usage: `make final-test` (resumable; slowest step, ALS, runs last).
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

SPLITS = Path("data/splits.parquet")
MOVIES = Path("data/ml-25m/movies.csv")
FEATURES_DIR = Path("data/features")
GENOME = FEATURES_DIR / "genome.parquet"
RUNS = Path("data/test_runs")
PER_USER = Path("data/eval_runs")
WORK = Path("data/ranker_test")
MODELS = Path("models")
TEST_CSV = Path("results/test.csv")
STATUS_CSV = Path("results/predictions_status.csv")
SEEDS = (42, 43, 44)
RANKER_TRAIN_USERS = 30_000
TV_FINE = ("train_core", "train_tail", "val")   # train + val, in fine slice names
TRAIN_FINE = ("train_core", "train_tail")


def log(msg):
    print(msg, flush=True)


def frozen(csv, model):
    b = pd.read_csv(csv)
    return json.loads(b[b.model == model].iloc[0].config)


def done(model, seed):
    return (RUNS / f"{model}_seed{seed}.json").exists()


def score_once(model_name, seed, scorer, test, fit_seconds, extra=None):
    """Evaluate on test and seal the result. Refuses to overwrite an existing one."""
    path = RUNS / f"{model_name}_seed{seed}.json"
    if path.exists():
        raise RuntimeError(f"{path} exists: test results are never re-scored")
    t = time.time()
    res = ev.evaluate(scorer, test)
    ev.save_per_user(res, PER_USER / f"{model_name}_user_test_seed{seed}.parquet")
    row = ev.result_row(model_name, seed, scorer.config, res, test)
    row.update({"fit_seconds": round(fit_seconds, 1), "eval_seconds": round(time.time() - t, 1),
                **(extra or {})})
    RUNS.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=1, default=str))
    log(f"  [test] {model_name} seed {seed}: ndcg={row['ndcg@10']:.5f} "
        f"recall={row['recall@10']:.5f} auc={row['auc']:.4f} cov={row['coverage']:.4f}")
    return row


class Named:
    """Attach a config to a scorer that doesn't carry one."""

    def __init__(self, scorer, config):
        self.scorer, self.config = scorer, config

    def score(self, users):
        return self.scorer.score(users)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", default=None, help="comma list of model names")
    p.add_argument("--spark-cores", type=int, default=6)
    args = p.parse_args()
    want = set(args.only.split(",")) if args.only else None
    run = lambda m: want is None or m in want
    t0 = time.time()

    items = ev.ItemIndex.from_movies_csv(MOVIES)
    coarse = ev.load_ratings(SPLITS, "user")
    fine = ev.load_ratings(SPLITS, "user", fine=True)
    test = ev.build_eval_data(coarse, items, "user", "test")
    td = bl.build_train_data(coarse, items, ("train", "val"))
    log(f"data ready ({time.time() - t0:.0f}s): {test.n_users} test users, "
        f"{td.X.nnz} train+val positives, coverage denominator {test.counts['coverage_denominator']}")

    base = "results/baselines.csv"
    # --- baselines (deterministic: one "seed" 0) ------------------------------------------
    if run("most_popular") and not done("most_popular", 0):
        t = time.time()
        score_once("most_popular", 0, bl.MostPopular(td), test, time.time() - t)
    gram_cache = {}

    def gram(min_pos):
        if min_pos not in gram_cache:
            gram_cache.clear()
            gram_cache[min_pos] = bl.ItemGram(td, min_pos)
        return gram_cache[min_pos]

    if run("item_knn") and not done("item_knn", 0):
        c = frozen(base, "item_knn")
        t = time.time()
        m = bl.ItemKNN(td, gram(c["min_pos"]), c["k"], c["shrink"])
        score_once("item_knn", 0, m, test, time.time() - t)
    if run("ease") and not done("ease", 0):
        c = frozen(base, "ease")
        t = time.time()
        m = bl.EASE(td, gram(c["min_pos"]), c["lam"])
        score_once("ease", 0, m, test, time.time() - t)
    seq_tv = None
    if run("ease_recent") and not done("ease_recent", 0):
        c = frozen(base, "ease_recent")
        seq_tv = seq_tv or tt.load_sequences(SPLITS, items, ("train", "val"))
        t = time.time()
        m = bl.RecentEASE(bl.EASE(td, gram(c["min_pos"]), c["lam"]), seq_tv, c["recent_n"])
        score_once("ease_recent", 0, m, test, time.time() - t)
    gram_cache.clear()

    # --- two-tower on train + val ------------------------------------------------------------
    tt_cfg = frozen("results/two_tower.csv", "two_tower")
    if run("two_tower") or run("two_stage"):
        seq_tv = seq_tv or tt.load_sequences(SPLITS, items, ("train", "val"))
    for s in SEEDS:
        path = MODELS / f"two_tower_tv_seed{s}.pt"
        if (run("two_tower") and not done("two_tower", s)) or (run("two_stage") and not path.exists()):
            t = time.time()
            if path.exists():
                sc = tt.load_scorer(path, seq_tv)
            else:
                log(f"  two-tower seed {s}: training on train + val ({tt_cfg['epochs']} epochs)")
                sc, info = tt.train_two_tower(seq_tv, tt_cfg, s, fixed_epochs=tt_cfg["epochs"], log=log)
                sc.config["device"] = info["device"]
                tt.save_scorer(sc, path, s)
            if run("two_tower") and not done("two_tower", s):
                score_once("two_tower", s, sc, test, time.time() - t)

    # --- two-stage (ranker trained on val labels, applied to test) ---------------------------
    if run("two_stage"):
        two_stage_test(items, fine, seq_tv, test, tt_cfg)

    # --- implicit ALS (slowest: last) --------------------------------------------------------
    if run("implicit_als") and not all(done("implicit_als", s) for s in SEEDS):
        from src.data_prep import get_spark
        c = frozen(base, "implicit_als")
        spark = get_spark(master=f"local[{args.spark_cores}]", driver_memory="6g")
        spark.sparkContext.setLogLevel("ERROR")
        spark.sparkContext.setCheckpointDir("/tmp/cineinfer_als_checkpoints")
        pos = bl.positives_for_spark(spark, SPLITS, "user", ("train", "val")).cache()
        pos.count()
        for s in SEEDS:
            if done("implicit_als", s):
                continue
            log(f"  ALS seed {s}: fitting on train + val {c}")
            t = time.time()
            m = bl.ImplicitALS(td, pos, rank=c["rank"], reg=c["reg"], alpha=c["alpha"],
                               max_iter=c["max_iter"], seed=s)
            score_once("implicit_als", s, m, test, time.time() - t)
        spark.stop()

    write_test_csv()
    settle()
    log(f"done ({time.time() - t0:.0f}s)")


def two_stage_test(items, fine, seq_tv, test, tt_cfg):
    from src.tune_ranker import ABLATIONS, HEADLINE
    variants = {"two_stage": HEADLINE, **{f"two_stage_{v}": f for v, (f, _) in ABLATIONS.items()}}
    if all(done(m, s) for m in variants for s in SEEDS):
        return
    content = rk.ItemContent.load(MOVIES, GENOME, items)
    ease_cfg = frozen("results/baselines.csv", "ease")
    rcfg = frozen("results/two_stage.csv", "two_stage")
    rounds = int(rcfg["rounds"])
    lgb_cfg = {k: v for k, v in rcfg.items() if k not in ("features", "rounds")}

    def ease_on(training):
        tdx = bl.build_train_data(fine, items, training)
        return bl.EASE(tdx, bl.ItemGram(tdx, ease_cfg["min_pos"]), ease_cfg["lam"])

    seq_train = tt.load_sequences(SPLITS, items, ("train",))
    val_pos = fine[(fine.split == "val") & (fine.rating >= 4)]
    L = sp.csr_matrix((np.ones(len(val_pos), dtype=np.int8),
                       (seq_train.rows(val_pos.userId.to_numpy()),
                        items.to_index(val_pos.movieId.to_numpy()))),
                      shape=(len(seq_train.user_ids), len(items)))
    for s in SEEDS:
        if all(done(m, s) for m in variants):
            continue
        d = WORK / f"seed{s}"
        t = time.time()
        if not (d / "meta.json").exists():  # ranker rows: full-train retriever, val labels
            ctx = rk.TrainingSetContext.build(
                "train", tt.load_scorer(MODELS / f"two_tower_seed{s}.pt", seq_train),
                ease_on(TRAIN_FINE), seq_train, fine, TRAIN_FINE, FEATURES_DIR, items, content)
            eligible = seq_train.user_ids[(seq_train.lengths > 0) & (np.diff(L.indptr) > 0)]
            users = np.sort(np.random.default_rng(s).choice(eligible, RANKER_TRAIN_USERS,
                                                            replace=False))
            meta = rk.write_training_rows(ctx, users, L, seq_train.rows(users), d)
            meta["eligible_users"] = int(len(eligible))
            (d / "meta.json").write_text(json.dumps(meta, indent=1))
            log(f"  two-stage seed {s}: ranker rows {meta['train_rows']} from "
                f"{meta['train_users_kept']} users (val labels)")
            del ctx
        if not (d / "eval_users.npy").exists():  # test candidates: train+val retriever
            ctx = rk.TrainingSetContext.build(
                "train_val", tt.load_scorer(MODELS / f"two_tower_tv_seed{s}.pt", seq_tv),
                ease_on(TV_FINE), seq_tv, fine, TV_FINE, FEATURES_DIR, items, content)
            rk.write_eval_features(ctx, test.users, d)
            rc = rk.recall_ceiling(ctx, test)
            (d / "recall_ceiling.json").write_text(json.dumps(rc))
            log(f"  two-stage seed {s}: test recall ceiling {rc}")
            del ctx
        prep_s = time.time() - t
        users = np.load(d / "eval_users.npy")
        cand = np.load(d / "eval_cand.npy", mmap_mode="r")
        # the retriever that produced the candidates scores the rest (AUC; see PrecomputedScorer)
        fallback = tt.load_scorer(MODELS / f"two_tower_tv_seed{s}.pt", seq_tv)
        rc = json.loads((d / "recall_ceiling.json").read_text())
        for name, feats in variants.items():
            if done(name, s):
                continue
            t = time.time()
            rep = rk.run_lgb(d, lgb_cfg, s, features=feats, fixed_rounds=rounds,
                             save_model=MODELS / f"test_{name}_seed{s}.txt",
                             pred_out=d / f"pred_{name}.npy")
            sc = rk.PrecomputedScorer(users, cand, np.load(rep["pred_path"]), len(items),
                                      {**lgb_cfg, "rounds": rounds,
                                       "features": name.replace("two_stage", "headline", 1)},
                                      fallback=fallback)
            score_once(name, s, sc, test, prep_s + time.time() - t,
                       {f"recall_ceiling@{k}": v for k, v in rc.items()})


def write_test_csv():
    rows = [json.loads(p.read_text()) for p in sorted(RUNS.glob("*.json"))]
    if not rows:
        return
    cols = ev.RESULT_COLUMNS + ["fit_seconds", "eval_seconds"] + \
        sorted({k for r in rows for k in r if k.startswith("recall_ceiling")})
    df = pd.DataFrame(rows)[cols].sort_values(["model", "seed"])
    TEST_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(TEST_CSV, index=False, float_format="%.6f")


def per_user_mean(model, metric):
    """Per-user metric averaged over the seeds of a model (for paired comparisons)."""
    files = sorted(PER_USER.glob(f"{model}_user_test_seed*.parquet"))
    return (pd.concat(pd.read_parquet(f) for f in files)
            .groupby("userId")[metric].mean().reset_index())


def settle():
    """Predictions #1-#4 from the test results, exactly as committed (results/predictions.csv)."""
    if not TEST_CSV.exists():
        return
    t = pd.read_csv(TEST_CSV)
    need = {"most_popular", "ease", "two_tower", "two_stage"}
    if not need <= set(t.model):
        log(f"  predictions not settled yet: missing {sorted(need - set(t.model))}")
        return
    med = t.groupby("model")[["ndcg@10", "recall@10", "coverage"]].median()
    rows = []
    # 1. two-tower NDCG >= 1.5 x most-popular
    r1 = med.loc["two_tower", "ndcg@10"] / med.loc["most_popular", "ndcg@10"]
    rows.append((1, r1 >= 1.5, f"two_tower/most_popular NDCG@10 = {r1:.3f} (needs >= 1.5)"))
    # 2. EASE matches or beats two-tower on Recall@10, two-tower within 10% of EASE
    b = ev.paired_bootstrap(per_user_mean("two_tower", "recall"), per_user_mean("ease", "recall"),
                            "recall")
    tt_sig_better = b["ci_low"] > 0
    gap = (med.loc["ease", "recall@10"] - med.loc["two_tower", "recall@10"]) / med.loc["ease", "recall@10"]
    rows.append((2, (not tt_sig_better) and gap <= 0.10,
                 f"(EASE - two_tower)/EASE Recall@10 = {gap:+.3f}; two_tower - EASE = "
                 f"{b['mean_diff']:+.4f} [{b['ci_low']:+.4f}, {b['ci_high']:+.4f}]"))
    # 3. ranking stage changes NDCG@10 by -1% .. +5% (refuted if > +5%, or < -1% and sig. worse)
    b3 = ev.paired_bootstrap(per_user_mean("two_stage", "ndcg"), per_user_mean("two_tower", "ndcg"),
                             "ndcg")
    rel = (med.loc["two_stage", "ndcg@10"] - med.loc["two_tower", "ndcg@10"]) / med.loc["two_tower", "ndcg@10"]
    ok3 = -0.01 <= rel <= 0.05 or (rel < -0.01 and not b3["ci_high"] < 0)
    rows.append((3, ok3, f"(two_stage - two_tower)/two_tower NDCG@10 = {rel:+.3f}; diff "
                         f"{b3['mean_diff']:+.4f} [{b3['ci_low']:+.4f}, {b3['ci_high']:+.4f}]"))
    # 4. most-popular has the lowest coverage of all models
    others = med.drop(index="most_popular")["coverage"]
    rows.append((4, med.loc["most_popular", "coverage"] < others.min(),
                 f"most_popular {med.loc['most_popular', 'coverage']:.4f} vs lowest other "
                 f"{others.min():.4f} ({others.idxmin()})"))
    out = pd.DataFrame([{"id": i, "verdict": "confirmed" if ok else "refuted", "evidence": ev_}
                        for i, ok, ev_ in rows])
    STATUS_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(STATUS_CSV, index=False)
    for r in out.itertuples():
        log(f"  prediction #{r.id}: {r.verdict.upper()} - {r.evidence}")


if __name__ == "__main__":
    main()

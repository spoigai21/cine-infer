"""Phase 8: the retrain -> evaluate -> publish-if-better steps that the Airflow DAG runs.

Plain Python with a CLI, so every step is testable without Airflow and the DAG only wires them
together (dags/cineinfer_retrain.py). Each step reads and writes files under
models/candidates/<run_id>/, so steps can run as separate processes (and as separate Airflow
tasks).

  train     two-tower on TRAIN with the frozen tuned config; `--epochs` can override the epoch
            count (a 1-epoch model is the deliberately worse candidate the DAG must reject)
  evaluate  full-validation NDCG@10 of that retriever (151,597 users, the Phase 2 harness)
  gate      compare with the live model's validation NDCG@10 from models/registry.json; writes
            decision.json ("publish" only if strictly better) and appends to
            results/publish_log.csv whatever the outcome
  publish   only after a "publish" decision: refit the same config on train + val, rebuild
            the ranker on its candidates (val labels, frozen config), export a bundle, smoke-test
            it in a torch-free subprocess, then point models/serving at it (atomic symlink swap)
            and update the registry
The live model's score is the validation NDCG@10 of the same config trained on train only
(results/two_tower.csv), which is exactly what `evaluate` measures for a candidate.

Usage: python -m src.pipeline {init-registry|train|evaluate|gate|publish} --run-id ID [--epochs N]
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "models"
REGISTRY = MODELS / "registry.json"
CANDIDATES = MODELS / "candidates"
BUNDLES = MODELS / "bundles"
SERVING = MODELS / "serving"
PUBLISH_LOG = REPO / "results" / "publish_log.csv"
SPLITS = REPO / "data" / "splits.parquet"
MOVIES = REPO / "data" / "ml-25m" / "movies.csv"
SEED = 42
LOG_COLUMNS = ["time", "run_id", "epochs", "candidate_val_ndcg@10", "live_val_ndcg@10",
               "live_model", "decision", "reason"]


def run_dir(run_id):
    d = CANDIDATES / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_registry():
    return json.loads(REGISTRY.read_text())


def write_json_atomic(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, path)


def frozen_two_tower_config():
    import pandas as pd
    b = pd.read_csv(REPO / "results" / "two_tower.csv")
    row = b[b.seed == SEED].iloc[0]
    return json.loads(row.config), float(row["ndcg@10"])


# ---------------------------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------------------------

def init_registry(force=False):
    """Register the model currently served (Phase 7's bundle) as live, if no registry exists."""
    if REGISTRY.exists() and not force:
        return load_registry()
    cfg, val_ndcg = frozen_two_tower_config()
    if SERVING.exists() and not SERVING.is_symlink():  # move the Phase 7 bundle under bundles/
        BUNDLES.mkdir(parents=True, exist_ok=True)
        target = BUNDLES / "phase7"
        os.replace(SERVING, target)
        os.symlink(target.relative_to(MODELS), SERVING)
    reg = {"live": {"model_id": "phase7", "val_ndcg@10": val_ndcg, "epochs": cfg["epochs"],
                    "bundle": str(os.readlink(SERVING)) if SERVING.is_symlink() else None,
                    "published": "phase 7 (results/two_tower.csv seed 42)"}, "history": []}
    MODELS.mkdir(exist_ok=True)
    write_json_atomic(REGISTRY, reg)
    return reg


def train(run_id, epochs=None):
    from src import evaluate as ev
    from src import two_tower as tt
    cfg, _ = frozen_two_tower_config()
    if epochs is not None:
        cfg["epochs"] = int(epochs)
    d = run_dir(run_id)
    items = ev.ItemIndex.from_movies_csv(MOVIES)
    seq = tt.load_sequences(SPLITS, items, ("train",))
    t = time.time()
    sc, info = tt.train_two_tower(seq, cfg, SEED, fixed_epochs=cfg["epochs"])
    tt.save_scorer(sc, d / "two_tower_train.pt", SEED)
    write_json_atomic(d / "candidate.json", {"run_id": run_id, "config": cfg, "seed": SEED,
                                             "device": info["device"],
                                             "train_seconds": round(time.time() - t, 1)})


def evaluate(run_id):
    from src import evaluate as ev
    from src import two_tower as tt
    d = run_dir(run_id)
    items = ev.ItemIndex.from_movies_csv(MOVIES)
    seq = tt.load_sequences(SPLITS, items, ("train",))
    sc = tt.load_scorer(d / "two_tower_train.pt", seq)
    val = ev.build_eval_data(ev.load_ratings(SPLITS, "user"), items, "user", "val")
    s = ev.evaluate(sc, val).summary
    write_json_atomic(d / "metrics.json", s)
    print(f"candidate {run_id}: val ndcg@10 {s['ndcg@10']:.5f}, recall@10 {s['recall@10']:.5f}")


def gate(run_id):
    d = run_dir(run_id)
    cand = json.loads((d / "metrics.json").read_text())["ndcg@10"]
    epochs = json.loads((d / "candidate.json").read_text())["config"]["epochs"]
    live = load_registry()["live"]
    better = cand > live["val_ndcg@10"]
    decision = "publish" if better else "reject"
    reason = (f"candidate {cand:.5f} {'>' if better else '<='} live {live['val_ndcg@10']:.5f} "
              f"validation NDCG@10")
    write_json_atomic(d / "decision.json", {"decision": decision, "reason": reason,
                                            "candidate_val_ndcg@10": cand,
                                            "live_val_ndcg@10": live["val_ndcg@10"],
                                            "live_model": live["model_id"]})
    append_log({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "run_id": run_id, "epochs": epochs,
                "candidate_val_ndcg@10": f"{cand:.6f}",
                "live_val_ndcg@10": f"{live['val_ndcg@10']:.6f}", "live_model": live["model_id"],
                "decision": decision, "reason": reason})
    print(f"gate: {decision.upper()} - {reason}")
    return decision


def publish(run_id):
    d = run_dir(run_id)
    decision = json.loads((d / "decision.json").read_text())
    if decision["decision"] != "publish":
        raise RuntimeError(f"refusing to publish {run_id}: gate decided {decision['decision']}")
    build_bundle(run_id, d)
    swap_live(run_id, BUNDLES / run_id, decision["candidate_val_ndcg@10"],
              json.loads((d / "candidate.json").read_text())["config"]["epochs"])


def build_bundle(run_id, d):
    """Refit on train + val, rebuild the ranker for that retriever, export, smoke-test."""
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    from src import baselines as bl
    from src import evaluate as ev
    from src import ranker as rk
    from src import two_tower as tt
    from src.export_serving import export_bundle
    from src.tune_ranker import HEADLINE
    cfg = json.loads((d / "candidate.json").read_text())["config"]
    items = ev.ItemIndex.from_movies_csv(MOVIES)
    seq_tv = tt.load_sequences(SPLITS, items, ("train", "val"))
    sc, _ = tt.train_two_tower(seq_tv, cfg, SEED, fixed_epochs=cfg["epochs"])
    tt.save_scorer(sc, d / "two_tower_tv.pt", SEED)
    # ranker: candidates from the candidate's train-only retriever, train features, val labels
    b = pd.read_csv(REPO / "results" / "baselines.csv")
    ease_cfg = json.loads(b[b.model == "ease"].iloc[0].config)
    rcfg = json.loads(pd.read_csv(REPO / "results" / "two_stage.csv").query("model == 'two_stage'").iloc[0].config)
    fine = ev.load_ratings(SPLITS, "user", fine=True)
    seq = tt.load_sequences(SPLITS, items, ("train",))
    td = bl.build_train_data(fine, items, ("train_core", "train_tail"))
    ctx = rk.TrainingSetContext.build(
        "train", tt.load_scorer(d / "two_tower_train.pt", seq),
        bl.EASE(td, bl.ItemGram(td, ease_cfg["min_pos"]), ease_cfg["lam"]), seq, fine,
        ("train_core", "train_tail"), REPO / "data" / "features", items,
        rk.ItemContent.load(MOVIES, REPO / "data" / "features" / "genome.parquet", items))
    val_pos = fine[(fine.split == "val") & (fine.rating >= 4)]
    L = sp.csr_matrix((np.ones(len(val_pos), dtype=np.int8),
                       (seq.rows(val_pos.userId.to_numpy()), items.to_index(val_pos.movieId.to_numpy()))),
                      shape=(len(seq.user_ids), len(items)))
    eligible = seq.user_ids[(seq.lengths > 0) & (np.diff(L.indptr) > 0)]
    users = np.sort(np.random.default_rng(SEED).choice(eligible, 30_000, replace=False))
    rk.write_training_rows(ctx, users, L, seq.rows(users), d / "ranker")
    lgb_cfg = {k: v for k, v in rcfg.items() if k not in ("features", "rounds")}
    rk.run_lgb(d / "ranker", lgb_cfg, SEED, features=HEADLINE, fixed_rounds=int(rcfg["rounds"]),
               save_model=d / "ranker.txt", predict=False)
    bundle = BUNDLES / run_id
    export_bundle(bundle, d / "two_tower_tv.pt", d / "ranker.txt", ease_cfg, HEADLINE, seed=SEED)
    smoke_test(bundle)


def smoke_test(bundle):
    """Load the bundle in a torch-free process and serve a known and an unknown user."""
    code = ("import json, sys, numpy as np; from src.serving import Recommender\n"
            f"r = Recommender({str(bundle)!r}, threads=1)\n"
            "u = int(r.user_ids[r.offsets[1:] > r.offsets[:-1]][0])\n"
            "a, b = r.recommend(u, 10), r.recommend(-1, 10)\n"
            "assert a['strategy'] == 'two_stage' and len(a['items']) == 10, a\n"
            "assert b['strategy'] == 'popularity_fallback' and len(b['items']) == 10, b\n"
            "assert 'torch' not in sys.modules\n"
            "print(json.dumps({'ok': True}))")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO)
    if r.returncode != 0:
        raise RuntimeError(f"bundle smoke test failed:\n{r.stderr[-2000:]}")


def swap_live(run_id, bundle, val_ndcg, epochs):
    """Point models/serving at the new bundle atomically, then update the registry."""
    reg = load_registry()
    tmp = MODELS / f".serving.{run_id}.tmp"
    if tmp.is_symlink() or tmp.exists():
        tmp.unlink()
    os.symlink(Path(bundle).relative_to(MODELS), tmp)
    os.replace(tmp, SERVING)  # atomic: the API never sees a half-published bundle
    reg["history"].append(reg["live"])
    reg["live"] = {"model_id": run_id, "val_ndcg@10": val_ndcg, "epochs": epochs,
                   "bundle": str(Path(bundle).relative_to(MODELS)),
                   "published": time.strftime("%Y-%m-%d %H:%M:%S")}
    write_json_atomic(REGISTRY, reg)
    print(f"published {run_id}; restart the API (make serve) to load it")


def append_log(row):
    PUBLISH_LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not PUBLISH_LOG.exists()
    with open(PUBLISH_LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_COLUMNS, lineterminator="\n")
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["init-registry", "train", "evaluate", "gate", "publish"])
    p.add_argument("--run-id", default=None)
    p.add_argument("--epochs", type=int, default=None)
    a = p.parse_args()
    if a.step == "init-registry":
        print(json.dumps(init_registry(), indent=1))
        return
    if not a.run_id:
        p.error("--run-id is required")
    run_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in a.run_id)
    {"train": lambda: train(run_id, a.epochs), "evaluate": lambda: evaluate(run_id),
     "gate": lambda: gate(run_id), "publish": lambda: publish(run_id)}[a.step]()


if __name__ == "__main__":
    main()

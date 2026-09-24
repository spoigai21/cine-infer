"""Phase 7: training/serving skew check.

The server (src/serving.py) recomputes candidates and ranker features for one user at a time in
NumPy. The batch pipeline (src/ranker.py) computed the same things for the test users when the
test results were scored (data/ranker_test/seed<s>/). This compares them on a sample of users:
candidates, every headline feature, and the final top-10 (server's ranker vs the batch ranker's
saved predictions). Never imports torch.

Usage: python -m scripts.serving_parity --bundle models/serving --eval-dir data/ranker_test/seed42 \
          --pred data/ranker_test/seed42/pred_two_stage.npy [--n 2000] [--out results/serving_parity.csv]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.serving import Recommender


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", required=True, type=Path)
    p.add_argument("--eval-dir", required=True, type=Path)
    p.add_argument("--pred", required=True, type=Path)
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args(argv)
    rec = Recommender(a.bundle)
    names = json.loads((a.eval_dir / "meta.json").read_text())["features"]
    cols = [names.index(f) for f in rec.features]
    users = np.load(a.eval_dir / "eval_users.npy")
    cand_b = np.load(a.eval_dir / "eval_cand.npy", mmap_mode="r")
    X_b = np.load(a.eval_dir / "eval_X.npy", mmap_mode="r")
    pred_b = np.load(a.pred, mmap_mode="r")
    rows = np.random.default_rng(a.seed).choice(len(users), min(a.n, len(users)), replace=False)
    same_cand, overlap_cand, same_topk, overlap_topk, max_diff = [], [], [], [], np.zeros(len(cols))
    for i in rows:
        r = rec.row(int(users[i]))
        cb = np.asarray(cand_b[i])
        valid = cb >= 0
        cb = cb[valid]
        cs, tts = rec.retrieve(r)
        same_cand.append(np.array_equal(cs, cb))
        overlap_cand.append(len(np.intersect1d(cs, cb)) / max(len(cb), 1))
        Xs = rec.features_for(r, cs, tts)
        pos_b = {int(c): j for j, c in enumerate(cb)}
        common = [(j, pos_b[int(c)]) for j, c in enumerate(cs) if int(c) in pos_b]
        if common:
            js, jb = map(list, zip(*common))
            xs = Xs[js].astype(np.float64)
            xb = np.asarray(X_b[i])[valid][jb][:, cols].astype(np.float64)
            both_nan = np.isnan(xs) & np.isnan(xb)
            d = np.where(both_nan, 0.0, np.abs(xs - xb))
            d[np.isnan(d)] = np.inf  # NaN on one side only = mismatch
            if not np.array_equal(cs, cb):  # ranks differ by construction when sets differ
                d[:, [rec.features.index(f) for f in ("tt_rank", "ease_rank") if f in rec.features]] = 0
            max_diff = np.maximum(max_diff, d.max(0))
        top_s = [it["movieId"] for it in rec.recommend(int(users[i]), a.k)["items"]]
        pb = np.asarray(pred_b[i])[valid]
        top_b = rec.movie_ids[cb[np.lexsort((cb, -pb))][:a.k]].tolist()
        same_topk.append(top_s == top_b)
        overlap_topk.append(len(set(top_s) & set(top_b)) / a.k)
    summary = {"users_checked": int(len(rows)),
               "candidates_identical_share": float(np.mean(same_cand)),
               "candidates_overlap_mean": float(np.mean(overlap_cand)),
               f"top{a.k}_identical_share": float(np.mean(same_topk)),
               f"top{a.k}_overlap_mean": float(np.mean(overlap_topk)),
               "max_abs_feature_diff": dict(zip(rec.features, map(float, max_diff))),
               "torch_loaded": "torch" in sys.modules}
    if a.out:
        import csv
        a.out.parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["check", "value"])
            for k, v in summary.items():
                if isinstance(v, dict):
                    for fk, fv in v.items():
                        w.writerow([f"max_abs_diff:{fk}", f"{fv:.3g}"])
                else:
                    w.writerow([k, f"{v:.6f}" if isinstance(v, float) else v])
    print(json.dumps(summary))


if __name__ == "__main__":
    main()

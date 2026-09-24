"""Validation metrics broken down by where each user's train/val boundary falls.

The per-user split cuts inside a same-second rating burst for ~19% of users, and within an hour
for most of the rest (Phase 1). Sequence-aware models (the two-tower's recency history) gain most
where validation is the continuation of the last training session. This table makes that
visible per model. Every seed's per-user metrics (data/eval_runs/) are averaged per user first.

Output: results/analysis/val_by_boundary.csv. Usage: `make analysis`.
"""
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

RUNS = Path("data/eval_runs")
OUT = Path("results/analysis/val_by_boundary.csv")
MODELS = ["most_popular", "item_knn", "implicit_als", "ease", "ease_recent", "two_tower"]


def boundaries():
    b = duckdb.sql("""
        SELECT userId,
               max(timestamp) FILTER (split_user = 'train') AS train_max,
               min(timestamp) FILTER (split_user = 'val')   AS val_min
        FROM 'data/splits.parquet/*.parquet' GROUP BY userId""").df()
    gap = b.val_min - b.train_max
    b["boundary"] = np.select([gap == 0, gap <= 3600], ["same_second", "within_1h"], "over_1h")
    return b[["userId", "boundary"]]


def main():
    b = boundaries()
    rows = []
    for model in MODELS:
        files = sorted(RUNS.glob(f"{model}_user_val_seed*.parquet"))
        if not files:
            continue
        per_user = (pd.concat(pd.read_parquet(f) for f in files)
                    .groupby("userId")[["ndcg", "recall"]].mean().reset_index())
        m = per_user.merge(b, on="userId")
        for boundary, g in [("all", m)] + list(m.groupby("boundary")):
            rows.append({"model": model, "boundary": boundary, "users": len(g),
                         "seeds": len(files), "ndcg@10": g.ndcg.mean(),
                         "recall@10": g.recall.mean()})
    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False, float_format="%.6f")
    print(out.pivot(index="model", columns="boundary", values="ndcg@10")
             .reindex(MODELS).round(4).to_string())


if __name__ == "__main__":
    main()

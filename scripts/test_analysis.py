"""Phase 10: test-slice comparisons that need the per-user metrics (data/eval_runs/, git-ignored),
written to committed CSVs so the README (and its --check in CI) never needs the dataset.

  results/test_gaps.csv                  paired bootstrap 95% CIs for the key gaps between models
                                         (per-user metrics averaged over each model's seeds first)
  results/analysis/test_by_boundary.csv  test NDCG@10 / Recall@10 by where each user's train+val /
                                         test boundary falls (same second / within 1 h / over 1 h)
Reads only sealed test results; scores nothing. Usage: `make test-analysis`.
"""
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src import evaluate as ev

RUNS = Path("data/eval_runs")
GAPS = Path("results/test_gaps.csv")
BOUNDARY = Path("results/analysis/test_by_boundary.csv")
MODELS = ["most_popular", "item_knn", "implicit_als", "ease", "ease_recent", "two_tower",
          "two_stage_no_ease", "two_stage", "two_stage_with_time"]
PAIRS = [("two_tower", "ease"), ("two_stage", "two_tower"), ("two_stage", "ease"),
         ("two_stage_no_ease", "two_stage"), ("two_stage_with_time", "two_stage"),
         ("ease_recent", "ease"), ("ease", "implicit_als"), ("implicit_als", "item_knn"),
         ("item_knn", "most_popular")]


def per_user(model):
    files = sorted(RUNS.glob(f"{model}_user_test_seed*.parquet"))
    if not files:
        raise FileNotFoundError(f"no test per-user results for {model}")
    df = pd.concat(pd.read_parquet(f) for f in files)
    out = df.groupby("userId")[["ndcg", "recall"]].mean().reset_index()
    out.attrs["seeds"] = len(files)
    return out


def main():
    pu = {m: per_user(m) for m in MODELS}
    rows = []
    for a, b in PAIRS:
        for metric in ("ndcg", "recall"):
            r = ev.paired_bootstrap(pu[a], pu[b], metric, n_boot=2000, seed=0)
            rows.append({"a": a, "b": b, "metric": f"{metric}@10", "mean_a": pu[a][metric].mean(),
                         "mean_b": pu[b][metric].mean(), "diff": r["mean_diff"],
                         "ci_low": r["ci_low"], "ci_high": r["ci_high"],
                         "rel_diff": pu[a][metric].mean() / pu[b][metric].mean() - 1,
                         "n_users": r["n_users"]})
    GAPS.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(GAPS, index=False, float_format="%.6f")

    b = duckdb.sql("""
        SELECT userId,
               max(timestamp) FILTER (split_user IN ('train', 'val')) AS seen_max,
               min(timestamp) FILTER (split_user = 'test') AS test_min
        FROM 'data/splits.parquet/*.parquet' GROUP BY userId""").df()
    gap = b.test_min - b.seen_max
    b["boundary"] = np.select([gap == 0, gap <= 3600], ["same_second", "within_1h"], "over_1h")
    rows = []
    for m in MODELS:
        d = pu[m].merge(b[["userId", "boundary"]], on="userId")
        for boundary, g in [("all", d)] + list(d.groupby("boundary")):
            rows.append({"model": m, "boundary": boundary, "users": len(g),
                         "seeds": pu[m].attrs["seeds"], "ndcg@10": g.ndcg.mean(),
                         "recall@10": g.recall.mean()})
    BOUNDARY.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(BOUNDARY, index=False, float_format="%.6f")
    print(pd.read_csv(GAPS)[["a", "b", "metric", "diff", "ci_low", "ci_high", "rel_diff"]].round(4).to_string(index=False))
    print(pd.DataFrame(rows).pivot(index="model", columns="boundary", values="ndcg@10").reindex(MODELS).round(4).to_string())


if __name__ == "__main__":
    main()

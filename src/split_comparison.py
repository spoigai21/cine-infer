"""Phase 10: prediction #7, how much each split scheme inflates the test metrics.

EASE at its per-user-tuned config (results/baselines.csv, frozen) on each scheme's test slice:
  user     the sealed Phase 6b result (data/test_runs/ease_seed0.json), not re-scored
  global   refit on the global split's train + val (ratings up to t90), score its test slice once
  random   refit on the random split's train + val, score its test slice once
Each scheme uses its own population (>= 1 train positive and >= 1 test positive under that
scheme), mask (train + val) and coverage denominator (§2.1, Phase 2). Results are sealed exactly
like Phase 6b (final_test.score_once): each is scored once and never overwritten.

The populations differ (the global cutoff's test users are the few active after t90), so there
is no paired comparison: each scheme's NDCG@10 gets a bootstrap CI over its own users, and the
random/global ratio gets a CI from independent resamples of the two populations.

Prediction #7 (committed): NDCG@10 random >= 1.5 x global, and random > per-user > global.
Outputs: results/split_comparison.csv, row 7 of results/predictions_status.csv (sealed once
settled). Usage: `make split-comparison`.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import baselines as bl
from src import evaluate as ev
from src import final_test as ft

SPLITS = Path("data/splits.parquet")
MOVIES = Path("data/ml-25m/movies.csv")
OUT = Path("results/split_comparison.csv")
STATUS = Path("results/predictions_status.csv")
SCHEMES = {"user": "ease", "global": "ease_global", "random": "ease_random"}
N_BOOT = 2000


def score_scheme(scheme, name, items, cfg):
    if ft.done(name, 0):
        print(f"  {scheme}: sealed result exists, not re-scored", flush=True)
        return
    t = time.time()
    r = ev.load_ratings(SPLITS, scheme)
    test = ev.build_eval_data(r, items, scheme, "test")
    td = bl.build_train_data(r, items, ("train", "val"))
    m = bl.EASE(td, bl.ItemGram(td, cfg["min_pos"]), cfg["lam"])
    print(f"  {scheme}: {test.n_users} test users, {td.X.nnz} train+val positives", flush=True)
    ft.score_once(name, 0, m, test, time.time() - t)


def boot_means(x, rng, n=N_BOOT):
    x = np.asarray(x)
    return np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])


def summarise():
    rng = np.random.default_rng(0)
    rows, boots = [], {}
    for scheme, name in SCHEMES.items():
        run = json.loads((ft.RUNS / f"{name}_seed0.json").read_text())
        pu = pd.read_parquet(ft.PER_USER / f"{name}_user_test_seed0.parquet")
        boots[scheme] = boot_means(pu.ndcg, rng)
        lo, hi = np.percentile(boots[scheme], [2.5, 97.5])
        rows.append({"scheme": scheme, "model": "ease", "n_users": run["n_users"],
                     "ndcg@10": run["ndcg@10"], "ndcg_ci_low": lo, "ndcg_ci_high": hi,
                     "recall@10": run["recall@10"], "coverage": run["coverage"],
                     "coverage_denominator": run["coverage_denominator"],
                     "excluded_no_train_positive": run["excluded_no_train_positive"],
                     "excluded_no_target_positive": run["excluded_no_target_positive"]})
    df = pd.DataFrame(rows)
    ratio = boots["random"] / boots["global"]
    nd = df.set_index("scheme")["ndcg@10"]
    df["ratio_vs_global"] = nd.reindex(df.scheme).to_numpy() / nd["global"]
    lo, hi = np.percentile(ratio, [2.5, 97.5])
    df.loc[df.scheme == "random", ["ratio_ci_low", "ratio_ci_high"]] = [lo, hi]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, float_format="%.6f")
    return df


def settle(df):
    status = pd.read_csv(STATUS)
    prior = status[status.id == 7]
    if len(prior) and prior.verdict.iloc[0] in ("confirmed", "refuted"):
        print(f"prediction #7 already settled ({prior.verdict.iloc[0]}); not changed")
        return
    d = df.set_index("scheme")
    r, u, g = d.loc["random", "ndcg@10"], d.loc["user", "ndcg@10"], d.loc["global", "ndcg@10"]
    ratio = r / g
    ok = ratio >= 1.5 and r > u > g
    evidence = (f"EASE test NDCG@10: random {r:.4f}, per-user {u:.4f}, global {g:.4f}; "
                f"random/global = {ratio:.2f} (95% CI {d.loc['random', 'ratio_ci_low']:.2f}-"
                f"{d.loc['random', 'ratio_ci_high']:.2f}; needs >= 1.5); order "
                f"{' > '.join(d['ndcg@10'].sort_values(ascending=False).index)}")
    verdict = "confirmed" if ok else "refuted"
    status = pd.concat([status[status.id != 7],
                        pd.DataFrame([{"id": 7, "verdict": verdict, "evidence": evidence}])])
    status.sort_values("id").to_csv(STATUS, index=False)
    print(f"prediction #7: {verdict.upper()} - {evidence}")


def main():
    t0 = time.time()
    items = ev.ItemIndex.from_movies_csv(MOVIES)
    b = pd.read_csv("results/baselines.csv")
    cfg = json.loads(b[b.model == "ease"].iloc[0].config)
    print(f"EASE frozen config {cfg}", flush=True)
    for scheme, name in SCHEMES.items():
        score_scheme(scheme, name, items, cfg)
    ft.write_test_csv()
    df = summarise()
    print(df.to_string(index=False))
    settle(df)
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

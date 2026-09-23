"""Run the Phase 2 harness on the real data with two reference models.

Oracle (scores each user's relevant items highest) must score exactly 1.0 on recall, NDCG and
AUC. Random must land near chance. The population must match results/data_stats.csv. Any
failure exits non-zero. Usage: `make eval-check`.
"""
import sys
import time

import pandas as pd

from src import evaluate as ev

failures = []


def check(name, cond, detail):
    print(f"{'PASS' if cond else 'FAIL'} {name}: {detail}")
    if not cond:
        failures.append(name)


items = ev.ItemIndex.from_movies_csv("data/ml-25m/movies.csv")
stats = pd.read_csv("results/data_stats.csv").set_index(["scheme", "stat"])["value"]
check("catalog size", len(items) == stats[("all", "movies_total")], len(items))

t = time.time()
ratings = ev.load_ratings("data/splits.parquet", "user")
print(f"loaded ratings ({time.time() - t:.1f}s)")

for slice_ in ev.SLICES:
    t = time.time()
    d = ev.build_eval_data(ratings, items, "user", slice_)
    print(f"\n[user/{slice_}] built eval data ({time.time() - t:.1f}s): {d.counts}")
    expected = stats[("user", f"users_eligible_with_{slice_}_positive")]
    check("population matches data_stats.csv", d.n_users == expected, (d.n_users, expected))
    if slice_ == "val":
        check("coverage denominator = movies with a train rating",
              d.counts["coverage_denominator"] == stats[("all", "movies_with_train_rating")],
              d.counts["coverage_denominator"])

    t = time.time()
    o = ev.evaluate(ev.OracleModel(d), d).summary
    print(f"  oracle ({time.time() - t:.1f}s): {o}")
    check("oracle recall/NDCG/AUC = 1", o["recall@10"] == o["ndcg@10"] == o["auc"] == 1.0,
          (o["recall@10"], o["ndcg@10"], o["auc"]))

    t = time.time()
    r = ev.evaluate(ev.RandomModel(len(items)), d).summary
    print(f"  random ({time.time() - t:.1f}s): {r}")
    check("random recall near 0", r["recall@10"] < 0.005, r["recall@10"])
    check("random AUC near 0.5", abs(r["auc"] - 0.5) < 0.01, r["auc"])

print("\nALL CHECKS PASSED" if not failures else f"\nFAILED: {failures}")
sys.exit(1 if failures else 0)

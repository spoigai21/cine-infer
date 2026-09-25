"""Phase 9: pandas vs Spark vs Polars vs DuckDB on the Phase 1 feature pipeline, and prediction #5.

1. Inputs: data/bench/ratings_{1M,5M,25M}.parquet from MovieLens ratings.csv. Smaller sizes take
   whole users in a seeded random order until the row target is reached, so every user's history
   is intact (the per-user split needs it). 25M is the full dataset.
2. Runs: REPEATS rounds; in each round every (size, engine) runs once in a fresh process
   (src/bench_pipeline.py), with the engine order rotated between rounds. Each run records
   startup, cold compute and warm compute, plus the power source and load average just before it.
3. Correctness: at every size, each engine's output must equal pandas' (bench_pipeline.compare_outputs).
   Timings from engines that disagree are worthless, so any mismatch stops the benchmark.
4. Outputs: results/benchmark_runs.csv (every run), results/benchmark.csv (medians),
   results/benchmark.png (two panels: with startup, without startup).
5. Prediction #5 (committed): pandas beats Spark at 1M, 5M and 25M rows, both "with startup"
   (startup + cold compute) and "without startup" (warm compute). It is settled only if every run
   was on AC power with a 1-minute load average <= LOAD_LIMIT before it (conditions fixed before
   the benchmark ran); otherwise the verdict says why it isn't settled. Once settled, it is
   sealed like #6.

Usage: `make benchmark` (~30 min). Needs the laptop plugged in and otherwise idle.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RATINGS = ROOT / "data" / "ml-25m" / "ratings.csv"
MOVIES = ROOT / "data" / "ml-25m" / "movies.csv"
BENCH = ROOT / "data" / "bench"
SIZES = {"1M": 1_000_000, "5M": 5_000_000, "25M": None}
ENGINES = ["pandas", "spark", "polars", "duckdb"]
REPEATS = 3
THREADS = 6
LOAD_LIMIT = 3.0  # 1-min load average on 10 cores, checked before every run
RUNS_CSV = ROOT / "results" / "benchmark_runs.csv"
SUMMARY_CSV = ROOT / "results" / "benchmark.csv"
CHART = ROOT / "results" / "benchmark.png"
STATUS = ROOT / "results" / "predictions_status.csv"
SEED = 9


def power_source():
    try:
        out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
        return "AC" if "AC Power" in out else "battery" if "Battery Power" in out else "unknown"
    except FileNotFoundError:
        return "unknown"


def make_inputs():
    import duckdb
    BENCH.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"""CREATE TEMP TABLE r AS SELECT userId::INTEGER AS userId, movieId::INTEGER AS movieId,
                    rating::DOUBLE AS rating, timestamp::BIGINT AS timestamp
                    FROM read_csv('{RATINGS}', header = true)""")
    counts = con.execute("SELECT userId, count(*) AS n FROM r GROUP BY userId ORDER BY userId").df()
    order = np.random.default_rng(SEED).permutation(len(counts))
    cum = counts.n.to_numpy()[order].cumsum()
    rows = {}
    for name, target in SIZES.items():
        path = BENCH / f"ratings_{name}.parquet"
        if target is None:
            keep = counts.userId.to_numpy()
        else:
            keep = counts.userId.to_numpy()[order][: int(np.searchsorted(cum, target) + 1)]
        con.register("keep", pd.DataFrame({"userId": keep.astype("int32")}))
        if not path.exists():
            con.execute(f"""COPY (SELECT r.* FROM r JOIN keep USING (userId))
                            TO '{path}' (FORMAT PARQUET, ROW_GROUP_SIZE 1000000)""")
        rows[name] = con.execute(f"SELECT count(*) FROM read_parquet('{path}')").fetchone()[0]
    return rows


def run_one(engine, size):
    inp = BENCH / f"ratings_{size}.parquet"
    load = os.getloadavg()[0]
    power = power_source()
    res = subprocess.run([sys.executable, "-m", "src.bench_pipeline", "--engine", engine,
                          "--input", str(inp), "--movies", str(MOVIES),
                          "--out", str(BENCH / "out" / size / engine), "--threads", str(THREADS)],
                         capture_output=True, text=True, cwd=ROOT, env=os.environ.copy())
    if res.returncode != 0:
        raise RuntimeError(f"{engine} @ {size} failed:\n{res.stderr[-3000:]}")
    t = json.loads(res.stdout.strip().splitlines()[-1])
    return {"engine": engine, "size": size, "startup_s": t["startup_s"],
            "cold_compute_s": t["cold_compute_s"], "warm_compute_s": t["warm_compute_s"],
            "loadavg_1m_before": round(load, 2), "power": power,
            "time": time.strftime("%Y-%m-%d %H:%M:%S")}


def check_agreement():
    from src.bench_pipeline import compare_outputs
    for size in SIZES:
        for e in ENGINES[1:]:
            compare_outputs(BENCH / "out" / size / "pandas" / "warm", BENCH / "out" / size / e / "warm")
        print(f"  {size}: all engines agree with pandas", flush=True)


def summarise(runs, rows):
    runs = runs.assign(with_startup_s=runs.startup_s + runs.cold_compute_s)
    med = (runs.groupby(["size", "engine"])[["startup_s", "cold_compute_s", "warm_compute_s",
                                            "with_startup_s"]].median().reset_index())
    med["rows"] = med["size"].map(rows)
    med["repeats"] = runs.groupby(["size", "engine"]).size().to_numpy()
    med = med.sort_values(["rows", "engine"])
    med.to_csv(SUMMARY_CSV, index=False, float_format="%.3f")
    return med


def chart(med):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"pandas": "#4C72B0", "spark": "#DD8452", "polars": "#55A868", "duckdb": "#8172B3"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, col, title in ((axes[0], "with_startup_s", "With startup (startup + cold compute)"),
                           (axes[1], "warm_compute_s", "Without startup (warm compute)")):
        for e in ENGINES:
            d = med[med.engine == e].sort_values("rows")
            ax.plot(d.rows, d[col], marker="o", label=e, color=colors[e])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("ratings (rows)")
        ax.set_title(title, fontsize=10)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("seconds (median of repeats)")
    axes[0].legend()
    fig.suptitle(f"Phase 1 feature pipeline on one laptop ({THREADS} threads where configurable)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(CHART, dpi=130)


def settle(runs, med):
    status = pd.read_csv(STATUS) if STATUS.exists() else pd.DataFrame(columns=["id", "verdict", "evidence"])
    prior = status[status.id == 5]
    if len(prior) and prior.verdict.iloc[0] in ("confirmed", "refuted"):
        print(f"prediction #5 already settled ({prior.verdict.iloc[0]}); not changed")
        return
    parts, ok = [], True
    for size in SIZES:
        m = med[med["size"] == size].set_index("engine")
        for col, label in (("with_startup_s", "with startup"), ("warm_compute_s", "without startup")):
            pd_t, sp_t = m.loc["pandas", col], m.loc["spark", col]
            ok &= pd_t < sp_t
            parts.append(f"{size} {label}: pandas {pd_t:.2f}s vs spark {sp_t:.2f}s")
    clean = (runs.power == "AC").all() and (runs.loadavg_1m_before <= LOAD_LIMIT).all()
    if clean:
        verdict = "confirmed" if ok else "refuted"
    else:
        verdict = (f"not settled (power {sorted(runs.power.unique())}, max load "
                   f"{runs.loadavg_1m_before.max():.1f} > {LOAD_LIMIT})")
    evidence = "; ".join(parts)
    status = pd.concat([status[status.id != 5],
                        pd.DataFrame([{"id": 5, "verdict": verdict, "evidence": evidence}])])
    status.sort_values("id").to_csv(STATUS, index=False)
    print(f"prediction #5: {verdict.upper()} - {evidence}")


def main():
    t0 = time.time()
    if power_source() != "AC" or os.getloadavg()[0] > LOAD_LIMIT:
        print(f"WARNING: power {power_source()}, load {os.getloadavg()[0]:.1f}: #5 will not be "
              f"settled from this run (needs AC and load <= {LOAD_LIMIT})", flush=True)
    rows = make_inputs()
    print(f"inputs: {rows}", flush=True)
    runs = []
    for rep in range(REPEATS):
        for size in SIZES:
            order = ENGINES[rep % len(ENGINES):] + ENGINES[:rep % len(ENGINES)]
            for e in order:
                r = run_one(e, size)
                r["repeat"] = rep + 1
                runs.append(r)
                pd.DataFrame(runs).to_csv(RUNS_CSV, index=False, float_format="%.4f")
                print(f"  [{rep + 1}/{REPEATS}] {size:>3} {e:<7} startup {r['startup_s']:6.2f}s  "
                      f"cold {r['cold_compute_s']:7.2f}s  warm {r['warm_compute_s']:7.2f}s  "
                      f"(load {r['loadavg_1m_before']}, {r['power']})", flush=True)
    check_agreement()
    runs = pd.DataFrame(runs)
    med = summarise(runs, rows)
    chart(med)
    print(med.to_string(index=False))
    settle(runs, med)
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

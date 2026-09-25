"""Phase 7: latency of the running API, and prediction #6.

Starts `uvicorn src.serve:app` in a subprocess, waits for the model to load, then sends
sequential HTTP requests from one client (concurrency 1):
  - 50 warm-up requests (discarded)
  - 1,000 requests for random test users (known users -> the two-stage path)
  - 200 requests for unknown user IDs (the popularity fallback)
Client-side latency is measured around each HTTP call; the server's own per-stage timings
(retrieval / features / ranking) come back in each response.

Prediction #6 (committed): two-stage p99 < 25 ms and p50 < 10 ms, "1,000 sequential HTTP
requests to uvicorn on localhost, after 50 warm-up requests, laptop plugged in". The power
source is recorded; if it isn't AC, the verdict is marked as not settled.

Outputs: results/latency.csv (scheme,stat,value), and row 6 of results/predictions_status.csv
the first time only: once #6 is confirmed or refuted it is sealed, and later runs (e.g. after
serving changes) only update results/latency.csv. The run that settled #6 is kept as
results/latency_prediction6.csv. Usage: `make load-test`.
"""
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

PORT = 8765
USERS = Path("data/ranker_test/seed42/eval_users.npy")  # the test population
OUT = Path("results/latency.csv")
STATUS = Path("results/predictions_status.csv")


def power_source():
    try:
        out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
        return "AC" if "AC Power" in out else "battery" if "Battery Power" in out else "unknown"
    except FileNotFoundError:
        return "unknown"


def cpu_name():
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                              text=True).stdout.strip() or platform.processor()
    except FileNotFoundError:
        return platform.processor()


def pct(x, q):
    return float(np.percentile(np.asarray(x), q))


def main():
    env = {**os.environ, "CINEINFER_BUNDLE": os.environ.get("CINEINFER_BUNDLE", "models/serving")}
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "src.serve:app", "--port", str(PORT),
                               "--log-level", "warning"], env=env)
    base = f"http://127.0.0.1:{PORT}"
    try:
        with httpx.Client(base_url=base, timeout=30) as c:
            t0 = time.time()
            while True:
                try:
                    h = c.get("/health").json()
                    if h["model_loaded"]:
                        break
                    if h.get("detail") and "no model bundle" in h["detail"]:
                        raise SystemExit(h["detail"])
                except httpx.TransportError:
                    pass
                if time.time() - t0 > 300:
                    raise SystemExit("server did not load the model within 300 s")
                time.sleep(0.5)
            load_s = time.time() - t0
            rng = np.random.default_rng(0)
            known = rng.choice(np.load(USERS), 1050, replace=False)
            unknown = np.arange(10**9, 10**9 + 200)

            def run(ids):
                lat, server_t, strategies = [], [], []
                for u in ids:
                    t = time.perf_counter()
                    r = c.get(f"/recommend/{int(u)}")
                    lat.append((time.perf_counter() - t) * 1e3)
                    r.raise_for_status()
                    body = r.json()
                    server_t.append(body["timings_ms"])
                    strategies.append(body["strategy"])
                return lat, server_t, strategies

            run(known[:50])  # warm-up, discarded
            lat, st, strat = run(known[50:])
            lat_c, st_c, strat_c = run(unknown)
    finally:
        server.terminate()
        server.wait(timeout=30)

    power = power_source()
    rows = [("setup", "requests_two_stage", len(lat)), ("setup", "requests_fallback", len(lat_c)),
            ("setup", "warmup_requests", 50), ("setup", "concurrency", 1),
            ("setup", "power_source", power), ("setup", "cpu", cpu_name()),
            ("setup", "platform", platform.platform()), ("setup", "model_load_seconds", round(load_s, 1)),
            ("setup", "date", time.strftime("%Y-%m-%d")),
            ("setup", "threads", os.environ.get("CINEINFER_THREADS", "default")),
            ("two_stage", "share_served_two_stage", float(np.mean([s == "two_stage" for s in strat])))]
    for scheme, l, s in (("two_stage", lat, st), ("popularity_fallback", lat_c, st_c)):
        for q in (50, 90, 99):
            rows.append((scheme, f"client_p{q}_ms", pct(l, q)))
        rows += [(scheme, "client_mean_ms", float(np.mean(l))), (scheme, "client_max_ms", float(np.max(l)))]
        for stage in ("retrieval_ms", "features_ms", "ranking_ms", "total_ms"):
            v = [t[stage] for t in s if stage in t]
            if v:
                rows += [(scheme, f"server_{stage[:-3]}_p50_ms", pct(v, 50)),
                         (scheme, f"server_{stage[:-3]}_p99_ms", pct(v, 99))]
    df = pd.DataFrame(rows, columns=["scheme", "stat", "value"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, float_format="%.3f")

    p50, p99 = pct(lat, 50), pct(lat, 99)
    ok = p99 < 25 and p50 < 10
    verdict = ("confirmed" if ok else "refuted") if power == "AC" else "not settled (not on AC power)"
    evidence = (f"two-stage client p50 {p50:.2f} ms (needs < 10), p99 {p99:.2f} ms (needs < 25); "
                f"1,000 sequential HTTP requests after 50 warm-up, power {power}")
    status = pd.read_csv(STATUS) if STATUS.exists() else pd.DataFrame(columns=["id", "verdict", "evidence"])
    prior = status[status.id == 6]
    print(df.to_string(index=False))
    if len(prior) and prior.verdict.iloc[0] in ("confirmed", "refuted"):
        print(f"prediction #6 already settled ({prior.verdict.iloc[0]}); not changed. "
              f"This run: {evidence}")
        return
    status = pd.concat([status[status.id != 6], pd.DataFrame([{"id": 6, "verdict": verdict,
                                                                "evidence": evidence}])])
    status.sort_values("id").to_csv(STATUS, index=False)
    print(f"prediction #6: {verdict.upper()} - {evidence}")


if __name__ == "__main__":
    main()

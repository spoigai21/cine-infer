"""Phase 9: the four engines compute exactly the same Phase 1 outputs (so the benchmark compares
like with like), and they match what src/data_prep.py produced for the fixture. Each engine runs
through the real timing harness in a subprocess."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src import bench_pipeline as bp
from src import data_prep as dp

FIXTURES = Path(__file__).resolve().parent / "fixtures"
ROOT = Path(__file__).resolve().parent.parent
ENGINES = ["pandas", "polars", "duckdb", "spark"]


@pytest.fixture(scope="module")
def runs(spark, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("bench")
    r = pd.read_csv(FIXTURES / "tiny_ratings.csv")
    r.to_parquet(tmp / "ratings.parquet", index=False)
    raw = tmp / "raw"
    raw.mkdir()
    (raw / "ratings.csv").write_text((FIXTURES / "tiny_ratings.csv").read_text())
    (raw / "movies.csv").write_text((FIXTURES / "tiny_movies.csv").read_text())
    dp.run(spark, raw, tmp / "phase1", tmp / "stats.csv", with_genome=False)
    timings = {}
    for e in ENGINES:
        res = subprocess.run([sys.executable, "-m", "src.bench_pipeline", "--engine", e,
                              "--input", str(tmp / "ratings.parquet"),
                              "--movies", str(FIXTURES / "tiny_movies.csv"),
                              "--out", str(tmp / e), "--threads", "2"],
                             capture_output=True, text=True, cwd=ROOT, env=os.environ.copy())
        assert res.returncode == 0, (e, res.stderr[-2000:])
        timings[e] = json.loads(res.stdout.strip().splitlines()[-1])
    return tmp, timings


@pytest.mark.parametrize("engine", ENGINES[1:])
def test_engines_agree_with_pandas(runs, engine):
    tmp, _ = runs
    for run in ("cold", "warm"):
        bp.compare_outputs(tmp / "pandas" / run, tmp / engine / run)


def test_outputs_match_phase1(runs):
    tmp, _ = runs
    got = bp.load_outputs(tmp / "pandas" / "cold")
    p1 = pd.read_parquet(tmp / "phase1" / "splits.parquet")
    m = got["splits"].merge(p1, on=["userId", "movieId"], suffixes=("", "_p1"))
    assert len(m) == len(p1) == 200
    assert (m.split_user == m.split_user_p1).all()
    assert (m.train_part.fillna("-") == m.train_part_p1.fillna("-")).all()
    for t in ("items", "users", "user_genres"):
        ref = pd.read_parquet(tmp / "phase1" / "features" / "train" / f"{t}.parquet")
        keys = bp.TABLE_KEYS[t]
        mm = got[t].merge(ref, on=keys, suffixes=("", "_p1"))
        assert len(mm) == len(ref) == len(got[t]), t
        for c in ref.columns:
            if c not in keys:
                assert ((mm[c] - mm[f"{c}_p1"]).abs() < 1e-9).all(), (t, c)


def test_timing_harness_reports_all_phases(runs):
    _, timings = runs
    for e, t in timings.items():
        assert t["engine"] == e and t["threads"] == 2
        assert t["startup_s"] > 0 and t["cold_compute_s"] > 0 and t["warm_compute_s"] > 0
    assert timings["spark"]["startup_s"] > timings["pandas"]["startup_s"]  # JVM boot


def test_compare_outputs_detects_a_difference(runs, tmp_path):
    tmp, _ = runs
    import shutil
    shutil.copytree(tmp / "pandas" / "cold", tmp_path / "bad")
    u = pd.read_parquet(tmp_path / "bad" / "users.parquet")
    u.loc[0, "n_positives"] += 1
    u.to_parquet(tmp_path / "bad" / "users.parquet", index=False)
    with pytest.raises(AssertionError):
        bp.compare_outputs(tmp / "pandas" / "cold", tmp_path / "bad")

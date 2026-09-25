"""Phase 0 checks: the fixture is well-formed and the dataset can't be committed."""
import csv
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"


def test_fixture_ratings_schema():
    r = pd.read_csv(FIXTURES / "tiny_ratings.csv")
    assert list(r.columns) == ["userId", "movieId", "rating", "timestamp"]
    assert len(r) == 200
    assert r["rating"].between(0.5, 5.0).all()
    assert (r["rating"] * 2 == (r["rating"] * 2).round()).all()  # half-star steps
    assert not r.duplicated(["userId", "movieId"]).any()


def test_fixture_has_edge_cases():
    r = pd.read_csv(FIXTURES / "tiny_ratings.csv")
    sizes = r.groupby("userId").size()
    assert sizes.min() < 10  # a user Phase 1 must route to train only
    assert r.duplicated(["userId", "timestamp"], keep=False).any()  # rating bursts
    movies = pd.read_csv(FIXTURES / "tiny_movies.csv")
    assert set(movies["movieId"]) - set(r["movieId"])  # a movie with no ratings


def test_titles_with_commas_parse():
    with open(FIXTURES / "tiny_movies.csv", newline="") as f:
        rows = list(csv.reader(f))
    assert all(len(row) == 3 for row in rows)
    assert any("," in row[1] for row in rows[1:])


def test_data_dir_is_gitignored():
    out = subprocess.run(["git", "check-ignore", "data/ml-25m/ratings.csv"],
                         cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, "data/ must be git-ignored (MovieLens license)"


def test_health_endpoint(tmp_path):
    # In a subprocess with no model bundle: the server loads LightGBM when a bundle exists, and
    # this test process has torch loaded (they segfault together on macOS). test_serving.py
    # covers the loaded-model case, also in subprocesses.
    import json
    import os
    import sys
    code = ("import json; from fastapi.testclient import TestClient; from src.serve import app\n"
            "with TestClient(app) as c: print(json.dumps(c.get('/health').json()))")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT,
                       env={**os.environ, "CINEINFER_BUNDLE": str(tmp_path / "none")})
    assert r.returncode == 0, r.stderr[-1500:]
    body = json.loads(r.stdout.strip().splitlines()[-1])
    assert body["status"] == "ok" and body["model_loaded"] is False

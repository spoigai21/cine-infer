"""Phase 7: export a bundle from the fixture, then check the server against the batch pipeline
(no training/serving skew), the API contract, the cold-start path, and that the server process
never loads torch. Server code runs in subprocesses: it loads LightGBM, and this test process
has torch loaded (they can't share a process on macOS)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

from src import data_prep as dp
from src import evaluate as ev
from src import ranker as rk
from src import two_tower as tt
from src.export_serving import export_bundle

FIXTURES = Path(__file__).resolve().parent / "fixtures"
ROOT = Path(__file__).resolve().parent.parent
HEADLINE = [f for f in rk.FEATURES if f not in ("item_days_since_last", "item_age_days")]
TV_FINE = ("train_core", "train_tail", "val")


@pytest.fixture(scope="module")
def built(spark, tmp_path_factory):
    out = tmp_path_factory.mktemp("serving")
    raw = out / "raw"
    raw.mkdir()
    (raw / "ratings.csv").write_text((FIXTURES / "tiny_ratings.csv").read_text())
    (raw / "movies.csv").write_text((FIXTURES / "tiny_movies.csv").read_text())
    dp.run(spark, raw, out, out / "stats.csv", with_genome=False)
    items = ev.ItemIndex.from_movies_csv(FIXTURES / "tiny_movies.csv")
    rng = np.random.default_rng(0)
    pd.DataFrame({"movieId": items.movie_ids[:25], "genome": list(rng.random((25, 6)))}).to_parquet(
        out / "genome.parquet")
    splits, movies = out / "splits.parquet", FIXTURES / "tiny_movies.csv"
    seq = tt.load_sequences(splits, items, ("train", "val"))
    sc, _ = tt.train_two_tower(seq, {"dim": 8, "hidden": 16, "batch": 16, "hist_len": 5}, 0,
                               fixed_epochs=2, log=lambda *_: None, dev=torch.device("cpu"))
    tt.save_scorer(sc, out / "tt.pt", 0)
    ease_cfg = {"lam": 5.0, "min_pos": 1}
    # batch pipeline on the same data: candidates + features (+ a ranker trained on test labels,
    # just to have a model; its quality is irrelevant here)
    from src import baselines as bl
    fine = ev.load_ratings(splits, "user", fine=True)
    td = bl.build_train_data(ev.load_ratings(splits, "user"), items, ("train", "val"))
    ctx = rk.TrainingSetContext.build(
        "train_val", tt.load_scorer(out / "tt.pt", seq),
        bl.EASE(td, bl.ItemGram(td, 1), 5.0), seq, fine, TV_FINE, out / "features", items,
        rk.ItemContent.load(movies, out / "genome.parquet", items))
    users = seq.user_ids[seq.lengths > 0]
    test_pos = fine[(fine.split == "test") & (fine.rating >= 4)]
    L = sp.csr_matrix((np.ones(len(test_pos)), (seq.rows(test_pos.userId.to_numpy()),
                                                 items.to_index(test_pos.movieId.to_numpy()))),
                      shape=(len(seq.user_ids), len(items)))
    batch = out / "batch"
    rk.write_training_rows(ctx, users, L, seq.rows(users), batch)
    rk.write_eval_features(ctx, users, batch)
    rk.run_lgb(batch, {"num_leaves": 4, "min_data_in_leaf": 1}, 0, features=HEADLINE,
               fixed_rounds=5, save_model=out / "ranker.txt", pred_out=batch / "pred.npy")
    bundle = out / "bundle"
    export_bundle(bundle, out / "tt.pt", out / "ranker.txt", ease_cfg, HEADLINE, splits=splits,
                  movies=movies, features_dir=out / "features", genome=out / "genome.parquet")
    return {"out": out, "bundle": bundle, "batch": batch, "users": users, "seq": seq}


def run_py(code_or_args, env=None):
    cmd = [sys.executable] + (code_or_args if isinstance(code_or_args, list) else ["-c", code_or_args])
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT,
                       env={**os.environ, **(env or {})})
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_server_matches_batch_pipeline(built):
    s = run_py(["-m", "scripts.serving_parity", "--bundle", str(built["bundle"]),
                "--eval-dir", str(built["batch"]), "--pred", str(built["batch"] / "pred.npy"),
                "--n", "1000", "--k", "5"])
    assert s["torch_loaded"] is False
    assert s["candidates_identical_share"] == 1.0
    assert s["top5_identical_share"] == 1.0
    assert max(s["max_abs_feature_diff"].values()) < 1e-4, s["max_abs_feature_diff"]


def test_api_contract_and_cold_start(built):
    known = int(built["users"][0])
    code = f"""
import json, sys
from fastapi.testclient import TestClient
from src.serve import app
with TestClient(app) as c:
    h = c.get("/health").json()
    r = c.get("/recommend/{known}?k=3").json()
    cold = c.get("/recommend/987654321").json()
    bad = c.get("/recommend/{known}?k=0").status_code
print(json.dumps({{"h": h, "r": r, "cold": cold, "bad": bad, "torch": "torch" in sys.modules}}))
"""
    o = run_py(code, env={"CINEINFER_BUNDLE": str(built["bundle"])})
    assert o["h"]["model_loaded"] is True and o["torch"] is False
    assert o["r"]["strategy"] == "two_stage" and len(o["r"]["items"]) == 3
    assert set(o["r"]["timings_ms"]) >= {"retrieval_ms", "features_ms", "ranking_ms", "total_ms"}
    assert o["cold"]["strategy"] == "popularity_fallback" and len(o["cold"]["items"]) == 10
    assert o["bad"] == 422
    # recommendations never include something the user already rated (train + val)
    seq = built["seq"]
    rated = pd.read_parquet(built["out"] / "splits.parquet")
    rated = set(rated[(rated.userId == known) & rated.split_user.isin(["train", "val"])].movieId)
    assert not rated & {it["movieId"] for it in o["r"]["items"]}


def test_popular_fallback_matches_train_val_popularity(built):
    code = """
import json
from src.serving import Recommender
import os
r = Recommender(os.environ["CINEINFER_BUNDLE"])
print(json.dumps([i["movieId"] for i in r.recommend(-1, 5)["items"]]))
"""
    got = run_py(code, env={"CINEINFER_BUNDLE": str(built["bundle"])})
    s = pd.read_parquet(built["out"] / "splits.parquet")
    pos = s[s.split_user.isin(["train", "val"]) & (s.rating >= 4)]
    counts = pos.groupby("movieId").size()
    ref = sorted(counts.index, key=lambda m: (-counts[m], m))[:5]
    assert got == ref


def test_health_without_bundle_reports_why(tmp_path):
    o = run_py("""
import json
from fastapi.testclient import TestClient
from src.serve import app
with TestClient(app) as c:
    print(json.dumps({"h": c.get("/health").json(), "code": c.get("/recommend/1").status_code}))
""", env={"CINEINFER_BUNDLE": str(tmp_path / "missing")})
    assert o["h"]["model_loaded"] is False and "make export" in o["h"]["detail"]
    assert o["code"] == 503

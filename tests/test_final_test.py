"""Phase 6b: test results are sealed (never re-scored), and predictions #1-#4 are settled
exactly as committed in results/predictions.csv."""
import json

import numpy as np
import pandas as pd
import pytest

from src import evaluate as ev
from src import final_test as ft


@pytest.fixture
def paths(tmp_path, monkeypatch):
    for name, sub in [("RUNS", "runs"), ("PER_USER", "eval_runs")]:
        monkeypatch.setattr(ft, name, tmp_path / sub)
    monkeypatch.setattr(ft, "TEST_CSV", tmp_path / "test.csv")
    monkeypatch.setattr(ft, "STATUS_CSV", tmp_path / "status.csv")
    return tmp_path


def toy_eval():
    items = ev.ItemIndex.from_ids(range(4))
    import scipy.sparse as sp
    rel = sp.csr_matrix(([1, 1], ([0, 1], [0, 1])), shape=(2, 4), dtype=np.int8)
    return ev.EvalData("user", "test", items, np.array([1, 2]), sp.csr_matrix((2, 4), dtype=np.int8),
                       rel, np.array([[2, 3], [2, 3]]), np.ones(4, bool), {})


class Const:
    config = {"x": 1}

    def score(self, users):
        return np.tile([4.0, 3.0, 2.0, 1.0], (len(users), 1))


def test_scores_are_sealed(paths):
    d = toy_eval()
    ft.score_once("m", 0, Const(), d, 0.0)
    assert ft.done("m", 0)
    with pytest.raises(RuntimeError, match="never re-scored"):
        ft.score_once("m", 0, Const(), d, 0.0)
    assert (paths / "eval_runs" / "m_user_test_seed0.parquet").exists()


def write_run(paths, model, seed, ndcg, recall, coverage, per_user_ndcg, per_user_recall):
    (paths / "runs").mkdir(exist_ok=True)
    row = {c: None for c in ev.RESULT_COLUMNS}
    row.update({"model": model, "seed": seed, "scheme": "user", "slice": "test",
                "ndcg@10": ndcg, "recall@10": recall, "coverage": coverage,
                "fit_seconds": 0, "eval_seconds": 0})
    (paths / "runs" / f"{model}_seed{seed}.json").write_text(json.dumps(row))
    (paths / "eval_runs").mkdir(exist_ok=True)
    pd.DataFrame({"userId": np.arange(len(per_user_ndcg)), "ndcg": per_user_ndcg,
                  "recall": per_user_recall}).to_parquet(
        paths / "eval_runs" / f"{model}_user_test_seed{seed}.parquet")


def status(paths):
    ft.write_test_csv()
    ft.settle()
    return pd.read_csv(paths / "status.csv").set_index("id").verdict.to_dict()


def test_settle_confirms_and_refutes_as_committed(paths):
    n = 400
    rng = np.random.default_rng(0)
    base = rng.random(n) * 0.2
    write_run(paths, "most_popular", 0, 0.05, 0.06, 0.005, base / 4, base / 4)
    write_run(paths, "item_knn", 0, 0.09, 0.12, 0.08, base, base)
    write_run(paths, "ease", 0, 0.12, 0.150, 0.06, base, base + 0.02)
    for s in (42, 43, 44):
        # two-tower: 2x popular, recall 5% below EASE (within 10%), not better than EASE
        write_run(paths, "two_tower", s, 0.10, 0.1425, 0.17, base, base + 0.0125)
        # two-stage: +3% over two-tower -> inside [-1%, +5%]
        write_run(paths, "two_stage", s, 0.103, 0.15, 0.12, base * 1.03, base)
    assert status(paths) == {1: "confirmed", 2: "confirmed", 3: "confirmed", 4: "confirmed"}


def test_settle_refutations(paths):
    n = 400
    base = np.random.default_rng(1).random(n) * 0.2
    write_run(paths, "most_popular", 0, 0.08, 0.06, 0.20, base / 4, base / 4)  # highest coverage
    write_run(paths, "ease", 0, 0.12, 0.15, 0.06, base, base)
    for s in (42, 43, 44):
        # two-tower: 1.25x popular (refutes #1); significantly better recall than EASE (refutes #2)
        write_run(paths, "two_tower", s, 0.10, 0.20, 0.17, base, base + 0.05)
        # two-stage: +20% (refutes #3)
        write_run(paths, "two_stage", s, 0.12, 0.25, 0.12, base * 1.2, base)
    assert status(paths) == {1: "refuted", 2: "refuted", 3: "refuted", 4: "refuted"}


def test_settle_waits_for_required_models(paths):
    write_run(paths, "ease", 0, 0.12, 0.15, 0.06, np.zeros(3), np.zeros(3))
    ft.write_test_csv()
    ft.settle()
    assert not (paths / "status.csv").exists()


def test_two_stage_test_path_end_to_end_on_fixture(spark, tmp_path, monkeypatch):
    """The whole shifted two-stage path (val labels -> test candidates) on the tiny fixture, so a
    bug can't surface only after real test results are sealed."""
    import torch
    from pathlib import Path
    from src import data_prep as dp
    from src import two_tower as tt
    fx = Path(__file__).resolve().parent / "fixtures"
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "ratings.csv").write_text((fx / "tiny_ratings.csv").read_text())
    (raw / "movies.csv").write_text((fx / "tiny_movies.csv").read_text())
    dp.run(spark, raw, tmp_path, tmp_path / "stats.csv", with_genome=False)
    items = ev.ItemIndex.from_movies_csv(fx / "tiny_movies.csv")
    rng = np.random.default_rng(0)
    pd.DataFrame({"movieId": items.movie_ids[:25], "genome": list(rng.random((25, 6)))}).to_parquet(
        tmp_path / "genome.parquet")
    splits = tmp_path / "splits.parquet"
    models = tmp_path / "models"
    cpu = torch.device("cpu")
    small = {"dim": 8, "hidden": 16, "batch": 16, "hist_len": 5}
    for training, name in ((("train",), "two_tower_seed1.pt"), (("train", "val"), "two_tower_tv_seed1.pt")):
        seq = tt.load_sequences(splits, items, training)
        sc, _ = tt.train_two_tower(seq, small, 1, fixed_epochs=1, log=lambda *_: None, dev=cpu)
        tt.save_scorer(sc, models / name, 1)
    res = tmp_path / "results"
    res.mkdir()
    pd.DataFrame([{"model": "ease", "config": json.dumps({"lam": 5.0, "min_pos": 1})}]).to_csv(res / "baselines.csv", index=False)
    pd.DataFrame([{"model": "two_stage", "config": json.dumps({"features": "headline", "learning_rate": 0.1, "min_data_in_leaf": 1, "num_leaves": 4, "rounds": 3})}]).to_csv(res / "two_stage.csv", index=False)
    monkeypatch.chdir(tmp_path)  # frozen() reads results/*.csv relative to the working dir
    for k, v in {"SPLITS": splits, "MOVIES": fx / "tiny_movies.csv", "FEATURES_DIR": tmp_path / "features",
                 "GENOME": tmp_path / "genome.parquet", "RUNS": tmp_path / "runs",
                 "PER_USER": tmp_path / "eval_runs", "WORK": tmp_path / "ranker_test",
                 "MODELS": models, "SEEDS": (1,), "RANKER_TRAIN_USERS": 3}.items():
        monkeypatch.setattr(ft, k, v)
    monkeypatch.setattr(tt, "device", lambda: cpu)
    fine = ev.load_ratings(splits, "user", fine=True)
    test = ev.build_eval_data(ev.load_ratings(splits, "user"), items, "user", "test", n_negatives=5)
    seq_tv = tt.load_sequences(splits, items, ("train", "val"))
    monkeypatch.setattr(ft.rk, "K_CANDIDATES", 15)
    import src.ranker as rkmod
    orig = rkmod.TrainingSetContext.build.__func__
    monkeypatch.setattr(rkmod.TrainingSetContext, "build",
                        classmethod(lambda cls, *a, **k: orig(cls, *a, **{**k, "k": 15})))
    ft.two_stage_test(items, fine, seq_tv, test, {})
    for m in ("two_stage", "two_stage_no_ease", "two_stage_with_time"):
        row = json.loads((tmp_path / "runs" / f"{m}_seed1.json").read_text())
        assert row["slice"] == "test" and row["n_users"] == test.n_users
        assert "recall_ceiling@200" in row
    meta = json.loads((tmp_path / "ranker_test" / "seed1" / "meta.json").read_text())
    assert meta["train_users"] == 3
    ft.two_stage_test(items, fine, seq_tv, test, {})  # second call: everything sealed -> no-op

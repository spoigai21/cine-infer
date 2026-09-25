"""Phase 8: the publish gate. A worse (or equal) candidate is rejected and logged, never
published; a better one swaps the live bundle atomically and updates the registry. All paths are
temporary: the real registry and bundles are never touched."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src import pipeline as pl

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def world(tmp_path, monkeypatch):
    models = tmp_path / "models"
    for name, path in {"MODELS": models, "REGISTRY": models / "registry.json",
                       "CANDIDATES": models / "candidates", "BUNDLES": models / "bundles",
                       "SERVING": models / "serving",
                       "PUBLISH_LOG": tmp_path / "results" / "publish_log.csv"}.items():
        monkeypatch.setattr(pl, name, path)
    monkeypatch.setattr(pl, "frozen_two_tower_config", lambda: ({"epochs": 6, "dim": 8}, 0.15))
    # an existing (Phase 7 style) bundle directory
    (models / "serving").mkdir(parents=True)
    (models / "serving" / "manifest.json").write_text("{}")
    return tmp_path


def candidate(run_id, ndcg, epochs):
    d = pl.run_dir(run_id)
    (d / "metrics.json").write_text(json.dumps({"ndcg@10": ndcg}))
    (d / "candidate.json").write_text(json.dumps({"config": {"epochs": epochs}}))


def test_init_registry_moves_bundle_behind_a_symlink(world):
    reg = pl.init_registry()
    assert reg["live"]["model_id"] == "phase7" and reg["live"]["val_ndcg@10"] == 0.15
    assert pl.SERVING.is_symlink() and (pl.SERVING / "manifest.json").exists()
    assert pl.init_registry() == reg  # idempotent


def test_worse_or_equal_candidate_is_rejected_and_logged(world):
    pl.init_registry()
    for run, score in (("worse", 0.12), ("equal", 0.15), ("within_noise", 0.15 + pl.MIN_GAIN / 2)):
        candidate(run, score, 1)
        assert pl.gate(run) == "reject"
        with pytest.raises(RuntimeError, match="refusing to publish"):
            pl.publish(run)
    log = pd.read_csv(pl.PUBLISH_LOG)
    assert list(log.decision) == ["reject"] * 3 and list(log.epochs) == [1] * 3
    assert pl.load_registry()["live"]["model_id"] == "phase7"   # live model untouched
    assert os.readlink(pl.SERVING) == "bundles/phase7"


def test_better_candidate_swaps_live_bundle_and_registry(world, monkeypatch):
    pl.init_registry()
    candidate("better", 0.15 + pl.MIN_GAIN + 0.001, 6)
    assert pl.gate("better") == "publish"
    built = []

    def fake_build(run_id, d):  # the real build refits models; the swap is what's tested here
        (pl.BUNDLES / run_id).mkdir(parents=True)
        (pl.BUNDLES / run_id / "manifest.json").write_text(json.dumps({"id": run_id}))
        built.append(run_id)

    monkeypatch.setattr(pl, "build_bundle", fake_build)
    pl.publish("better")
    assert built == ["better"]
    assert json.loads((pl.SERVING / "manifest.json").read_text()) == {"id": "better"}
    reg = pl.load_registry()
    assert reg["live"]["model_id"] == "better" and reg["live"]["val_ndcg@10"] > 0.15 + pl.MIN_GAIN
    assert reg["history"][-1]["model_id"] == "phase7"
    assert not list(pl.MODELS.glob(".serving.*.tmp"))


def test_dag_structure_in_airflow_env():
    airflow_py = ROOT / ".venv-airflow" / "bin" / "python"
    if not airflow_py.exists():
        pytest.skip("Airflow environment not installed (make airflow-install)")
    code = ("import logging, json; logging.disable(logging.WARNING)\n"
            "from airflow.models import DagBag\n"
            "b = DagBag(dag_folder='dags', include_examples=False)\n"
            "d = b.get_dag('cineinfer_retrain')\n"
            "print(json.dumps({'errors': {k: str(v) for k, v in b.import_errors.items()},"
            " 'edges': {t.task_id: sorted(t.downstream_task_ids) for t in d.tasks}}))")
    env = {**os.environ, "AIRFLOW_HOME": str(ROOT / "airflow_home"),
           "AIRFLOW__CORE__DAGS_FOLDER": str(ROOT / "dags"), "AIRFLOW__CORE__LOAD_EXAMPLES": "false"}
    r = subprocess.run([str(airflow_py), "-c", code], capture_output=True, text=True, cwd=ROOT, env=env)
    assert r.returncode == 0, r.stderr[-1500:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["errors"] == {}
    assert out["edges"] == {"prep": ["train"], "train": ["evaluate"], "evaluate": ["gate"],
                            "gate": ["publish", "reject"], "publish": [], "reject": []}


def test_seed_registry_from_candidate(world):
    candidate("weak", 0.10, 1)
    reg = pl.seed_registry(pl.CANDIDATES / "weak")
    assert reg["live"] == {**reg["live"], "model_id": "weak", "val_ndcg@10": 0.10, "bundle": None}
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        pl.seed_registry(pl.CANDIDATES / "weak")

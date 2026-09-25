"""Phase 8: retrain -> evaluate -> publish-if-better.

    prep -> train -> evaluate -> gate -+-> publish
                                       +-> reject

Every task runs a step of src/pipeline.py in the PROJECT's environment (.venv: PyTorch with the
Apple GPU, Spark, LightGBM). Airflow itself lives in .venv-airflow and only orchestrates.
`gate` compares the candidate's validation NDCG@10 with the live model's (models/registry.json;
$CINEINFER_MODELS_DIR relocates the registry, e.g. to a sandbox), requires a margin above noise,
logs the decision to results/publish_log.csv, and branches: a worse model never reaches publish.

Params: epochs (int, optional) overrides the tuned epoch count; epochs=1 gives the deliberately
worse candidate. Run: `make airflow-reject-demo` (airflow dags test, no scheduler needed).
"""
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator

REPO = Path(__file__).resolve().parents[1]
PY = f"{REPO}/.venv/bin/python -m src.pipeline"
RUN = "{{ run_id | replace(':', '_') | replace('+', '_') | replace('.', '_') }}"


def safe_run_id(run_id: str) -> str:
    """The same sanitising as src.pipeline.main (only [A-Za-z0-9_-] survive)."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in run_id)


def decide(run_id, **_):
    out = subprocess.run(f"{PY} gate --run-id {safe_run_id(run_id)}", shell=True, cwd=REPO,
                         capture_output=True, text=True)
    print(out.stdout, out.stderr)
    out.check_returncode()
    models = Path(os.environ.get("CINEINFER_MODELS_DIR", REPO / "models")).resolve()
    decision = json.loads((models / "candidates" / safe_run_id(run_id) /
                           "decision.json").read_text())
    print(f"gate decision: {decision['decision']} ({decision['reason']})")
    return "publish" if decision["decision"] == "publish" else "reject"


with DAG(
    dag_id="cineinfer_retrain",
    description="prep -> train -> evaluate -> publish only if better than the live model",
    start_date=datetime(2026, 9, 1),
    schedule=None,
    catchup=False,
    params={"epochs": Param(None, type=["null", "integer"], minimum=1,
                            description="override the tuned epoch count (1 = worse model)")},
    tags=["cineinfer"],
) as dag:
    prep = BashOperator(task_id="prep", bash_command=f"cd {REPO} && make prep", cwd=str(REPO))
    train = BashOperator(
        task_id="train", cwd=str(REPO),
        bash_command=f"{PY} train --run-id {RUN}"
                     "{% if params.epochs %} --epochs {{ params.epochs }}{% endif %}")
    evaluate = BashOperator(task_id="evaluate", cwd=str(REPO),
                            bash_command=f"{PY} evaluate --run-id {RUN}")
    gate = BranchPythonOperator(task_id="gate", python_callable=decide,
                                op_kwargs={"run_id": "{{ run_id }}"})
    publish = BashOperator(task_id="publish", cwd=str(REPO),
                           bash_command=f"{PY} publish --run-id {RUN}")
    reject = EmptyOperator(task_id="reject")
    prep >> train >> evaluate >> gate >> [publish, reject]

SHELL := /bin/bash
PYTHON ?= python3.11
VENV   := .venv
PY     := $(VENV)/bin/python

# Spark 3.5 supports Java 8/11/17, not 21. Prefer a native arm64 JDK 17, then x86_64 ones (Rosetta).
# A JAVA_HOME from your shell (e.g. SDKMAN's Java 21) is deliberately ignored; override with
# `make SPARK_JAVA_HOME=/path/to/jdk <target>`.
JAVA_CANDIDATES := /opt/homebrew/opt/openjdk@17 $(HOME)/.sdkman/candidates/java/17.0.5-tem \
                   /usr/local/opt/openjdk@17 /usr/local/opt/openjdk@11
SPARK_JAVA_HOME ?= $(firstword $(foreach d,$(JAVA_CANDIDATES),$(wildcard $(d))))
export JAVA_HOME := $(SPARK_JAVA_HOME)
export PATH := $(JAVA_HOME)/bin:$(PATH)

.PHONY: help install check-java data prep eval-check baselines two-tower ranker final-test export serve serving-parity load-test airflow-install airflow-reject-demo airflow-ui benchmark split-comparison readme readme-check analysis fixture test up down clean

help:
	@echo "make install     create .venv and install requirements"
	@echo "make check-java  confirm Spark can start with the selected JDK"
	@echo "make data        download + verify MovieLens 25M into data/ (never committed)"
	@echo "make prep        Phase 1: splits, burst stats, features -> data/, results/data_stats.csv"
	@echo "make eval-check  Phase 2: run the harness on real data with oracle + random models"
	@echo "make baselines   Phase 3: tune the 4 baselines on validation -> results/baselines.csv"
	@echo "make two-tower   Phase 5: tune the two-tower model on validation -> results/two_tower.csv"
	@echo "make ranker      Phase 6: two-stage (two-tower -> LightGBM) on validation -> results/ablation.csv"
	@echo "make final-test  Phase 6b: refit on train+val, score TEST once -> results/test.csv"
	@echo "make export      Phase 7: export the serving bundle -> models/serving/"
	@echo "make serve       Phase 7: run the API on :8000"
	@echo "make serving-parity  Phase 7: server vs batch pipeline (skew check) -> results/serving_parity.csv"
	@echo "make load-test   Phase 7: latency, prediction #6 -> results/latency.csv"
	@echo "make airflow-install   Phase 8: Airflow 2.10 in .venv-airflow (+ metadata DB)"
	@echo "make airflow-reject-demo  Phase 8: run the DAG with a 1-epoch (worse) model; it must be rejected"
	@echo "make airflow-ui   Phase 8: Airflow web UI on :8080 (airflow standalone)"
	@echo "make benchmark   Phase 9: pandas vs Spark vs Polars vs DuckDB (plug in, idle machine) -> results/benchmark*"
	@echo "make split-comparison  Phase 10: EASE on each split scheme's test slice (prediction #7)"
	@echo "make readme      Phase 10: regenerate README.md from results/*.csv"
	@echo "make readme-check  Phase 10: fail if README.md disagrees with results/*.csv"
	@echo "make analysis    validation metrics by train/val boundary type -> results/analysis/"
	@echo "make fixture     regenerate tests/fixtures/*.csv (synthetic, safe to commit)"
	@echo "make test        run pytest on the fixture"
	@echo "make up / down   start / stop the Docker Compose stack (Spark, API)"
	@echo "make clean       remove caches and derived data (keeps the downloaded zip)"

$(VENV)/.installed: requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	touch $@

install: $(VENV)/.installed

check-java: install
	@test -n "$(JAVA_HOME)" || { echo "No JDK 11/17 found. Install one: brew install openjdk@17"; exit 1; }
	@echo "JAVA_HOME=$(JAVA_HOME)"
	@java -version 2>&1 | head -1
	$(PY) -c "from pyspark.sql import SparkSession; \
	s = SparkSession.builder.master('local[1]').appName('check').getOrCreate(); \
	print('Spark', s.version, 'OK, rows =', s.range(10).count()); s.stop()"

data:
	bash scripts/get_data.sh

prep: install
	$(PY) -m src.data_prep

eval-check: install
	$(PY) -u -m scripts.check_harness

baselines: install
	$(PY) -u -m src.tune_baselines $(ARGS)

two-tower: install
	$(PY) -u -m src.tune_two_tower $(ARGS)

ranker: install
	$(PY) -u -m src.tune_ranker $(ARGS)

final-test: install
	$(PY) -u -m src.final_test $(ARGS)

export: install
	$(PY) -u -m src.export_serving $(ARGS)

serve: install
	$(VENV)/bin/uvicorn src.serve:app --port 8000

serving-parity: install
	$(PY) -m scripts.serving_parity --bundle models/serving --eval-dir data/ranker_test/seed42 \
		--pred data/ranker_test/seed42/pred_two_stage.npy --n 2000 --out results/serving_parity.csv

load-test: install
	$(PY) -m scripts.load_test

AIRFLOW_ENV := AIRFLOW_HOME=$(CURDIR)/airflow_home AIRFLOW__CORE__DAGS_FOLDER=$(CURDIR)/dags \
	AIRFLOW__CORE__LOAD_EXAMPLES=false
AIRFLOW_CONSTRAINTS := https://raw.githubusercontent.com/apache/airflow/constraints-2.10.5/constraints-3.11.txt

.venv-airflow/.installed:
	$(PYTHON) -m venv .venv-airflow
	.venv-airflow/bin/python -m pip install --upgrade pip
	.venv-airflow/bin/python -m pip install "apache-airflow==2.10.5" --constraint $(AIRFLOW_CONSTRAINTS)
	$(AIRFLOW_ENV) .venv-airflow/bin/airflow db migrate
	touch $@

airflow-install: .venv-airflow/.installed

airflow-reject-demo: install airflow-install
	$(PY) -m src.pipeline init-registry
	$(AIRFLOW_ENV) .venv-airflow/bin/airflow dags test cineinfer_retrain -c '{"epochs": 1}'

airflow-ui: airflow-install
	$(AIRFLOW_ENV) .venv-airflow/bin/airflow standalone

benchmark: install
	$(PY) -u -m scripts.benchmark

split-comparison: install
	$(PY) -u -m src.split_comparison

analysis: install
	$(PY) -m scripts.boundary_breakdown

fixture: install
	$(PY) scripts/make_fixture.py

test: install
	$(PY) -m pytest -q

up:
	docker compose up -d --build

down:
	docker compose down

clean:
	rm -rf .pytest_cache spark-warehouse metastore_db derby.log data/ml-25m data/splits_*
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

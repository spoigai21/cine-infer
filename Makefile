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

.PHONY: help install check-java data prep eval-check baselines two-tower ranker analysis fixture test up down clean

help:
	@echo "make install     create .venv and install requirements"
	@echo "make check-java  confirm Spark can start with the selected JDK"
	@echo "make data        download + verify MovieLens 25M into data/ (never committed)"
	@echo "make prep        Phase 1: splits, burst stats, features -> data/, results/data_stats.csv"
	@echo "make eval-check  Phase 2: run the harness on real data with oracle + random models"
	@echo "make baselines   Phase 3: tune the 4 baselines on validation -> results/baselines.csv"
	@echo "make two-tower   Phase 5: tune the two-tower model on validation -> results/two_tower.csv"
	@echo "make ranker      Phase 6: two-stage (two-tower -> LightGBM) on validation -> results/ablation.csv"
	@echo "make analysis    validation metrics by train/val boundary type -> results/analysis/"
	@echo "make fixture     regenerate tests/fixtures/*.csv (synthetic, safe to commit)"
	@echo "make test        run pytest on the fixture"
	@echo "make up / down   start / stop the Docker Compose stack (Spark, Airflow, API)"
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

analysis: install
	$(PY) -m scripts.boundary_breakdown

fixture: install
	$(PY) scripts/make_fixture.py

test: install
	$(PY) -m pytest -q

up:
	@test -f .env || { echo "Missing .env: cp .env.example .env and set a password"; exit 1; }
	docker compose up -d --build

down:
	docker compose down

clean:
	rm -rf .pytest_cache spark-warehouse metastore_db derby.log data/ml-25m data/splits_*
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

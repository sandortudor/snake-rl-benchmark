PYTHON ?= python3
RUN_DIR ?= runs/full_2000_tuned
REPORT_DIR ?= reports/full_2000_tuned

.DEFAULT_GOAL := help
.PHONY: help install test lint smoke demo evaluate report resources verify

help:
	@printf '%s\n' \
	  'Snake RL commands:' \
	  '  make install    Install the package and development tools' \
	  '  make test       Run the unit and production tests' \
	  '  make lint       Run Ruff static checks' \
	  '  make smoke      Run a small non-canonical training check' \
	  '  make demo       Play three games with the included compact checkpoint' \
	  '  make evaluate   Re-evaluate RUN_DIR into REPORT_DIR' \
	  '  make report     Regenerate aggregate plots for RUN_DIR' \
	  '  make resources  Measure local checkpoint size and inference latency' \
	  '  make verify     Run lint, tests, and the safe smoke experiment'

install:
	$(PYTHON) -m pip install -e '.[dev]'

test:
	$(PYTHON) -m unittest discover -s tests -v

lint:
	$(PYTHON) -m ruff check snake_rl tests

smoke:
	$(PYTHON) -m snake_rl.train --models random greedy q_learning --episodes 20 --seeds 1 --out runs/smoke
	$(PYTHON) -m snake_rl.plots --runs runs/smoke --reports reports/smoke

demo:
	$(PYTHON) -m snake_rl.visualize --model lightweight_snake_ddqn --train-seed 1 --games 3

evaluate:
	$(PYTHON) -m snake_rl.evaluate_benchmark --runs $(RUN_DIR) --episodes 100 --out $(REPORT_DIR)/evaluation_summary.csv

report:
	$(PYTHON) -m snake_rl.plots --runs $(RUN_DIR) --reports $(REPORT_DIR)

resources:
	$(PYTHON) -m snake_rl.benchmark_resources --runs $(RUN_DIR) --out $(REPORT_DIR)/deployment_metrics.csv

verify: lint test smoke

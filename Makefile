.PHONY: install dev test lint eval gate baseline calibrate serve report eval-local eval-models docker

install:            ## core + API
	pip install -e ".[api]"

dev:                ## everything needed to run CI locally
	pip install -e ".[api,dev]"

lint:
	ruff check askdocs tests

test:
	pytest

eval:               ## offline ablation suite (what CI runs) + update README table
	askdocs eval --config configs/offline.yaml --suite eval/suites/ablation.yaml --update-readme --readme-section CI

gate:               ## fail if results regress vs eval/baseline.json
	askdocs gate

baseline:           ## accept current results as the new baseline (commit the file)
	askdocs gate --update-baseline

calibrate:          ## sweep the abstention threshold
	askdocs calibrate --config configs/offline.yaml

eval-local:         ## Ollama + cross-encoder (needs `ollama serve` and `pip install -e .[ml]`)
	askdocs eval --config configs/local.yaml --suite eval/suites/ablation.yaml --out eval/results/local --update-readme --readme-section LOCAL

eval-models:        ## compare 3 local LLMs on the same retrieval
	askdocs eval --config configs/local.yaml --suite eval/suites/models.yaml --out eval/results/models --update-readme --readme-section MODELS

serve:
	askdocs serve --config configs/offline.yaml

report:
	askdocs report --html runs/dashboard.html

docker:
	docker compose up --build

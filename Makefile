.PHONY: install lint test db ingest eval run

install:
	pip install -e ".[dev]"

lint:
	ruff check . && ruff format --check .

test:
	pytest --cov=docs_qa

db:
	docker compose up -d --wait db

ingest:
	python -m docs_qa.ingest

# Ingests sample_docs + README, then prints hit@k / MRR for evals/questions.jsonl
eval:
	python -m docs_qa.eval --ingest

run:
	uvicorn docs_qa.api:create_app --factory --reload

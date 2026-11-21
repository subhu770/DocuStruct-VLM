.PHONY: install test benchmark lint run docker-build docker-run clean

install:
	pip install -e ".[dev]"

test:
	pytest -v tests/

benchmark:
	python benchmarks/eval_extraction.py

lint:
	ruff check .

run:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker build -t docustruct-vlm .

docker-run:
	docker-compose up --build

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +

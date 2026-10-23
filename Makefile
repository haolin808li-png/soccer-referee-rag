.PHONY: install test lint dry-run preflight
install:
	python -m pip install -e '.[dev]'
test:
	pytest -q
lint:
	ruff check src scripts tests
dry-run:
	python scripts/run_pipeline.py --config configs/pipeline.yaml --stages semantics index decisions prepare train evaluate --dry-run
preflight:
	python scripts/preflight.py --config configs/pipeline.yaml

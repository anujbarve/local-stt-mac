.PHONY: test train serve build clean lint typecheck docker-up docker-down

test:
	KMP_DUPLICATE_LIB_OK=TRUE python -m pytest brain/tests/ vault/tests/ tests/ -v

test-vault:
	KMP_DUPLICATE_LIB_OK=TRUE python -m pytest vault/tests/ -v

test-brain:
	KMP_DUPLICATE_LIB_OK=TRUE python -m pytest brain/tests/ -v

train:
	python scripts/train.py --quick

serve:
	python -m vault.api serve

build:
	docker build -t second-brain .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

lint:
	ruff check brain/ vault/

typecheck:
	mypy brain/ vault/ --ignore-missing-imports

clean:
	rm -rf __pycache__ brain/__pycache__ vault/__pycache__
	rm -rf .pytest_cache brain/.pytest_cache vault/.pytest_cache
	rm -rf vault_index/*
	find . -name "*.pyc" -delete

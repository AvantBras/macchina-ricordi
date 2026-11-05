.PHONY: install test lint format setup-musicgen musicgen-runtime-check musicgen-preflight

install:
	uv sync --all-extras

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

setup-musicgen:
	uv run python -m macchina_ricordi.cli runtime setup

musicgen-runtime-check:
	uv run python -m macchina_ricordi.cli runtime check

musicgen-preflight:
	uv run python -m macchina_ricordi.cli runtime check

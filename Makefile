.PHONY: install test lint format setup-musicgen musicgen-runtime-check musicgen-preflight

install:
	uv sync --all-extras

test:
	uv run pytest

# Correctness lint only (Ruff F + B); formatting is not enforced here.
lint:
	uv run ruff check .

# Optional formatter. Applies formatting only — no automatic bug fixes.
# Run `uv run ruff check --fix .` explicitly when you want autofixes.
format:
	uv run ruff format .

setup-musicgen:
	uv run python -m macchina_ricordi.cli runtime setup

musicgen-runtime-check:
	uv run python -m macchina_ricordi.cli runtime check

musicgen-preflight:
	uv run python -m macchina_ricordi.cli runtime check

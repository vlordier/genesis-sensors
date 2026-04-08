# Contributing

Contributions are welcome.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Checks

```bash
ruff check .
pytest tests -q
```

## Scope

This repo is intentionally small:
- reusable sensor rigs
- simple Genesis scene builders
- self-contained examples

Larger engine-level changes should stay upstream in `Genesis`.

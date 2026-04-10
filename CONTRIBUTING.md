# Contributing

Thanks for contributing to `genesis-sensors`.

## Repository flow

This repo follows a lightweight Git-flow style structure:

- `develop` — default integration branch for day-to-day work
- `main` — protected release branch
- short-lived topic branches created from `develop`

### Allowed branch prefixes

Use one of these prefixes for every working branch:

- `feat/...`
- `fix/...`
- `docs/...`
- `refactor/...`
- `chore/...`
- `test/...`
- `ci/...`
- `build/...`
- `release/...`

Examples:

```bash
git checkout develop
git pull
git checkout -b feat/add-barometer-demo
git checkout -b fix/pypi-metadata
```

## Pull request targets

- Open normal feature/fix/docs PRs **into `develop`**
- Open release PRs **from `develop` into `main`**

PR titles should follow this format:

```text
feat(scope): short summary
fix(scope): short summary
chore(scope): short summary
```

> Keep the squash-merge title in this format: release automation reads these Conventional Commit titles to decide the next version bump.

## Versioning

This project uses **Semantic Versioning**:

- `feat` → **minor** release bump
- `fix` → **patch** release bump
- breaking change → **major** release bump

To bump versions locally:

```bash
pip install -e .[dev]
bump-my-version patch   # or minor / major
```

Then review the diff, update `CHANGELOG.md`, and publish a `vX.Y.Z` tag.

See `RELEASE.md` for the full release checklist.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
# Required for scene-backed Genesis tests and demos
pip install torch --index-url https://download.pytorch.org/whl/cpu
pre-commit install
```

## Checks

```bash
pre-commit run --all-files
ruff check .
pytest tests -q
# Fast subset when Torch/Genesis runtime is not installed locally
pytest tests/test_compat.py tests/test_architecture.py tests/test_rigs.py -q
python -m build
python -m twine check dist/*
mkdocs build --strict
```

## Scope

This repo is intentionally small:
- reusable sensor rigs
- simple Genesis scene builders
- self-contained examples

Larger engine-level changes should stay upstream in `Genesis`.

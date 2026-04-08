# Release Guide

## Versioning policy

`genesis-sensors` uses **Semantic Versioning**:

- **MAJOR**: breaking API or behavior changes
- **MINOR**: backwards-compatible features
- **PATCH**: backwards-compatible fixes, packaging, docs, or CI updates

## Branch flow

1. Merge feature branches into `develop`
2. Stabilize and update changelog on `develop`
3. Open a PR from `develop` into `main`
4. After merge, create and push a `vX.Y.Z` tag from `main`

## Release checklist

```bash
# 1. sync and branch from the latest release code
git checkout main
git pull

# 2. bump the package version
pip install -e .[dev]
bump-my-version patch   # or minor / major

# 3. update release notes
$EDITOR CHANGELOG.md

# 4. verify artifacts
ruff check .
pytest tests -q
python -m build
python -m twine check dist/*

# 5. commit and tag
git add pyproject.toml src/genesis_sensors/__init__.py CHANGELOG.md
git commit -m "chore(release): cut vX.Y.Z"
git tag vX.Y.Z
git push origin main --tags
```

## Publishing

Tagged releases trigger `.github/workflows/publish.yml`.
Configure **PyPI Trusted Publishing** for:

- owner: `vlordier`
- repository: `genesis-sensors`
- workflow: `publish.yml`
- environment: `pypi`

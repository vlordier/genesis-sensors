# Release Guide

## Versioning policy

`genesis-sensors` uses **Semantic Versioning**:

- **MAJOR**: breaking API or behavior changes
- **MINOR**: backwards-compatible features
- **PATCH**: backwards-compatible fixes, packaging, docs, or CI updates

## Branch flow

1. Merge feature branches into `develop`
2. Stabilize and review on `develop`
3. Open a release PR from `develop` into `main`
4. Merge with a Conventional Commit title such as `feat(api): add ...` or `fix(sensor): ...`
5. `semantic-release` on `main` computes the next version, updates `CHANGELOG.md`, creates the Git tag, and publishes a GitHub release
6. The tagged `publish.yml` workflow pushes the package to PyPI

Release bumps are triggered by:
- `feat(...)` → minor release
- `fix(...)` or `perf(...)` → patch release
- `docs(...)`, `chore(...)`, `ci(...)`, `test(...)` → no package release by default

## Normal release flow

```bash
# 1. merge reviewed work into develop
# 2. open a release PR from develop -> main
# 3. use a Conventional Commit PR title
# 4. merge the PR
```

After the merge, the release is automated:

- `.github/workflows/semantic-release.yml` calculates the next SemVer bump
- `CHANGELOG.md` and the GitHub Release are updated automatically
- `.github/workflows/publish.yml` publishes the tagged build to PyPI

> To let semantic-release write back to protected `main`, add a repo secret named `RELEASE_PAT`
> with `contents:write` access (a classic PAT or fine-grained token for this repo). Without it,
> the workflow stays in validation mode and prints the next release version without publishing.

## Manual fallback

```bash
git checkout main
git pull
pip install -e .[dev]
bump-my-version patch   # or minor / major
$EDITOR CHANGELOG.md
python -m build
python -m twine check dist/*
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

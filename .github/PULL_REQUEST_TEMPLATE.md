## Summary

- what changed?
- why is it needed?

## Checklist

- [ ] Branch name follows the required prefix (`feat/`, `fix/`, `docs/`, `refactor/`, `chore/`, `test/`, `ci/`, `build/`, `release/`)
- [ ] PR title follows the convention (`feat(scope): summary`)
- [ ] Tests or validation steps were run
- [ ] Docs and changelog were updated when needed
- [ ] Target branch is `develop` (or `main` only for release PRs)

## Validation

```bash
ruff check .
pytest tests -q
python -m build
python -m twine check dist/*
```

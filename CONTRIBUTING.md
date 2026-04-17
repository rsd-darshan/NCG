# Contributing to NCG

Thanks for your interest in improving NCG.

## Development Setup

```bash
git clone https://github.com/rsd-darshan/NCG.git
cd NCG
pip install -e ".[dev]"
```

## Running Tests

Run the fast unit-test suite (same policy as CI):

```bash
pytest -q tests -m "not integration"
```

Run all tests including dataset-dependent integration tests:

```bash
pytest -q tests
```

## CI Behavior

GitHub Actions runs tests on Python 3.9 and 3.11.
Tests marked `integration` are excluded in CI because they require dataset downloads.

## Reproducibility Guidance

- Always report the seed list you used.
- Prefer multiple-seed summaries rather than single-seed claims.
- Keep final tables in `results*/results_table.csv`.

## Code Style

- Keep modules small and focused.
- Add tests for behavior changes.
- Avoid committing local environment files (`.venv*`, `.DS_Store`) or checkpoints.

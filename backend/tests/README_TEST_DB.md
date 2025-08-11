# Test Database Strategy

## Overview
Tests use completely isolated, temporary SQLite databases that are automatically created and destroyed for each test run.

## Key Principles

### 1. **Complete Isolation**
- Test databases are created in the system temp directory (`/tmp` or equivalent)
- Database files use unique names with pattern: `gene_disease_*_test.db`
- Each test function gets a fresh database

### 2. **No Production Database Access**
- Tests never touch the production database at `/gits/database/gene_disease.db`
- Database engine is lazily initialized to prevent accidental creation
- Test fixtures override the database dependency injection

### 3. **Automatic Cleanup**
- Databases are deleted after each test completes
- `pytest_sessionfinish` hook ensures no leftover test databases
- Force cleanup even if tests fail or are interrupted

## Implementation Details

### Test Database Fixtures (`conftest.py`)

```python
@pytest.fixture(scope="function")
def test_engine():
    """Create isolated test database in temp directory."""
    db_fd, db_path = tempfile.mkstemp(suffix='_test.db', prefix='gene_disease_')
    # ... create engine and tables
    yield engine
    # ... cleanup
```

### Lazy Database Initialization (`main.py`)

```python
def get_engine():
    """Lazy initialization prevents database creation at import time."""
    global _engine
    if _engine is None:
        _engine = get_db_engine()
    return _engine
```

### Database Cleanup Plugin (`test_db_cleanup.py`)

- `pytest_sessionstart`: Cleans up any leftover test databases
- `pytest_sessionfinish`: Verifies all test databases were removed
- `ensure_no_production_db_access`: Prevents accidental production DB access

## Running Tests

```bash
# Run all tests
pytest

# Run integration tests only
pytest -m integration

# Run with verbose database operations (debugging)
pytest -vv --capture=no
```

## Troubleshooting

### Check for leftover test databases
```bash
ls /tmp/gene_disease_*_test.db
```

### Clean up manually if needed
```bash
rm -f /tmp/gene_disease_*_test.db
```

## Benefits

1. **Safety**: Production data is never at risk
2. **Speed**: Each test starts with a clean slate
3. **Reproducibility**: Tests always run in identical conditions
4. **Debugging**: Temp database files can be inspected if needed
5. **CI/CD Ready**: Works in any environment without setup
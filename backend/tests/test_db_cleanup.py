"""
Pytest plugin to ensure test database cleanup.
This module provides hooks to verify that test databases are properly cleaned up.
"""

import os
import glob
import tempfile
import pytest


def pytest_sessionstart(session):
    """Called at the start of the test session."""
    # Record any existing test databases (shouldn't be any)
    temp_dir = tempfile.gettempdir()
    existing_test_dbs = glob.glob(os.path.join(temp_dir, "gene_disease_*_test.db"))
    
    if existing_test_dbs:
        print(f"\nWarning: Found {len(existing_test_dbs)} leftover test database(s)")
        for db in existing_test_dbs:
            try:
                os.unlink(db)
                print(f"  Cleaned up: {db}")
            except OSError:
                print(f"  Could not remove: {db}")


def pytest_sessionfinish(session, exitstatus):
    """Called at the end of the test session."""
    # Verify all test databases were cleaned up
    temp_dir = tempfile.gettempdir()
    remaining_test_dbs = glob.glob(os.path.join(temp_dir, "gene_disease_*_test.db"))
    
    if remaining_test_dbs:
        print(f"\nWarning: {len(remaining_test_dbs)} test database(s) not cleaned up:")
        for db in remaining_test_dbs:
            print(f"  {db}")
            # Try to clean up
            try:
                os.unlink(db)
                print(f"    -> Cleaned up")
            except OSError as e:
                print(f"    -> Could not remove: {e}")


@pytest.fixture(autouse=True)
def ensure_no_production_db_access():
    """Fixture to ensure tests don't accidentally use production database."""
    # This runs for every test automatically
    production_db_paths = [
        "/Users/michaelnedzelsky/Dropbox/PARA/1_Projects/genestack_test/gits/database/gene_disease.db",
        "/Users/michaelnedzelsky/Dropbox/PARA/1_Projects/genestack_test/gits/backend/database/gene_disease.db"
    ]
    
    # Store original file access function
    original_open = open
    original_exists = os.path.exists
    
    def protected_open(file, *args, **kwargs):
        """Prevent opening production database files during tests."""
        if any(db_path in str(file) for db_path in production_db_paths):
            if "test" not in str(file):  # Allow test databases
                raise RuntimeError(f"Test attempted to access production database: {file}")
        return original_open(file, *args, **kwargs)
    
    def protected_exists(path):
        """Log when checking for production database."""
        if any(db_path in str(path) for db_path in production_db_paths):
            if "test" not in str(path):
                print(f"Warning: Test checking for production database: {path}")
        return original_exists(path)
    
    # Temporarily replace file operations during test
    import builtins
    builtins.open = protected_open
    os.path.exists = protected_exists
    
    yield
    
    # Restore original functions
    builtins.open = original_open
    os.path.exists = original_exists
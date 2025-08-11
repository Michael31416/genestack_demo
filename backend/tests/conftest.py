"""
Test configuration and fixtures for the Gene-Disease Analysis Service.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient
import tempfile
import os

from app.models import Base, get_db_engine, get_session_maker
from app.main import app


# Removed deprecated event_loop fixture - using pytest-asyncio default




@pytest.fixture(scope="function")
def test_engine():
    """Create a test database engine with complete isolation."""
    import tempfile
    import os
    
    # Create a temporary database file in system temp directory
    # This ensures it's completely isolated from the project
    db_fd, db_path = tempfile.mkstemp(suffix='_test.db', prefix='gene_disease_')
    os.close(db_fd)
    
    try:
        # Create engine with the temporary file
        # Use stricter settings for testing
        engine = create_engine(
            f"sqlite:///{db_path}", 
            connect_args={
                "check_same_thread": False,
                "timeout": 5.0  # Shorter timeout for tests
            },
            echo=False  # Set to True for debugging
        )
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        
        yield engine
    
    finally:
        # Ensure complete cleanup
        engine.dispose()
        # Force close all connections
        engine.pool.dispose()
        try:
            os.unlink(db_path)
        except (OSError, PermissionError) as e:
            # Log but don't fail if cleanup has issues
            print(f"Warning: Could not delete test database {db_path}: {e}")


@pytest.fixture
def test_session_factory(test_engine):
    """Create a session factory for the test database."""
    return sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture
def db_session(test_session_factory):
    """Create a database session for integration tests with automatic rollback."""
    session = test_session_factory()
    # Begin a transaction that will be rolled back
    session.begin()
    try:
        yield session
    finally:
        # Rollback any changes made during the test
        session.rollback()
        session.close()


@pytest.fixture
def client(test_session_factory):
    """Create a test client that uses the same database as db_session fixture."""
    def override_get_db():
        db = test_session_factory()
        try:
            yield db
        finally:
            db.close()
    
    # Import app and override the dependency
    from app.main import get_db
    app.dependency_overrides[get_db] = override_get_db
    
    try:
        # Create test client
        with TestClient(app) as test_client:
            yield test_client
    finally:
        # Clean up
        app.dependency_overrides.clear()


@pytest.fixture
def mock_httpx_client():
    """Mock httpx.AsyncClient for testing external API calls."""
    mock_client = AsyncMock()
    return mock_client


@pytest.fixture
def sample_gene_data():
    """Sample gene data for testing."""
    return {
        "id": "ENSG00000141510",
        "display_name": "TP53",
        "description": "tumor protein p53"
    }


@pytest.fixture
def sample_disease_data():
    """Sample disease data for testing."""
    return {
        "response": {
            "docs": [
                {
                    "ontology_name": "efo",
                    "short_form": "0000616",
                    "label": "lung carcinoma",
                    "synonym": ["lung cancer", "pulmonary carcinoma"]
                }
            ]
        }
    }


@pytest.fixture
def sample_opentargets_data():
    """Sample Open Targets data for testing."""
    return {
        "data": {
            "disease": {
                "id": "EFO_0000616",
                "name": "lung carcinoma",
                "associatedTargets": {
                    "count": 1,
                    "rows": [
                        {
                            "score": 0.85,
                            "target": {
                                "id": "ENSG00000141510",
                                "approvedSymbol": "TP53"
                            },
                            "datatypeScores": [
                                {"id": "genetic_association", "score": 0.9},
                                {"id": "somatic_mutation", "score": 0.8}
                            ]
                        }
                    ]
                }
            }
        }
    }


@pytest.fixture
def sample_literature_data():
    """Sample literature data for testing."""
    return {
        "resultList": {
            "result": [
                {
                    "pmid": "12345678",
                    "title": "TP53 mutations in lung cancer",
                    "pubYear": "2023",
                    "source": "PubMed",
                    "authorString": "Smith J, Doe A",
                    "abstractText": "TP53 is frequently mutated in lung cancer. The tumor suppressor gene TP53 plays a crucial role in lung cancer development."
                }
            ]
        }
    }


@pytest.fixture
def sample_gwas_data():
    """Sample GWAS data for testing."""
    return {
        "_embedded": {
            "associations": [
                {
                    "associationId": "12345",
                    "pvalueMantissa": 5.2,
                    "pvalueExponent": -8,
                    "orPerCopyNum": 1.5,
                    "ci": "1.2-1.8",
                    "pubmedId": "87654321",
                    "trait": "lung carcinoma",
                    "studyAccession": "GCST123456",
                    "loci": [
                        {
                            "strongestRiskAlleles": [
                                {
                                    "ensemblGenes": [
                                        {"geneName": "TP53"}
                                    ]
                                }
                            ],
                            "authorReportedGenes": ["TP53"]
                        }
                    ]
                }
            ]
        }
    }


@pytest.fixture
def sample_openai_response():
    """Sample OpenAI API response for testing."""
    return {
        "choices": [
            {
                "message": {
                    "content": '{"verdict": "strong", "confidence": 0.85, "drivers": {"genetic": {"present": true, "summary": "Strong genetic evidence", "source_ids": ["gwas_catalog:PMID12345"]}}, "key_points": [{"statement": "TP53 is frequently mutated in lung cancer", "source_ids": ["literature:PMID12345678"]}], "conflicts_or_gaps": [], "recommended_next_steps": ["Validate findings in clinical trials"]}'
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150
        }
    }


@pytest.fixture
def sample_anthropic_response():
    """Sample Anthropic API response for testing."""
    return {
        "content": [
            {
                "text": '{"verdict": "moderate", "confidence": 0.75, "drivers": {"functional": {"present": true, "summary": "Functional evidence available", "source_ids": ["opentargets:association"]}}, "key_points": [{"statement": "Moderate evidence for association", "source_ids": ["literature:PMID12345678"]}], "conflicts_or_gaps": [], "recommended_next_steps": ["More studies needed"]}'
            }
        ],
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50
        }
    }


@pytest.fixture
def mock_analysis_request():
    """Sample analysis request for testing."""
    return {
        "gene": "TP53",
        "disease": "lung cancer",
        "since_year": 2020,
        "max_abstracts": 5,
        "include_gwas": True,
        "model": "gpt-5-mini"
    }


@pytest.fixture
def mock_login_request():
    """Sample login request for testing."""
    return {
        "username": "testuser",
        "api_provider": "openai",
        "api_key": "sk-test-key-123"
    }
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Gene-Disease Analysis Service that analyzes potential correlations between genes and diseases using public life sciences data and LLM capabilities. The project consists of a FastAPI backend with a web frontend and includes both a CLI tool and web service implementation.

## Project Requirements

### Core Components
1. **Frontend Interface**: Web interface for user input (username, API key, gene/disease names) and displaying results
2. **Backend API**: RESTful API for handling sessions, processing requests, and managing history
3. **Data Integration**: Integration with public life sciences APIs (e.g., OpenTargets, Ensembl, Europe PMC, GWAS Catalog)
4. **LLM Integration**: Support for OpenAI and/or Anthropic APIs for correlation analysis
5. **Database**: SQLite for storing analysis history
6. **Docker**: Complete docker-compose setup for running the application

### Key Technical Requirements
- Handle multiple concurrent requests efficiently using async/await patterns
- No authentication required for data source APIs
- Proper error handling and rate limiting for LLM APIs
- Single command deployment via `docker-compose up`

## Development Commands

```bash
# Run the application
docker-compose up

# Run in development mode
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Use the CLI tool directly (requires uv)
cd genestack_test
uv run --with typer --with httpx --with tenacity --with pydantic \
  gene_disease_cli.py analyze --gene IL22 --disease psoriasis --since 2015 --max-abstracts 8 --model gpt-4o-mini

# View CLI history and export results
uv run --with typer --with httpx --with tenacity --with pydantic gene_disease_cli.py history
uv run --with typer --with httpx --with tenacity --with pydantic gene_disease_cli.py show 1
uv run --with typer --with httpx --with tenacity --with pydantic gene_disease_cli.py export 1 --out run1.json
```

## Architecture

### Backend Structure (FastAPI)
- **app/main.py**: Main FastAPI application with endpoints, WebSocket support, and rate limiting
- **app/models/**: SQLAlchemy models for Users, Sessions, Analysis, and Results
- **app/schemas/**: Pydantic schemas for request/response validation
- **app/services/**: Business logic layer
  - **analysis_service.py**: Orchestrates the complete analysis pipeline
  - **data_fetcher.py**: Handles all external API integrations with retry logic
  - **llm_service.py**: LLM integration for correlation analysis
- **static/**: Web frontend (HTML/CSS/JS)

### Database Schema
- **users**: User management with username
- **sessions**: User sessions with encrypted API keys and provider info
- **analyses**: Analysis requests with gene/disease info and status tracking
- **results**: Analysis results with evidence, LLM output, and verdicts

### API Design
- RESTful endpoints under `/api/v1/`
- Session-based authentication using UUIDs
- WebSocket support for real-time analysis updates
- Rate limiting (10/min for login, 20/min for analyses)

## Data Source Integration

### External APIs (No Authentication Required)
- **Ensembl REST**: Gene symbol → Ensembl ID + synonyms
- **EBI OLS4**: Disease name → EFO/MONDO ID + synonyms  
- **Open Targets Platform GraphQL**: Target-disease association scores and evidence types
- **Europe PMC REST**: Literature search for gene-disease co-mentions
- **GWAS Catalog REST**: Genetic associations filtered by gene and trait

### Data Processing Pipeline
1. **Resolution Phase**: Convert gene symbols and disease names to standardized IDs
2. **Evidence Collection**: Fetch data from all sources concurrently using asyncio
3. **LLM Analysis**: Optional correlation analysis with verdict and confidence scoring
4. **Storage**: Persist evidence and results in SQLite database

## LLM Integration Best Practices

- Implement retry logic with exponential backoff using tenacity
- Store API keys securely (currently stored encrypted in sessions)
- Design prompts that specifically ask for correlation analysis with scoring
- Support both OpenAI and Anthropic providers
- Handle model compatibility issues (older models without structured output)

## Important Implementation Notes

1. **Concurrency**: Uses async/await patterns throughout for handling multiple requests
2. **Error Handling**: Graceful degradation when external APIs are unavailable
3. **Session Management**: UUID-based sessions with API key storage
4. **WebSocket Updates**: Real-time progress updates during analysis
5. **Rate Limiting**: Implemented using SlowAPI middleware
6. **Docker Configuration**: Single-service setup with volume mounting for database

## CLI Tool (genestack_test/gene_disease_cli.py)

A standalone CLI version that provides the same functionality with local SQLite storage in `~/.gene_disease_cli/history.db`. Useful for batch processing or direct command-line usage.

## Notable Public Life Sciences APIs Used

### Primary Data Sources
- **Open Targets Platform (GraphQL)**: Unified target-disease association scores with typed evidence
- **Ensembl REST**: Gene symbol resolution and normalization
- **EBI OLS4**: Disease name to ontology ID mapping (EFO/MONDO)
- **Europe PMC**: Literature search with date filtering and abstract extraction
- **GWAS Catalog**: SNP-trait associations with p-values and mapped genes

### Data Quality Features
- Automatic synonym resolution for both genes and diseases
- Evidence sentence extraction from abstracts
- Structured data types with confidence scoring
- Comprehensive error handling for API failures

## Development Guidelines

- Follow async/await patterns for all external API calls
- Use tenacity for retry logic on network requests
- Implement proper error handling with graceful degradation
- Store sensitive data (API keys) securely
- Follow SQLAlchemy best practices for database operations
- Use Pydantic models for all request/response validation
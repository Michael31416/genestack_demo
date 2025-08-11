# Gene-Disease Analysis Service

A web application that analyzes potential correlations between genes and diseases using public life sciences data and LLM capabilities.

## Features

- **Multi-source Data Integration**: Fetches data from Open Targets, Europe PMC, GWAS Catalog, Ensembl, and EBI OLS
- **LLM Analysis**: Supports both OpenAI and Anthropic APIs for intelligent correlation analysis
- **Real-time Updates**: WebSocket-based live progress updates during analysis
- **User Sessions**: Simple session management with API key storage
- **Analysis History**: Track and review all previous analyses
- **Concurrent Processing**: Handles multiple analysis requests efficiently
- **Docker Deployment**: Single-command deployment with Docker Compose

## Quick Start

### Prerequisites

- Docker and Docker Compose installed
- An API key from either OpenAI or Anthropic

### Running the Application

1. Clone the repository:
```bash
git clone <repository-url>
cd genestack_test
```

2. Start the application:
```bash
docker-compose up
```

3. Open your browser and navigate to:
```
http://localhost:8000
```

4. Login with:
   - Your chosen username
   - Select your API provider (OpenAI or Anthropic)
   - Enter your API key

5. Start analyzing gene-disease correlations!

## Usage

### Login
1. Enter any username (this creates a session for you)
2. Select your LLM provider (OpenAI or Anthropic)
3. Enter your API key (stored securely for the session)

### Running an Analysis
1. Enter a gene symbol (e.g., "TP53", "BRCA1", "IL6")
2. Enter a disease name (e.g., "lung cancer", "psoriasis", "diabetes")
3. Configure optional parameters:
   - Literature date range (default: since 2015)
   - Maximum abstracts to analyze (default: 8)
   - Include GWAS data (default: enabled)
   - Select LLM model based on your provider
4. Click "Analyze" and watch real-time progress updates

### Viewing Results
- **Verdict**: Strong, Moderate, Weak, No Evidence, or Inconclusive
- **Confidence Score**: 0-100% confidence in the correlation
- **Key Findings**: Important evidence points with source citations
- **Evidence Summary**: Data from Open Targets, literature, and GWAS
- **Recommended Next Steps**: Suggested follow-up investigations

### History
- View all previous analyses
- Click on any history item to view full results
- Filter by gene, disease, or verdict

## Architecture

### Technology Stack

**Backend:**
- FastAPI (Python web framework)
- SQLAlchemy (ORM)
- SQLite (Database)
- httpx (Async HTTP client)
- Tenacity (Retry logic)

**Frontend:**
- Vanilla JavaScript (ES6+)
- HTML5 & CSS3
- WebSocket API for real-time updates
- LocalStorage for session persistence

**Data Sources:**
- Open Targets Platform (GraphQL API)
- Europe PMC (REST API)
- GWAS Catalog (REST API)
- Ensembl (REST API)
- EBI OLS4 (REST API)

### Design Decisions

1. **FastAPI over Flask/Django**: Superior async support for concurrent requests, built-in API documentation, and modern Python features

2. **Vanilla JavaScript over React/Vue**: Simplifies deployment, eliminates build steps, demonstrates core web development skills

3. **SQLite over PostgreSQL/MySQL**: Perfect for this use case - file-based, zero configuration, sufficient performance

4. **WebSockets for Real-time Updates**: Better user experience than polling, shows analysis progress as it happens

5. **Session-based Authentication**: Simple and appropriate for this use case, no complex auth overhead

6. **Monolithic Docker Container**: Simplifies deployment, FastAPI serves both API and static files

### API Endpoints

- `POST /api/v1/auth/login` - Create user session
- `POST /api/v1/analyses` - Start new analysis
- `GET /api/v1/analyses/{id}` - Get analysis results
- `GET /api/v1/analyses` - List user's history
- `WS /api/v1/ws/{analysis_id}` - WebSocket for live updates

## Development

### Local Development Setup

1. Create a Python virtual environment:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
uvicorn app.main:app --reload
```

4. Access at http://localhost:8000

### Running Tests

The application includes a comprehensive test suite with unit and integration tests:

```bash
cd backend

# Install test dependencies (if not already installed)
pip install -r requirements.txt

# Run all tests
pytest

# Run with coverage report
pytest --cov=app --cov-report=html

# Run only unit tests
pytest tests/unit/

# Run only integration tests  
pytest tests/integration/

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/unit/test_llm_service.py
```

**Test Coverage:**
- **Unit Tests**: 102 tests covering all service modules with mocked external APIs
  - `test_data_fetcher.py`: Tests for all external API integrations (Ensembl, OLS, OpenTargets, Europe PMC, GWAS)
  - `test_llm_service.py`: Tests for OpenAI and Anthropic LLM integrations with response format handling
  - `test_analysis_service.py`: Tests for the analysis orchestration layer
- **Integration Tests**: 26 tests covering FastAPI endpoints with test database
  - Authentication, analysis creation, result retrieval, WebSocket connections
- **Mock Strategy**: All external APIs are mocked to ensure fast, reliable tests without dependencies

### Project Structure

```
genestack_test/
├── backend/
│   ├── app/
│   │   ├── main.py           # FastAPI application
│   │   ├── api/              # API endpoints
│   │   ├── services/         # Business logic
│   │   │   ├── data_fetcher.py    # External API integration
│   │   │   ├── llm_service.py     # LLM integration
│   │   │   └── analysis_service.py # Analysis orchestration
│   │   ├── models/           # Database models
│   │   └── schemas/          # Pydantic schemas
│   ├── static/               # Frontend files
│   │   ├── index.html
│   │   ├── style.css
│   │   └── app.js
│   ├── tests/                # Test suite
│   │   ├── unit/             # Unit tests with mocked dependencies
│   │   ├── integration/      # Integration tests with test database
│   │   └── conftest.py       # Test fixtures and configuration
│   ├── requirements.txt
│   ├── pytest.ini           # Pytest configuration
│   └── Dockerfile
├── database/                 # SQLite database (created at runtime)
├── docker-compose.yml
└── README.md
```

## Configuration

### Environment Variables

You can set these environment variables to customize the application:

- `OPENAI_BASE_URL` - Custom OpenAI API endpoint (default: https://api.openai.com/v1)
- `ANTHROPIC_BASE_URL` - Custom Anthropic API endpoint (default: https://api.anthropic.com/v1)

### Database

The SQLite database is automatically created in the `database/` directory on first run. It persists across container restarts via Docker volumes.

## Trade-offs and Limitations

1. **Security**: API keys are stored in plain text in the database (should be encrypted in production)
2. **Scalability**: SQLite limits concurrent writes (consider PostgreSQL for high load)
3. **Rate Limiting**: Basic implementation, could be enhanced with Redis
4. **Error Recovery**: Limited retry logic for external API failures
5. **Testing**: Comprehensive test suite included with mocked external dependencies
6. **Monitoring**: No logging or metrics (would add in production)

## Future Enhancements

- Add API key encryption
- Improve test coverage and add end-to-end tests
- Add export functionality (PDF, CSV)
- Enhanced caching for external API responses
- User authentication and authorization
- Advanced search and filtering
- Batch analysis capabilities
- Integration with more data sources
- Visualization of gene-disease networks

## Troubleshooting

### Common Issues

1. **Port already in use**: Change the port in docker-compose.yml
2. **API key errors**: Verify your API key is valid and has sufficient credits
3. **Slow analysis**: Some genes/diseases have extensive data; be patient
4. **WebSocket connection failed**: Check if your browser supports WebSockets

### Logs

View application logs:
```bash
docker-compose logs -f
```

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Open Targets Platform for comprehensive gene-disease associations
- Europe PMC for literature access
- GWAS Catalog for genetic associations
- Ensembl and EBI for biological data services

## GPT-5 Model Support

This application now supports the latest GPT-5 model family:
- gpt-5-mini (default, optimized for speed and cost)
- gpt-5 (full capability model)  
- gpt-5-nano (ultra-fast model)

All models have been tested and verified to work with the gene-disease analysis pipeline.


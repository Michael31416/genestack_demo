"""
Configuration constants for the Gene-Disease Analysis Service.

This module centralizes all magic constants used throughout the application,
making them easy to modify and maintain.
"""

from typing import Dict, Any
import os

# =============================================================================
# SESSION & SECURITY CONFIGURATION
# =============================================================================

SESSION_TIMEOUT_MINUTES = int(os.getenv("SESSION_TIMEOUT_MINUTES", "20"))
"""Session timeout in minutes based on inactivity"""

SESSION_CLEANUP_INTERVAL_SECONDS = int(os.getenv("SESSION_CLEANUP_INTERVAL_SECONDS", "3600"))
"""How often to clean up expired sessions (1 hour)"""

SECONDS_PER_MINUTE = 60
"""Conversion factor from minutes to seconds"""

# =============================================================================
# RATE LIMITING CONFIGURATION  
# =============================================================================

# API endpoint rate limits (requests per minute)
LOGIN_RATE_LIMIT = "10/minute"
"""Rate limit for login/logout endpoints"""

ANALYSIS_RATE_LIMIT = "20/minute" 
"""Rate limit for analysis creation endpoint"""

# Internal LLM rate limiting
LLM_REQUESTS_PER_MINUTE = 20
"""Internal rate limiting for LLM API calls"""

LLM_RATE_WINDOW_SECONDS = 60
"""Rate limiting window duration"""

LLM_RATE_BUFFER_SECONDS = 61
"""Buffer time to wait when rate limit exceeded"""

# =============================================================================
# HTTP TIMEOUT CONFIGURATION
# =============================================================================

HTTP_TIMEOUT_SHORT = 30
"""Timeout for fast API calls (seconds)"""

HTTP_TIMEOUT_MEDIUM = 45
"""Timeout for medium API calls (seconds)"""

HTTP_TIMEOUT_LONG = 90
"""Timeout for slow LLM API calls (seconds)"""

# =============================================================================
# WEBSOCKET CONFIGURATION
# =============================================================================

WEBSOCKET_KEEPALIVE_SECONDS = 1
"""WebSocket keep-alive ping interval"""

# =============================================================================
# DATA FETCHING CONFIGURATION
# =============================================================================

# Default pagination limits
DEFAULT_ENSEMBL_ROWS = 25
"""Default number of rows for Ensembl API queries"""

DEFAULT_OPENTARGETS_SIZE = 50  
"""Default page size for OpenTargets API queries"""

DEFAULT_LITERATURE_PAGE_SIZE = 25
"""Default page size for literature searches"""

# Maximum limits to prevent abuse
MAX_GWAS_RECORDS = 1000
"""Maximum GWAS records to fetch"""

# Year range for literature searches
LITERATURE_FUTURE_YEAR = 3000
"""Future year upper bound for literature searches"""

# =============================================================================
# LLM CONFIGURATION
# =============================================================================

# Token limits
OPENAI_MAX_COMPLETION_TOKENS = 4000
"""Maximum completion tokens for OpenAI API"""

ANTHROPIC_MAX_TOKENS = 2000
"""Maximum tokens for Anthropic API"""

# API versions
ANTHROPIC_API_VERSION = "2023-06-01"
"""Anthropic API version header"""

# Retry configuration
RETRY_MIN_WAIT_SECONDS = 2
"""Minimum wait time for exponential backoff"""

RETRY_MAX_WAIT_SECONDS = 30
"""Maximum wait time for exponential backoff"""

RETRY_MULTIPLIER = 1
"""Exponential backoff multiplier"""

# =============================================================================
# VALIDATION CONFIGURATION
# =============================================================================

# String length limits
USERNAME_MAX_LENGTH = 100
"""Maximum username length"""

GENE_MAX_LENGTH = 50
"""Maximum gene symbol length"""

DISEASE_MAX_LENGTH = 200
"""Maximum disease name length"""

# Year validation
MIN_PUBLICATION_YEAR = 1990
"""Minimum valid publication year"""

MAX_PUBLICATION_YEAR = 2025
"""Maximum valid publication year"""

DEFAULT_SINCE_YEAR = 2015
"""Default starting year for searches"""

# Abstract limits
DEFAULT_MAX_ABSTRACTS = 8
"""Default maximum abstracts to fetch"""

MIN_ABSTRACTS = 1
"""Minimum abstracts that can be requested"""

MAX_ABSTRACTS = 25
"""Maximum abstracts that can be requested"""

# History pagination
DEFAULT_HISTORY_LIMIT = 50
"""Default maximum history items to return"""

# =============================================================================
# HTTP STATUS CODES
# =============================================================================

HTTP_STATUS_UNAUTHORIZED = 401
HTTP_STATUS_NOT_FOUND = 404  
HTTP_STATUS_TOO_MANY_REQUESTS = 429
HTTP_STATUS_BAD_REQUEST = 400
HTTP_STATUS_SERVER_ERROR = 500

# =============================================================================
# APPLICATION CONFIGURATION
# =============================================================================

APP_VERSION = "1.0.0"
"""Application version"""

DEFAULT_LLM_MODEL = "gpt-4o-mini"
"""Default LLM model for analysis"""

# =============================================================================
# ENVIRONMENT-SPECIFIC OVERRIDES
# =============================================================================

def get_config() -> Dict[str, Any]:
    """
    Get configuration dictionary with environment variable overrides.
    
    Returns:
        Dictionary of all configuration values
    """
    return {
        # Sessions
        "session_timeout_minutes": SESSION_TIMEOUT_MINUTES,
        "session_cleanup_interval": SESSION_CLEANUP_INTERVAL_SECONDS,
        
        # Rate limits  
        "login_rate_limit": LOGIN_RATE_LIMIT,
        "analysis_rate_limit": ANALYSIS_RATE_LIMIT,
        "llm_requests_per_minute": LLM_REQUESTS_PER_MINUTE,
        
        # Timeouts
        "http_timeout_short": HTTP_TIMEOUT_SHORT,
        "http_timeout_medium": HTTP_TIMEOUT_MEDIUM, 
        "http_timeout_long": HTTP_TIMEOUT_LONG,
        
        # Data limits
        "max_abstracts": MAX_ABSTRACTS,
        "default_history_limit": DEFAULT_HISTORY_LIMIT,
        
        # App info
        "version": APP_VERSION,
        "default_model": DEFAULT_LLM_MODEL,
    }
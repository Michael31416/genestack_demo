"""
Unit tests for LLM service rate limiting and error handling.
Tests the new exception types and rate limiting functionality.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import httpx
import time

from app.services.llm_service import (
    LLMService,
    LLMRateLimitError,
    LLMQuotaExceededError,
    LLMAuthenticationError,
    LLMServiceUnavailableError,
    _check_rate_limit,
    _rate_limit_tracker
)


@pytest.mark.unit
class TestLLMRateLimiting:
    
    def setup_method(self):
        """Clear rate limit tracker before each test."""
        _rate_limit_tracker.clear()
    
    def test_rate_limit_tracking_first_request(self):
        """Test that first request is allowed."""
        _check_rate_limit("test_provider", requests_per_minute=3)
        assert len(_rate_limit_tracker["test_provider"]) == 1
    
    def test_rate_limit_tracking_within_limit(self):
        """Test that requests within limit are allowed."""
        for i in range(3):
            _check_rate_limit("test_provider", requests_per_minute=3)
        
        assert len(_rate_limit_tracker["test_provider"]) == 3
    
    def test_rate_limit_exceeded(self):
        """Test that rate limit is enforced."""
        # Fill up the limit
        for i in range(3):
            _check_rate_limit("test_provider", requests_per_minute=3)
        
        # Next request should fail
        with pytest.raises(LLMRateLimitError) as exc_info:
            _check_rate_limit("test_provider", requests_per_minute=3)
        
        assert "Rate limit exceeded" in str(exc_info.value)
        assert exc_info.value.retry_after is not None
    
    def test_rate_limit_window_expiry(self):
        """Test that rate limit window resets after time."""
        # Mock time to control the window
        with patch('app.services.llm_service.time.time') as mock_time:
            # Start at time 0
            mock_time.return_value = 0
            
            # Fill up the limit
            for i in range(3):
                _check_rate_limit("test_provider", requests_per_minute=3)
            
            # Advance time by 61 seconds (past the window)
            mock_time.return_value = 61
            
            # Should be able to make requests again
            _check_rate_limit("test_provider", requests_per_minute=3)
            assert len(_rate_limit_tracker["test_provider"]) == 1
    
    @pytest.mark.asyncio
    async def test_openai_authentication_error(self):
        """Test OpenAI authentication error handling."""
        service = LLMService("openai", "invalid-key")
        
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Invalid API key"
        mock_response.headers = {}
        
        error = httpx.HTTPStatusError("Unauthorized", request=None, response=mock_response)
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.post.side_effect = error
            
            with pytest.raises(LLMAuthenticationError) as exc_info:
                await service.analyze_correlation({})
            
            assert "Invalid OpenAI API key" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_openai_quota_exceeded_error(self):
        """Test OpenAI quota exceeded error handling."""
        service = LLMService("openai", "sk-test-key")
        
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "You have exceeded your quota for requests"
        mock_response.headers = {'retry-after': '3600'}
        
        error = httpx.HTTPStatusError("Too Many Requests", request=None, response=mock_response)
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.post.side_effect = error
            
            with pytest.raises(LLMQuotaExceededError) as exc_info:
                await service.analyze_correlation({})
            
            assert "quota exceeded" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_openai_rate_limit_error(self):
        """Test OpenAI rate limit error handling."""
        service = LLMService("openai", "sk-test-key")
        
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "Rate limit exceeded"
        mock_response.headers = {'retry-after': '60'}
        
        error = httpx.HTTPStatusError("Too Many Requests", request=None, response=mock_response)
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.post.side_effect = error
            
            with pytest.raises(LLMRateLimitError) as exc_info:
                await service.analyze_correlation({})
            
            assert "rate limit exceeded" in str(exc_info.value).lower()
            assert exc_info.value.retry_after == 60
    
    @pytest.mark.asyncio
    async def test_openai_service_unavailable_error(self):
        """Test OpenAI service unavailable error handling."""
        service = LLMService("openai", "sk-test-key")
        
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_response.text = "Service temporarily unavailable"
        mock_response.headers = {}
        
        error = httpx.HTTPStatusError("Service Unavailable", request=None, response=mock_response)
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.post.side_effect = error
            
            with pytest.raises(Exception) as exc_info:
                await service.analyze_correlation({})
            
            # Should get RetryError after 3 attempts of LLMServiceUnavailableError
            assert "service unavailable" in str(exc_info.value).lower() or "RetryError" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_anthropic_authentication_error(self):
        """Test Anthropic authentication error handling."""
        service = LLMService("anthropic", "invalid-key")
        
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Invalid API key"
        mock_response.headers = {}
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.post.return_value = mock_response
            
            with pytest.raises(LLMAuthenticationError) as exc_info:
                await service.analyze_correlation({})
            
            assert "Invalid Anthropic API key" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test timeout error handling."""
        service = LLMService("openai", "sk-test-key")
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.post.side_effect = httpx.TimeoutException("Request timeout")
            
            with pytest.raises(Exception) as exc_info:
                await service.analyze_correlation({})
            
            # Should get RetryError after 3 attempts of timeout
            assert "timeout" in str(exc_info.value).lower() or "RetryError" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_connection_error_handling(self):
        """Test connection error handling."""
        service = LLMService("openai", "sk-test-key")
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.post.side_effect = httpx.ConnectError("Connection failed")
            
            with pytest.raises(Exception) as exc_info:
                await service.analyze_correlation({})
            
            # Should get RetryError after 3 attempts of connection error  
            assert "connect" in str(exc_info.value).lower() or "RetryError" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_internal_rate_limiting(self):
        """Test internal rate limiting mechanism."""
        service = LLMService("openai", "sk-test-key")
        
        # Mock successful API responses
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "choices": [{"message": {"content": '{"verdict": "strong", "confidence": 0.8}'}}]
            }
            mock_response.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_response
            
            # Make requests up to the limit
            for i in range(20):
                await service.analyze_correlation({})
            
            # Next request should trigger internal rate limiting
            with pytest.raises(Exception):  # Either LLMRateLimitError or RetryError
                await service.analyze_correlation({})
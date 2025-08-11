"""
Unit tests for LLM service.
Tests OpenAI and Anthropic API integrations with mocked responses.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import json
import httpx
import asyncio
import os

from app.services.llm_service import LLMService


@pytest.mark.unit
class TestLLMService:
    
    def test_llm_service_init_openai(self):
        """Test LLMService initialization with OpenAI provider."""
        service = LLMService("openai", "sk-test-key")
        assert service.provider == "openai"
        assert service.api_key == "sk-test-key"
    
    def test_llm_service_init_anthropic(self):
        """Test LLMService initialization with Anthropic provider."""
        service = LLMService("anthropic", "ant-test-key")
        assert service.provider == "anthropic"
        assert service.api_key == "ant-test-key"
    
    @pytest.mark.asyncio
    async def test_llm_service_init_invalid_provider(self):
        """Test LLMService with invalid provider."""
        service = LLMService("invalid", "test-key")
        
        with pytest.raises((ValueError, Exception)) as exc_info:
            await service.analyze_correlation({})
        
        # Check that the original error is ValueError about unsupported provider
        if hasattr(exc_info.value, 'last_attempt'):
            assert "Unsupported provider" in str(exc_info.value.last_attempt.exception())
        else:
            assert "Unsupported provider" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_openai_analysis_success(self, sample_openai_response):
        """Test successful OpenAI analysis."""
        service = LLMService("openai", "sk-test-key")
        
        mock_response = MagicMock()
        mock_response.json.return_value = sample_openai_response
        mock_response.raise_for_status = MagicMock()
        
        evidence = {
            "query": {"gene": {"symbol": "TP53"}, "disease": {"label": "lung cancer"}},
            "opentargets": {"overall_association_score": 0.85}
        }
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.post.return_value = mock_response
            
            result = await service.analyze_correlation(evidence, "gpt-5-mini")
        
        assert result["verdict"] == "strong"
        assert result["confidence"] == 0.85
        assert "drivers" in result
        assert "key_points" in result
    
    @pytest.mark.asyncio
    async def test_openai_analysis_with_response_format_fallback(self, sample_openai_response):
        """Test OpenAI analysis with response_format fallback for older models."""
        service = LLMService("openai", "sk-test-key")
        
        # Mock the first call to fail with response_format error, second to succeed
        mock_response_error = MagicMock()
        mock_response_error.status_code = 400
        mock_response_error.text = "response_format not supported"
        
        mock_response_success = MagicMock()
        mock_response_success.json.return_value = sample_openai_response
        mock_response_success.raise_for_status = MagicMock()
        
        error = httpx.HTTPStatusError("Bad Request", request=None, response=mock_response_error)
        
        evidence = {"query": {"gene": {"symbol": "TP53"}}}
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.post.side_effect = [error, mock_response_success]
            
            result = await service.analyze_correlation(evidence, "gpt-3.5-turbo")
        
        assert result["verdict"] == "strong"
        assert mock_client.post.call_count == 2  # First call fails, second succeeds
    
    @pytest.mark.asyncio
    async def test_openai_analysis_json_parse_error(self):
        """Test OpenAI analysis with JSON parsing error."""
        service = LLMService("openai", "sk-test-key")
        
        mock_response = MagicMock()
        # Return invalid JSON
        invalid_response = {
            "choices": [{"message": {"content": "This is not valid JSON"}}]
        }
        mock_response.json.return_value = invalid_response
        mock_response.raise_for_status = MagicMock()
        
        evidence = {"query": {"gene": {"symbol": "TP53"}}}
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.post.return_value = mock_response
            
            result = await service.analyze_correlation(evidence)
        
        assert result["verdict"] == "inconclusive"
        assert result["confidence"] == 0.0
        assert "_raw" in result
        assert result["_raw"] == "This is not valid JSON"
    
    @pytest.mark.asyncio
    async def test_anthropic_analysis_success(self, sample_anthropic_response):
        """Test successful Anthropic analysis."""
        service = LLMService("anthropic", "ant-test-key")
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_anthropic_response
        
        evidence = {
            "query": {"gene": {"symbol": "BRCA1"}, "disease": {"label": "breast cancer"}},
            "opentargets": {"overall_association_score": 0.75}
        }
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.post.return_value = mock_response
            
            result = await service.analyze_correlation(evidence, "claude-3-haiku-20240307")
        
        assert result["verdict"] == "moderate"
        assert result["confidence"] == 0.75
        assert "drivers" in result
    
    @pytest.mark.asyncio
    async def test_anthropic_analysis_api_error(self):
        """Test Anthropic analysis with API error."""
        service = LLMService("anthropic", "ant-test-key")
        
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        
        evidence = {"query": {"gene": {"symbol": "TP53"}}}
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.post.return_value = mock_response
            
            with pytest.raises(Exception) as exc_info:
                await service.analyze_correlation(evidence)
            
            # Check that the underlying error was about Anthropic API
            retry_error = exc_info.value
            if hasattr(retry_error, 'last_attempt'):
                original_error = retry_error.last_attempt.exception()
                assert "Anthropic API error 500" in str(original_error)
            else:
                assert "API error 500" in str(retry_error)
    
    @pytest.mark.asyncio
    async def test_anthropic_analysis_json_extraction(self):
        """Test Anthropic analysis with JSON extraction from mixed content."""
        service = LLMService("anthropic", "ant-test-key")
        
        # Response with JSON embedded in text
        mixed_response = {
            "content": [
                {
                    "text": 'Here is my analysis: {"verdict": "weak", "confidence": 0.3} This concludes the analysis.'
                }
            ]
        }
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mixed_response
        
        evidence = {"query": {"gene": {"symbol": "TP53"}}}
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.post.return_value = mock_response
            
            result = await service.analyze_correlation(evidence)
        
        assert result["verdict"] == "weak"
        assert result["confidence"] == 0.3
    
    @pytest.mark.asyncio
    async def test_anthropic_analysis_json_parse_error(self):
        """Test Anthropic analysis with JSON parsing error."""
        service = LLMService("anthropic", "ant-test-key")
        
        invalid_response = {
            "content": [{"text": "This response has no valid JSON"}]
        }
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = invalid_response
        
        evidence = {"query": {"gene": {"symbol": "TP53"}}}
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.post.return_value = mock_response
            
            result = await service.analyze_correlation(evidence)
        
        assert result["verdict"] == "inconclusive"
        assert result["confidence"] == 0.0
        assert "_raw" in result
    
    @pytest.mark.asyncio
    async def test_retry_mechanism(self):
        """Test that retry mechanism works for LLM service."""
        service = LLMService("openai", "sk-test-key")
        
        # First call fails, subsequent calls succeed
        error_response = MagicMock()
        error_response.status_code = 500
        
        success_response = MagicMock()
        success_response.json.return_value = {
            "choices": [{"message": {"content": '{"verdict": "strong", "confidence": 0.8}'}}]
        }
        success_response.raise_for_status = MagicMock()
        
        evidence = {"query": {"gene": {"symbol": "TP53"}}}
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            # First call raises error, second succeeds
            mock_client.post.side_effect = [
                httpx.HTTPStatusError("Server Error", request=None, response=error_response),
                success_response
            ]
            
            with patch('asyncio.sleep'):  # Speed up the test
                result = await service.analyze_correlation(evidence)
        
        assert result["verdict"] == "strong"
        assert mock_client.post.call_count >= 2
    
    @pytest.mark.asyncio
    async def test_openai_environment_variables(self):
        """Test OpenAI service uses environment variables."""
        service = LLMService("openai", "sk-test-key")
        
        with patch.dict(os.environ, {"OPENAI_BASE_URL": "https://custom.openai.com/v1"}):
            with patch('httpx.AsyncClient') as mock_client_class:
                mock_client = AsyncMock()
                mock_client_class.return_value.__aenter__.return_value = mock_client
                mock_response = MagicMock()
                mock_response.json.return_value = {"choices": [{"message": {"content": "{}"}}]}
                mock_response.raise_for_status = MagicMock()
                mock_client.post.return_value = mock_response
                
                await service.analyze_correlation({})
                
                # Check that the custom URL was used
                mock_client.post.assert_called_once()
                call_args = mock_client.post.call_args
                assert call_args[0][0] == "https://custom.openai.com/v1/chat/completions"
    
    @pytest.mark.asyncio
    async def test_anthropic_environment_variables(self):
        """Test Anthropic service uses environment variables."""
        service = LLMService("anthropic", "ant-test-key")
        
        with patch.dict(os.environ, {"ANTHROPIC_BASE_URL": "https://custom.anthropic.com/v1"}):
            with patch('httpx.AsyncClient') as mock_client_class:
                mock_client = AsyncMock()
                mock_client_class.return_value.__aenter__.return_value = mock_client
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"content": [{"text": "{}"}]}
                mock_client.post.return_value = mock_response
                
                await service.analyze_correlation({})
                
                # Check that the custom URL was used
                mock_client.post.assert_called_once()
                call_args = mock_client.post.call_args
                assert call_args[0][0] == "https://custom.anthropic.com/v1/messages"
    
    def test_system_prompt_content(self):
        """Test that system prompt contains required elements."""
        from app.services.llm_service import LLM_SYSTEM_PROMPT
        
        # Check that the prompt contains key instructions
        assert "biomedical evidence-synthesis" in LLM_SYSTEM_PROMPT
        assert "verdict" in LLM_SYSTEM_PROMPT
        assert "confidence" in LLM_SYSTEM_PROMPT
        assert "strong|moderate|weak|no_evidence|inconclusive" in LLM_SYSTEM_PROMPT
        assert "JSON" in LLM_SYSTEM_PROMPT
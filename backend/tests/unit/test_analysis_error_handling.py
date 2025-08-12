"""
Unit tests for analysis service error handling.
Tests graceful handling of LLM service errors.
"""

import pytest
from unittest.mock import patch, MagicMock
import json

from app.services.analysis_service import AnalysisService
from app.services.llm_service import (
    LLMRateLimitError,
    LLMQuotaExceededError,
    LLMAuthenticationError,
    LLMServiceUnavailableError
)
from app.models import Analysis, Result, Session as UserSession
from app.schemas import AnalysisRequest


@pytest.mark.unit
class TestAnalysisErrorHandling:
    
    @pytest.mark.asyncio
    async def test_handle_authentication_error(self, db_session):
        """Test handling of LLM authentication errors."""
        analysis = Analysis(
            id=1,
            user_id=1,
            session_id="test-session",
            gene_symbol="TP53",
            disease_label="lung cancer"
        )
        db_session.add(analysis)
        db_session.commit()
        
        user_session = UserSession(
            id="test-session",
            api_provider="openai",
            api_key_encrypted="invalid-key"
        )
        
        request = AnalysisRequest(
            gene="TP53",
            disease="lung cancer",
            model="gpt-5-mini"
        )
        
        service = AnalysisService(db_session)
        
        # Mock data fetching to succeed
        mock_evidence = {"query": {"gene": {"symbol": "TP53"}}}
        
        with patch.object(service, '_fetch_evidence', return_value=mock_evidence):
            with patch('app.services.analysis_service.LLMService') as mock_llm_class:
                mock_llm_instance = MagicMock()
                mock_llm_instance.analyze_correlation.side_effect = LLMAuthenticationError("Invalid API key")
                mock_llm_class.return_value = mock_llm_instance
                
                await service.run_analysis(1, request, user_session)
        
        # Check result
        result = db_session.query(Result).filter(Result.analysis_id == 1).first()
        assert result is not None
        assert result.verdict == "error"
        assert result.confidence == 0.0
        assert "API Authentication Error" in result.error_message
    
    @pytest.mark.asyncio
    async def test_handle_quota_exceeded_error(self, db_session):
        """Test handling of LLM quota exceeded errors."""
        analysis = Analysis(
            id=1,
            user_id=1,
            session_id="test-session",
            gene_symbol="TP53",
            disease_label="lung cancer"
        )
        db_session.add(analysis)
        db_session.commit()
        
        user_session = UserSession(
            id="test-session",
            api_provider="openai",
            api_key_encrypted="sk-test-key"
        )
        
        request = AnalysisRequest(
            gene="TP53",
            disease="lung cancer",
            model="gpt-5-mini"
        )
        
        service = AnalysisService(db_session)
        
        mock_evidence = {"query": {"gene": {"symbol": "TP53"}}}
        
        with patch.object(service, '_fetch_evidence', return_value=mock_evidence):
            with patch('app.services.analysis_service.LLMService') as mock_llm_class:
                mock_llm_instance = MagicMock()
                mock_llm_instance.analyze_correlation.side_effect = LLMQuotaExceededError("Quota exceeded")
                mock_llm_class.return_value = mock_llm_instance
                
                await service.run_analysis(1, request, user_session)
        
        # Check result
        result = db_session.query(Result).filter(Result.analysis_id == 1).first()
        assert result is not None
        assert result.verdict == "error"
        assert result.confidence == 0.0
        assert "API Quota Exceeded" in result.error_message
    
    @pytest.mark.asyncio
    async def test_handle_rate_limit_error(self, db_session):
        """Test handling of LLM rate limit errors."""
        analysis = Analysis(
            id=1,
            user_id=1,
            session_id="test-session",
            gene_symbol="TP53",
            disease_label="lung cancer"
        )
        db_session.add(analysis)
        db_session.commit()
        
        user_session = UserSession(
            id="test-session",
            api_provider="openai", 
            api_key_encrypted="sk-test-key"
        )
        
        request = AnalysisRequest(
            gene="TP53",
            disease="lung cancer",
            model="gpt-5-mini"
        )
        
        service = AnalysisService(db_session)
        
        mock_evidence = {"query": {"gene": {"symbol": "TP53"}}}
        
        with patch.object(service, '_fetch_evidence', return_value=mock_evidence):
            with patch('app.services.analysis_service.LLMService') as mock_llm_class:
                mock_llm_instance = MagicMock()
                mock_llm_instance.analyze_correlation.side_effect = LLMRateLimitError("Rate limited", retry_after=60)
                mock_llm_class.return_value = mock_llm_instance
                
                await service.run_analysis(1, request, user_session)
        
        # Check result
        result = db_session.query(Result).filter(Result.analysis_id == 1).first()
        assert result is not None
        assert result.verdict == "retry_later"
        assert result.confidence == 0.0
        assert "API Rate Limited" in result.error_message
    
    @pytest.mark.asyncio
    async def test_handle_service_unavailable_error(self, db_session):
        """Test handling of LLM service unavailable errors."""
        analysis = Analysis(
            id=1,
            user_id=1,
            session_id="test-session",
            gene_symbol="TP53",
            disease_label="lung cancer"
        )
        db_session.add(analysis)
        db_session.commit()
        
        user_session = UserSession(
            id="test-session",
            api_provider="anthropic",
            api_key_encrypted="ant-test-key"
        )
        
        request = AnalysisRequest(
            gene="TP53",
            disease="lung cancer",
            model="claude-sonnet-4-20250514"
        )
        
        service = AnalysisService(db_session)
        
        mock_evidence = {"query": {"gene": {"symbol": "TP53"}}}
        
        with patch.object(service, '_fetch_evidence', return_value=mock_evidence):
            with patch('app.services.analysis_service.LLMService') as mock_llm_class:
                mock_llm_instance = MagicMock()
                mock_llm_instance.analyze_correlation.side_effect = LLMServiceUnavailableError("Service unavailable")
                mock_llm_class.return_value = mock_llm_instance
                
                await service.run_analysis(1, request, user_session)
        
        # Check result
        result = db_session.query(Result).filter(Result.analysis_id == 1).first()
        assert result is not None
        assert result.verdict == "service_unavailable"
        assert result.confidence == 0.0
        assert "LLM Service Unavailable" in result.error_message
    
    @pytest.mark.asyncio
    async def test_analysis_completes_without_llm_on_error(self, db_session):
        """Test that analysis completes with evidence even if LLM fails."""
        analysis = Analysis(
            id=1,
            user_id=1,
            session_id="test-session",
            gene_symbol="TP53",
            disease_label="lung cancer"
        )
        db_session.add(analysis)
        db_session.commit()
        
        user_session = UserSession(
            id="test-session",
            api_provider="openai",
            api_key_encrypted="sk-test-key"
        )
        
        request = AnalysisRequest(
            gene="TP53",
            disease="lung cancer",
            model="gpt-5-mini"
        )
        
        service = AnalysisService(db_session)
        
        mock_evidence = {
            "query": {"gene": {"symbol": "TP53", "ensembl_id": "ENSG00000141510"}},
            "opentargets": {"overall_association_score": 0.85}
        }
        
        with patch.object(service, '_fetch_evidence', return_value=mock_evidence):
            with patch('app.services.analysis_service.LLMService') as mock_llm_class:
                mock_llm_instance = MagicMock()
                mock_llm_instance.analyze_correlation.side_effect = LLMAuthenticationError("Invalid API key")
                mock_llm_class.return_value = mock_llm_instance
                
                await service.run_analysis(1, request, user_session)
        
        # Check analysis completed
        updated_analysis = db_session.query(Analysis).filter(Analysis.id == 1).first()
        assert updated_analysis.status == "completed"
        assert updated_analysis.completed_at is not None
        
        # Check result has evidence even though LLM failed
        result = db_session.query(Result).filter(Result.analysis_id == 1).first()
        assert result is not None
        assert result.verdict == "error"
        assert result.evidence_json is not None
        
        evidence = json.loads(result.evidence_json)
        assert evidence["opentargets"]["overall_association_score"] == 0.85
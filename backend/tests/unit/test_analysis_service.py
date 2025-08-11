"""
Unit tests for analysis service.
Tests the analysis orchestration with mocked dependencies.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import json

from app.services.analysis_service import AnalysisService
from app.models import Analysis, Result, Session as UserSession
from app.schemas import AnalysisRequest


@pytest.mark.unit
class TestAnalysisService:
    
    def test_analysis_service_init(self, db_session):
        """Test AnalysisService initialization."""
        service = AnalysisService(db_session)
        assert service.db == db_session
    
    @pytest.mark.asyncio
    async def test_run_analysis_success(self, db_session, mock_analysis_request):
        """Test successful analysis run."""
        # Create test data
        analysis = Analysis(
            id=1,
            user_id=1,
            session_id="test-session",
            gene_symbol="TP53",
            disease_label="lung cancer",
            params_json=json.dumps(mock_analysis_request)
        )
        db_session.add(analysis)
        db_session.commit()
        
        user_session = UserSession(
            id="test-session",
            api_provider="openai",
            api_key_encrypted="sk-test-key"
        )
        
        request = AnalysisRequest(**mock_analysis_request)
        
        # Mock the evidence fetching
        mock_evidence = {
            "query": {
                "gene": {"symbol": "TP53", "ensembl_id": "ENSG00000141510"},
                "disease": {"label": "lung cancer", "efo_id": "EFO_0000616"}
            },
            "synonyms": {"gene": ["TP53"], "disease": ["lung cancer"]},
            "opentargets": {"overall_association_score": 0.85},
            "literature": [],
            "gwas_catalog": []
        }
        
        mock_llm_output = {
            "verdict": "strong",
            "confidence": 0.85,
            "drivers": {"genetic": {"present": True, "summary": "Strong evidence"}},
            "key_points": [{"statement": "TP53 is mutated in lung cancer"}],
            "conflicts_or_gaps": [],
            "recommended_next_steps": []
        }
        
        service = AnalysisService(db_session)
        
        with patch.object(service, '_fetch_evidence', return_value=mock_evidence) as mock_fetch:
            with patch('app.services.analysis_service.LLMService') as mock_llm_class:
                mock_llm_instance = AsyncMock()
                mock_llm_instance.analyze_correlation.return_value = mock_llm_output
                mock_llm_class.return_value = mock_llm_instance
                
                await service.run_analysis(1, request, user_session)
        
        # Verify analysis was updated
        updated_analysis = db_session.query(Analysis).filter(Analysis.id == 1).first()
        assert updated_analysis.status == "completed"
        assert updated_analysis.completed_at is not None
        assert updated_analysis.ensembl_id == "ENSG00000141510"
        assert updated_analysis.disease_efo == "EFO_0000616"
        
        # Verify result was created
        result = db_session.query(Result).filter(Result.analysis_id == 1).first()
        assert result is not None
        assert result.verdict == "strong"
        assert result.confidence == 0.85
        assert result.evidence_json is not None
        assert result.llm_output_json is not None
    
    @pytest.mark.asyncio
    async def test_run_analysis_no_model(self, db_session, mock_analysis_request):
        """Test analysis run without LLM model."""
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
        
        # Request without model
        request_data = mock_analysis_request.copy()
        request_data["model"] = None
        request = AnalysisRequest(**request_data)
        
        mock_evidence = {
            "query": {
                "gene": {"symbol": "TP53", "ensembl_id": "ENSG00000141510"},
                "disease": {"label": "lung cancer", "efo_id": "EFO_0000616"}
            },
            "synonyms": {"gene": ["TP53"], "disease": ["lung cancer"]},
            "opentargets": None,
            "literature": [],
            "gwas_catalog": []
        }
        
        service = AnalysisService(db_session)
        
        with patch.object(service, '_fetch_evidence', return_value=mock_evidence):
            await service.run_analysis(1, request, user_session)
        
        # Verify analysis completed without LLM
        updated_analysis = db_session.query(Analysis).filter(Analysis.id == 1).first()
        assert updated_analysis.status == "completed"
        
        result = db_session.query(Result).filter(Result.analysis_id == 1).first()
        assert result is not None
        assert result.verdict is None
        assert result.confidence is None
        assert result.llm_output_json is None
        assert result.evidence_json is not None
    
    @pytest.mark.asyncio
    async def test_run_analysis_llm_error(self, db_session, mock_analysis_request):
        """Test analysis run with LLM error."""
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
        
        request = AnalysisRequest(**mock_analysis_request)
        
        mock_evidence = {
            "query": {
                "gene": {"symbol": "TP53", "ensembl_id": "ENSG00000141510"},
                "disease": {"label": "lung cancer", "efo_id": "EFO_0000616"}
            }
        }
        
        service = AnalysisService(db_session)
        
        with patch.object(service, '_fetch_evidence', return_value=mock_evidence):
            with patch('app.services.analysis_service.LLMService') as mock_llm_class:
                mock_llm_instance = AsyncMock()
                mock_llm_instance.analyze_correlation.side_effect = Exception("LLM API Error")
                mock_llm_class.return_value = mock_llm_instance
                
                await service.run_analysis(1, request, user_session)
        
        # Verify error was handled
        result = db_session.query(Result).filter(Result.analysis_id == 1).first()
        assert result is not None
        assert "LLM API Error" in result.error_message
        assert result.verdict == "error"
        assert result.confidence == 0.0
    
    @pytest.mark.asyncio
    async def test_run_analysis_data_fetch_error(self, db_session, mock_analysis_request):
        """Test analysis run with data fetching error."""
        analysis = Analysis(
            id=1,
            user_id=1,
            session_id="test-session",
            gene_symbol="INVALID_GENE",
            disease_label="invalid disease"
        )
        db_session.add(analysis)
        db_session.commit()
        
        user_session = UserSession(
            id="test-session",
            api_provider="openai",
            api_key_encrypted="sk-test-key"
        )
        
        request = AnalysisRequest(**mock_analysis_request)
        service = AnalysisService(db_session)
        
        with patch.object(service, '_fetch_evidence', side_effect=Exception("Data fetch error")):
            await service.run_analysis(1, request, user_session)
        
        # Verify analysis failed
        updated_analysis = db_session.query(Analysis).filter(Analysis.id == 1).first()
        assert updated_analysis.status == "failed"
        assert updated_analysis.completed_at is not None
        
        result = db_session.query(Result).filter(Result.analysis_id == 1).first()
        assert result is not None
        assert result.error_message == "Data fetch error"
        assert result.verdict == "error"
    
    @pytest.mark.asyncio
    async def test_fetch_evidence_success(self, db_session):
        """Test successful evidence fetching."""
        service = AnalysisService(db_session)
        
        request = AnalysisRequest(
            gene="TP53",
            disease="lung cancer",
            since_year=2020,
            max_abstracts=5,
            include_gwas=True,
            model="gpt-5-mini"
        )
        
        # Mock all the data fetching functions
        with patch('app.services.analysis_service.resolve_gene_symbol', 
                  return_value=("ENSG00000141510", ["TP53"])) as mock_gene:
            with patch('app.services.analysis_service.resolve_disease_label',
                      return_value=("EFO_0000616", "MONDO_123", ["lung cancer"])) as mock_disease:
                with patch('app.services.analysis_service.get_opentargets_association',
                          return_value={"overall_association_score": 0.85}) as mock_ot:
                    with patch('app.services.analysis_service.get_literature_evidence',
                              return_value=[{"pmid": "123", "title": "Test"}]) as mock_lit:
                        with patch('app.services.analysis_service.get_gwas_associations',
                                  return_value=[{"association_id": "456"}]) as mock_gwas:
                            
                            evidence = await service._fetch_evidence(request)
        
        assert evidence["query"]["gene"]["symbol"] == "TP53"
        assert evidence["query"]["gene"]["ensembl_id"] == "ENSG00000141510"
        assert evidence["query"]["disease"]["label"] == "lung cancer"
        assert evidence["query"]["disease"]["efo_id"] == "EFO_0000616"
        assert evidence["query"]["disease"]["mondo_id"] == "MONDO_123"
        assert evidence["synonyms"]["gene"] == ["TP53"]
        assert evidence["synonyms"]["disease"] == ["lung cancer"]
        assert evidence["opentargets"]["overall_association_score"] == 0.85
        assert len(evidence["literature"]) == 1
        assert len(evidence["gwas_catalog"]) == 1
    
    @pytest.mark.asyncio
    async def test_fetch_evidence_no_gwas(self, db_session):
        """Test evidence fetching without GWAS data."""
        service = AnalysisService(db_session)
        
        request = AnalysisRequest(
            gene="TP53",
            disease="lung cancer",
            include_gwas=False,
            model="gpt-5-mini"
        )
        
        with patch('app.services.analysis_service.resolve_gene_symbol', 
                  return_value=("ENSG00000141510", ["TP53"])):
            with patch('app.services.analysis_service.resolve_disease_label',
                      return_value=("EFO_0000616", None, ["lung cancer"])):
                with patch('app.services.analysis_service.get_opentargets_association',
                          return_value=None):
                    with patch('app.services.analysis_service.get_literature_evidence',
                              return_value=[]):
                        
                        evidence = await service._fetch_evidence(request)
        
        assert evidence["gwas_catalog"] == []
        assert evidence["opentargets"] is None
        assert evidence["literature"] == []
    
    @pytest.mark.asyncio
    async def test_fetch_evidence_with_exceptions(self, db_session):
        """Test evidence fetching with some API failures."""
        service = AnalysisService(db_session)
        
        request = AnalysisRequest(
            gene="TP53",
            disease="lung cancer",
            include_gwas=True,
            model="gpt-5-mini"
        )
        
        # Mock the individual API functions instead of asyncio.gather
        with patch('app.services.analysis_service.resolve_gene_symbol', 
                  return_value=("ENSG00000141510", ["TP53"])):
            with patch('app.services.analysis_service.resolve_disease_label',
                      return_value=("EFO_0000616", None, ["lung cancer"])):
                with patch('app.services.analysis_service.get_opentargets_association',
                          return_value={"overall_association_score": 0.85}):
                    with patch('app.services.analysis_service.get_literature_evidence',
                              side_effect=Exception("Literature API error")):
                        with patch('app.services.analysis_service.get_gwas_associations',
                                  return_value=[{"association_id": "123"}]):
                            
                            evidence = await service._fetch_evidence(request)
        
        # Should handle exceptions gracefully
        assert evidence["opentargets"]["overall_association_score"] == 0.85
        assert evidence["literature"] == []  # Failed, so empty list
        assert len(evidence["gwas_catalog"]) == 1
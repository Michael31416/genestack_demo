"""
Unit tests for data fetching services.
Tests external API integrations with mocked HTTP responses.
"""

import pytest
from unittest.mock import AsyncMock, patch
import httpx

from app.services.data_fetcher import (
    resolve_gene_symbol,
    resolve_disease_label,
    get_opentargets_association,
    get_literature_evidence,
    get_gwas_associations,
    split_sentences
)


@pytest.mark.unit
class TestDataFetcher:
    
    @pytest.mark.asyncio
    async def test_resolve_gene_symbol_success(self, sample_gene_data):
        """Test successful gene symbol resolution."""
        mock_client = AsyncMock()
        mock_client.get.return_value.json.return_value = sample_gene_data
        mock_client.get.return_value.raise_for_status = AsyncMock()
        
        with patch('app.services.data_fetcher.fetch_json', return_value=sample_gene_data):
            ensembl_id, synonyms = await resolve_gene_symbol(mock_client, "TP53")
        
        assert ensembl_id == "ENSG00000141510"
        assert "TP53" in synonyms
        assert len(synonyms) > 0
    
    @pytest.mark.asyncio
    async def test_resolve_gene_symbol_failure(self):
        """Test gene symbol resolution failure."""
        mock_client = AsyncMock()
        
        with patch('app.services.data_fetcher.fetch_json', side_effect=Exception("API Error")):
            with pytest.raises(RuntimeError, match="Could not resolve gene symbol"):
                await resolve_gene_symbol(mock_client, "INVALID")
    
    @pytest.mark.asyncio
    async def test_resolve_disease_label_success(self, sample_disease_data):
        """Test successful disease label resolution."""
        mock_client = AsyncMock()
        
        with patch('app.services.data_fetcher.fetch_json', return_value=sample_disease_data):
            efo_id, mondo_id, synonyms = await resolve_disease_label(mock_client, "lung cancer")
        
        assert efo_id == "EFO_0000616"
        assert "lung cancer" in synonyms
        assert len(synonyms) > 0
    
    @pytest.mark.asyncio
    async def test_resolve_disease_label_no_results(self):
        """Test disease resolution with no results."""
        mock_client = AsyncMock()
        empty_response = {"response": {"docs": []}}
        
        with patch('app.services.data_fetcher.fetch_json', return_value=empty_response):
            with pytest.raises(RuntimeError, match="Could not resolve disease"):
                await resolve_disease_label(mock_client, "unknown disease")
    
    @pytest.mark.asyncio
    async def test_get_opentargets_association_found(self, sample_opentargets_data):
        """Test successful Open Targets association retrieval."""
        mock_client = AsyncMock()
        
        with patch('app.services.data_fetcher.fetch_json', return_value=sample_opentargets_data):
            result = await get_opentargets_association(mock_client, "ENSG00000141510", "EFO_0000616")
        
        assert result is not None
        assert result["overall_association_score"] == 0.85
        assert result["target"]["id"] == "ENSG00000141510"
        assert len(result["datatype_scores"]) == 2
    
    @pytest.mark.asyncio
    async def test_get_opentargets_association_not_found(self):
        """Test Open Targets association not found."""
        mock_client = AsyncMock()
        empty_response = {
            "data": {
                "disease": {
                    "associatedTargets": {
                        "count": 0,
                        "rows": []
                    }
                }
            }
        }
        
        with patch('app.services.data_fetcher.fetch_json', return_value=empty_response):
            result = await get_opentargets_association(mock_client, "ENSG00000000000", "EFO_0000000")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_literature_evidence_success(self, sample_literature_data):
        """Test successful literature evidence retrieval."""
        mock_client = AsyncMock()
        
        with patch('app.services.data_fetcher.fetch_json', return_value=sample_literature_data):
            result = await get_literature_evidence(
                mock_client, 
                ["TP53"], 
                ["lung cancer"], 
                2020, 
                5
            )
        
        assert len(result) == 1
        assert result[0]["pmid"] == "12345678"
        assert result[0]["title"] == "TP53 mutations in lung cancer"
        assert len(result[0]["sentences"]) > 0
    
    @pytest.mark.asyncio
    async def test_get_literature_evidence_no_matches(self):
        """Test literature search with no evidence sentences."""
        mock_client = AsyncMock()
        no_match_data = {
            "resultList": {
                "result": [
                    {
                        "pmid": "12345678",
                        "title": "Unrelated study",
                        "pubYear": "2023",
                        "source": "PubMed",
                        "authorString": "Smith J",
                        "abstractText": "This abstract doesn't mention our gene or disease."
                    }
                ]
            }
        }
        
        with patch('app.services.data_fetcher.fetch_json', return_value=no_match_data):
            result = await get_literature_evidence(
                mock_client, 
                ["TP53"], 
                ["lung cancer"], 
                2020, 
                5
            )
        
        assert len(result) == 0
    
    @pytest.mark.asyncio
    async def test_get_gwas_associations_success(self, sample_gwas_data):
        """Test successful GWAS associations retrieval."""
        mock_client = AsyncMock()
        
        with patch('app.services.data_fetcher.fetch_json', return_value=sample_gwas_data):
            result = await get_gwas_associations(mock_client, "lung carcinoma", "TP53", 10)
        
        assert len(result) == 1
        assert result[0]["association_id"] == "12345"
        assert result[0]["pvalue"] == 5.2
        assert result[0]["pvalueExponent"] == -8
    
    @pytest.mark.asyncio
    async def test_get_gwas_associations_gene_not_found(self, sample_gwas_data):
        """Test GWAS associations when gene is not found."""
        mock_client = AsyncMock()
        # Modify the sample data to not include our target gene
        modified_data = sample_gwas_data.copy()
        modified_data["_embedded"]["associations"][0]["loci"][0]["authorReportedGenes"] = ["OTHER_GENE"]
        modified_data["_embedded"]["associations"][0]["loci"][0]["strongestRiskAlleles"][0]["ensemblGenes"] = [{"geneName": "OTHER_GENE"}]
        
        with patch('app.services.data_fetcher.fetch_json', return_value=modified_data):
            result = await get_gwas_associations(mock_client, "lung carcinoma", "TP53", 10)
        
        assert len(result) == 0
    
    @pytest.mark.asyncio
    async def test_get_gwas_associations_api_error(self):
        """Test GWAS associations with API error."""
        mock_client = AsyncMock()
        
        with patch('app.services.data_fetcher.fetch_json', side_effect=Exception("API Error")):
            result = await get_gwas_associations(mock_client, "lung carcinoma", "TP53", 10)
        
        assert result == []
    
    def test_split_sentences_basic(self):
        """Test basic sentence splitting functionality."""
        text = "This is sentence one. This is sentence two! Is this sentence three?"
        sentences = split_sentences(text)
        
        assert len(sentences) == 3
        assert sentences[0] == "This is sentence one."
        assert sentences[1] == "This is sentence two!"
        assert sentences[2] == "Is this sentence three?"
    
    def test_split_sentences_with_abbreviations(self):
        """Test sentence splitting with abbreviations."""
        text = "This mentions Dr. Smith and e.g. some examples. This is another sentence."
        sentences = split_sentences(text)
        
        assert len(sentences) == 2
        assert "Dr. Smith" in sentences[0]
        assert "e.g." in sentences[0]
    
    def test_split_sentences_empty(self):
        """Test sentence splitting with empty input."""
        assert split_sentences("") == []
        assert split_sentences(None) == []
    
    @pytest.mark.asyncio
    async def test_fetch_json_retry_mechanism(self):
        """Test that retry mechanism works for fetch_json."""
        mock_client = AsyncMock()
        
        # First call fails, second succeeds
        mock_client.get.side_effect = [
            httpx.HTTPStatusError("Server Error", request=None, response=AsyncMock()),
            AsyncMock()
        ]
        mock_client.get.return_value.json.return_value = {"test": "data"}
        mock_client.get.return_value.raise_for_status = AsyncMock()
        
        from app.services.data_fetcher import fetch_json
        
        # This should succeed after retry
        with patch('asyncio.sleep'):  # Speed up the test
            result = await fetch_json(mock_client, "http://test.com")
        
        assert mock_client.get.call_count >= 1
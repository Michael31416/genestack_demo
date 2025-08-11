"""
Analysis service that orchestrates data fetching and LLM analysis.
"""

import json
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
import httpx
from sqlalchemy.orm import Session

from ..models import Analysis, Result, Session as UserSession
from ..schemas import AnalysisRequest
from .data_fetcher import (
    resolve_gene_symbol,
    resolve_disease_label,
    get_opentargets_association,
    get_literature_evidence,
    get_gwas_associations
)
from .llm_service import (
    LLMService, 
    LLMRateLimitError, 
    LLMQuotaExceededError,
    LLMAuthenticationError,
    LLMServiceUnavailableError
)


class AnalysisService:
    def __init__(self, db: Session):
        self.db = db
    
    async def run_analysis(
        self,
        analysis_id: int,
        request: AnalysisRequest,
        session: UserSession
    ):
        """Run the complete analysis pipeline."""
        analysis = self.db.query(Analysis).filter(Analysis.id == analysis_id).first()
        if not analysis:
            return
        
        try:
            # Update status to processing
            analysis.status = "processing"
            self.db.commit()
            
            # Fetch all data
            evidence = await self._fetch_evidence(request)
            
            # Store evidence
            result = Result(
                analysis_id=analysis_id,
                evidence_json=json.dumps(evidence)
            )
            
            # Run LLM analysis if model specified
            if request.model:
                llm_service = LLMService(
                    provider=session.api_provider,
                    api_key=session.api_key_encrypted  # Will be decrypted in real implementation
                )
                
                try:
                    llm_output = await llm_service.analyze_correlation(
                        evidence, 
                        request.model
                    )
                    
                    result.llm_output_json = json.dumps(llm_output)
                    result.verdict = llm_output.get("verdict", "inconclusive")
                    result.confidence = float(llm_output.get("confidence", 0.0))
                    
                except LLMAuthenticationError as e:
                    result.error_message = f"API Authentication Error: {str(e)}"
                    result.verdict = "error"
                    result.confidence = 0.0
                except LLMQuotaExceededError as e:
                    result.error_message = f"API Quota Exceeded: {str(e)}"
                    result.verdict = "error"  
                    result.confidence = 0.0
                except LLMRateLimitError as e:
                    result.error_message = f"API Rate Limited: {str(e)}"
                    result.verdict = "retry_later"
                    result.confidence = 0.0
                except LLMServiceUnavailableError as e:
                    result.error_message = f"LLM Service Unavailable: {str(e)}"
                    result.verdict = "service_unavailable"
                    result.confidence = 0.0
                except Exception as e:
                    result.error_message = f"Analysis Error: {str(e)}"
                    result.verdict = "error"
                    result.confidence = 0.0
            
            # Save result
            self.db.add(result)
            
            # Update analysis status
            analysis.status = "completed"
            analysis.completed_at = datetime.utcnow()
            
            # Store resolved IDs
            if evidence.get("query"):
                gene_info = evidence["query"].get("gene", {})
                disease_info = evidence["query"].get("disease", {})
                analysis.ensembl_id = gene_info.get("ensembl_id")
                analysis.disease_efo = disease_info.get("efo_id")
                
            self.db.commit()
            
        except Exception as e:
            # Handle failure
            analysis.status = "failed"
            analysis.completed_at = datetime.utcnow()
            
            result = Result(
                analysis_id=analysis_id,
                error_message=str(e),
                verdict="error",
                confidence=0.0
            )
            self.db.add(result)
            self.db.commit()
    
    async def _fetch_evidence(self, request: AnalysisRequest) -> Dict[str, Any]:
        """Fetch evidence from all data sources."""
        async with httpx.AsyncClient() as client:
            # Resolve gene and disease
            ensg, gene_syns = await resolve_gene_symbol(client, request.gene)
            efo_id, mondo_id, dis_syns = await resolve_disease_label(client, request.disease)
            
            # Fetch data from all sources concurrently
            tasks = [
                get_opentargets_association(client, ensg, efo_id),
                get_literature_evidence(
                    client, 
                    gene_syns, 
                    dis_syns, 
                    request.since_year, 
                    request.max_abstracts
                )
            ]
            
            if request.include_gwas:
                tasks.append(
                    get_gwas_associations(
                        client,
                        request.disease if "EFO_" not in request.disease else request.disease,
                        request.gene,
                        max_records=15
                    )
                )
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle results
            ot_assoc = results[0] if not isinstance(results[0], Exception) else None
            literature = results[1] if not isinstance(results[1], Exception) else []
            gwas = []
            if request.include_gwas and len(results) > 2:
                gwas = results[2] if not isinstance(results[2], Exception) else []
            
            return {
                "query": {
                    "gene": {"symbol": request.gene, "ensembl_id": ensg},
                    "disease": {"label": request.disease, "efo_id": efo_id, "mondo_id": mondo_id}
                },
                "synonyms": {"gene": gene_syns, "disease": dis_syns},
                "opentargets": ot_assoc,
                "gwas_catalog": gwas,
                "literature": literature
            }
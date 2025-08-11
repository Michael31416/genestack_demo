from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class APIProvider(str, Enum):
    openai = "openai"
    anthropic = "anthropic"


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=100)
    api_provider: APIProvider
    api_key: str = Field(..., min_length=1)


class LoginResponse(BaseModel):
    session_id: str
    username: str
    message: str


class AnalysisRequest(BaseModel):
    gene: str = Field(..., min_length=1, max_length=50)
    disease: str = Field(..., min_length=1, max_length=200)
    since_year: int = Field(2015, ge=1990, le=2025)
    max_abstracts: int = Field(8, ge=1, le=25)
    include_gwas: bool = Field(True)
    model: Optional[str] = Field("gpt-4o-mini", description="LLM model to use")


class AnalysisStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class AnalysisResponse(BaseModel):
    id: int
    status: AnalysisStatus
    gene_symbol: str
    disease_label: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class AnalysisResult(BaseModel):
    id: int
    status: AnalysisStatus
    gene_symbol: str
    disease_label: str
    ensembl_id: Optional[str] = None
    disease_efo: Optional[str] = None
    evidence: Optional[Dict[str, Any]] = None
    verdict: Optional[str] = None
    confidence: Optional[float] = None
    llm_output: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class HistoryItem(BaseModel):
    id: int
    gene_symbol: str
    disease_label: str
    status: AnalysisStatus
    verdict: Optional[str] = None
    confidence: Optional[float] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


class ErrorResponse(BaseModel):
    detail: str
    error_type: Optional[str] = None
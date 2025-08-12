from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

from ..config import (
    USERNAME_MAX_LENGTH, GENE_MAX_LENGTH, DISEASE_MAX_LENGTH,
    MIN_PUBLICATION_YEAR, MAX_PUBLICATION_YEAR, DEFAULT_SINCE_YEAR,
    DEFAULT_MAX_ABSTRACTS, MIN_ABSTRACTS, MAX_ABSTRACTS, DEFAULT_LLM_MODEL
)


class APIProvider(str, Enum):
    openai = "openai"
    anthropic = "anthropic"


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=USERNAME_MAX_LENGTH)
    api_provider: APIProvider
    api_key: str = Field(..., min_length=1)


class LoginResponse(BaseModel):
    session_id: str
    username: str
    message: str


class AnalysisRequest(BaseModel):
    gene: str = Field(..., min_length=1, max_length=GENE_MAX_LENGTH)
    disease: str = Field(..., min_length=1, max_length=DISEASE_MAX_LENGTH)
    since_year: int = Field(DEFAULT_SINCE_YEAR, ge=MIN_PUBLICATION_YEAR, le=MAX_PUBLICATION_YEAR)
    max_abstracts: int = Field(DEFAULT_MAX_ABSTRACTS, ge=MIN_ABSTRACTS, le=MAX_ABSTRACTS)
    include_gwas: bool = Field(True)
    model: Optional[str] = Field(DEFAULT_LLM_MODEL, description="LLM model to use")


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
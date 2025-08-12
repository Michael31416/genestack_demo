"""
Main FastAPI application for Gene-Disease Analysis Service.
"""

import os
import uuid
import json
import asyncio
from contextlib import asynccontextmanager
from typing import List, Optional
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect, BackgroundTasks, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
import uvicorn

from .models import Base, User, Session as UserSession, Analysis, Result, get_db_engine, get_session_maker
from .schemas import (
    LoginRequest, LoginResponse, 
    AnalysisRequest, AnalysisResponse, AnalysisResult,
    HistoryItem, ErrorResponse
)
from .services.analysis_service import AnalysisService


# Database setup - lazy initialization
_engine = None
_SessionLocal = None


def get_engine():
    """Get or create the database engine."""
    global _engine
    if _engine is None:
        _engine = get_db_engine()
    return _engine


def get_session_local():
    """Get or create the session maker."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = get_session_maker(get_engine())
    return _SessionLocal


def get_db():
    """Dependency to get database session."""
    SessionLocal = get_session_local()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# WebSocket manager for real-time updates
class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[int, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, analysis_id: int):
        await websocket.accept()
        if analysis_id not in self.active_connections:
            self.active_connections[analysis_id] = []
        self.active_connections[analysis_id].append(websocket)
    
    def disconnect(self, websocket: WebSocket, analysis_id: int):
        if analysis_id in self.active_connections:
            self.active_connections[analysis_id].remove(websocket)
            if not self.active_connections[analysis_id]:
                del self.active_connections[analysis_id]
    
    async def send_update(self, analysis_id: int, message: dict):
        if analysis_id in self.active_connections:
            for connection in self.active_connections[analysis_id]:
                try:
                    await connection.send_json(message)
                except:
                    pass


manager = ConnectionManager()

# Create rate limiter
limiter = Limiter(key_func=get_remote_address)

# Create FastAPI app
app = FastAPI(
    title="Gene-Disease Analysis Service",
    description="Analyze correlations between genes and diseases using public data and LLMs",
    version="1.0.0"
)

# Add rate limit error handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# API Routes
@app.post("/api/v1/auth/login", response_model=LoginResponse)
@limiter.limit("10/minute")
async def login(request: Request, login_request: LoginRequest, db: Session = Depends(get_db)):
    """Create or retrieve user session."""
    # Get or create user
    user = db.query(User).filter(User.username == login_request.username).first()
    if not user:
        user = User(username=login_request.username)
        db.add(user)
        db.commit()
        db.refresh(user)
    
    # Create new session
    session_id = str(uuid.uuid4())
    session = UserSession(
        id=session_id,
        user_id=user.id,
        api_provider=login_request.api_provider,
        api_key_encrypted=login_request.api_key  # In production, encrypt this!
    )
    db.add(session)
    db.commit()
    
    return LoginResponse(
        session_id=session_id,
        username=user.username,
        message="Login successful"
    )


@app.post("/api/v1/analyses", response_model=AnalysisResponse)
@limiter.limit("20/minute")
async def create_analysis(
    request: Request,
    analysis_request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    session_id: str,
    db: Session = Depends(get_db)
):
    """Start a new gene-disease analysis."""
    # Validate session
    session = db.query(UserSession).filter(UserSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=401, detail="Invalid session")
    
    # Update session last_used
    session.last_used = datetime.utcnow()
    
    # Create analysis record
    analysis = Analysis(
        user_id=session.user_id,
        session_id=session_id,
        gene_symbol=analysis_request.gene,
        disease_label=analysis_request.disease,
        params_json=json.dumps(analysis_request.dict())
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    
    # Start analysis in background
    background_tasks.add_task(
        run_analysis_task,
        analysis.id,
        analysis_request,
        session.id,
        session.api_provider,
        session.api_key_encrypted
    )
    
    return AnalysisResponse(
        id=analysis.id,
        status="pending",
        gene_symbol=analysis.gene_symbol,
        disease_label=analysis.disease_label,
        created_at=analysis.created_at
    )


async def run_analysis_task(
    analysis_id: int,
    request: AnalysisRequest,
    session_id: str,
    api_provider: str,
    api_key: str
):
    """Background task to run analysis."""
    # Create new database session for background task
    SessionLocal = get_session_local()
    db = SessionLocal()
    service = AnalysisService(db)
    
    # Send WebSocket update - starting
    await manager.send_update(analysis_id, {
        "status": "processing",
        "message": "Starting analysis..."
    })
    
    # Create session object for service
    session = UserSession(
        id=session_id,
        api_provider=api_provider,
        api_key_encrypted=api_key
    )
    
    # Run analysis
    await service.run_analysis(analysis_id, request, session)
    
    # Send WebSocket update - completed
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    result = db.query(Result).filter(Result.analysis_id == analysis_id).first()
    
    update_data = {
        "status": analysis.status,
        "message": "Analysis complete" if analysis.status == "completed" else "Analysis failed"
    }
    
    if result and analysis.status == "completed":
        update_data["verdict"] = result.verdict
        update_data["confidence"] = result.confidence
    
    await manager.send_update(analysis_id, update_data)
    
    # Close database session
    db.close()


@app.get("/api/v1/analyses/{analysis_id}", response_model=AnalysisResult)
async def get_analysis(
    analysis_id: int,
    session_id: str,
    db: Session = Depends(get_db)
):
    """Get analysis results."""
    # Validate session
    session = db.query(UserSession).filter(UserSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=401, detail="Invalid session")
    
    # Get analysis
    analysis = db.query(Analysis).filter(
        Analysis.id == analysis_id,
        Analysis.user_id == session.user_id
    ).first()
    
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    # Get result if available
    result = db.query(Result).filter(Result.analysis_id == analysis_id).first()
    
    response = AnalysisResult(
        id=analysis.id,
        status=analysis.status,
        gene_symbol=analysis.gene_symbol,
        disease_label=analysis.disease_label,
        ensembl_id=analysis.ensembl_id,
        disease_efo=analysis.disease_efo,
        created_at=analysis.created_at,
        completed_at=analysis.completed_at
    )
    
    if result:
        if result.evidence_json:
            response.evidence = json.loads(result.evidence_json)
        if result.llm_output_json:
            response.llm_output = json.loads(result.llm_output_json)
        response.verdict = result.verdict
        response.confidence = result.confidence
        response.error_message = result.error_message
    
    return response


@app.get("/api/v1/analyses", response_model=List[HistoryItem])
async def get_history(
    session_id: str,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Get user's analysis history."""
    # Validate session
    session = db.query(UserSession).filter(UserSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=401, detail="Invalid session")
    
    # Get analyses
    analyses = db.query(Analysis).filter(
        Analysis.user_id == session.user_id
    ).order_by(Analysis.created_at.desc()).limit(limit).all()
    
    history = []
    for analysis in analyses:
        result = db.query(Result).filter(Result.analysis_id == analysis.id).first()
        
        item = HistoryItem(
            id=analysis.id,
            gene_symbol=analysis.gene_symbol,
            disease_label=analysis.disease_label,
            status=analysis.status,
            created_at=analysis.created_at
        )
        
        if result:
            item.verdict = result.verdict
            item.confidence = result.confidence
        
        history.append(item)
    
    return history


@app.websocket("/api/v1/ws/{analysis_id}")
async def websocket_endpoint(websocket: WebSocket, analysis_id: int):
    """WebSocket endpoint for real-time analysis updates."""
    await manager.connect(websocket, analysis_id)
    try:
        while True:
            # Keep connection alive
            await asyncio.sleep(1)
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, analysis_id)


# Serve static files
static_path = Path(__file__).parent.parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Serve index.html at root
@app.get("/")
async def read_index():
    index_path = static_path / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Gene-Disease Analysis Service API"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
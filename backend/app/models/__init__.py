from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    sessions = relationship("Session", back_populates="user")
    analyses = relationship("Analysis", back_populates="user")


class Session(Base):
    __tablename__ = "sessions"
    
    id = Column(String, primary_key=True, index=True)  # UUID
    user_id = Column(Integer, ForeignKey("users.id"))
    api_provider = Column(String)  # "openai" or "anthropic"
    api_key_encrypted = Column(Text)  # We'll store encrypted
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="sessions")


class Analysis(Base):
    __tablename__ = "analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    session_id = Column(String, ForeignKey("sessions.id"))
    gene_symbol = Column(String, index=True)
    disease_label = Column(String, index=True)
    ensembl_id = Column(String)
    disease_efo = Column(String)
    disease_mondo = Column(String)
    params_json = Column(Text)
    status = Column(String, default="pending")  # pending, processing, completed, failed
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    
    user = relationship("User", back_populates="analyses")
    result = relationship("Result", back_populates="analysis", uselist=False)


class Result(Base):
    __tablename__ = "results"
    
    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(Integer, ForeignKey("analyses.id"), unique=True)
    evidence_json = Column(Text)
    llm_output_json = Column(Text)
    verdict = Column(String)
    confidence = Column(Float)
    error_message = Column(Text, nullable=True)
    
    analysis = relationship("Analysis", back_populates="result")


def get_db_engine(db_path: str = "database/gene_disease.db"):
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    return engine


def get_session_maker(engine):
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)
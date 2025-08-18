from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker, Session

Base = declarative_base()

class Interaction(Base):
    """Represents a chat interaction with the system."""
    __tablename__ = 'interactions'

    id = Column(Integer, primary_key=True)
    query = Column(String, nullable=False)
    response = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.now)

    # Judge metrics
    summary = Column(String, nullable=True)
    verdict = Column(String, nullable=True)
    query_understanding_score = Column(Float, nullable=True)
    retrieval_quality = Column(Float, nullable=True)
    generation_quality = Column(Float, nullable=True)
    error_detection_score = Column(Float, nullable=True)
    citation_integrity = Column(Float, nullable=True)
    hallucination_index = Column(Float, nullable=True)

    # User feedback
    # user_rating = Column(Integer, nullable=True)  # +1 and -1

    # Store context info
    used_documents = Column(JSON, nullable=True)  # Store zotero keys of used documents
    processed_query = Column(String, nullable=True)  # Store rewritten query

    # Store strengths and weaknesses
    strengths = Column(JSON, nullable=True)
    weaknesses = Column(JSON, nullable=True)

def init_db(db_path: str = "sqlite:///metrics.db"):
    """Initialize the database and create tables."""
    engine = create_engine(db_path)
    Base.metadata.create_all(engine)
    return engine

def get_session(engine) -> Session:
    """Create a new database session."""
    Session = sessionmaker(bind=engine)
    return Session()

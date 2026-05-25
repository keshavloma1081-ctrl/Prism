"""
PRISM — db/database.py
Database Connection + Session Management

Supports:
  - PostgreSQL (production)
  - SQLite    (development / testing)

Set DATABASE_URL in .env:
  PostgreSQL: postgresql://user:password@localhost:5432/prism
  SQLite:     sqlite:///./prism.db
"""

from __future__ import annotations
import os
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from db.models import Base

# ── Database URL ──────────────────────────────────────────────────────────────

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "sqlite:///./prism.db"  # SQLite default for development
)

# ── Engine ────────────────────────────────────────────────────────────────────

def create_db_engine(database_url: str = DATABASE_URL):
    """Create SQLAlchemy engine with appropriate settings."""

    if database_url.startswith("sqlite"):
        engine = create_engine(
            database_url,
            connect_args = {"check_same_thread": False},
            poolclass    = StaticPool,
            echo         = False
        )
        # Enable WAL mode for SQLite concurrency
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    else:
        engine = create_engine(
            database_url,
            pool_size    = 10,
            max_overflow = 20,
            pool_timeout = 30,
            pool_recycle = 1800,
            echo         = False
        )

    return engine


engine       = create_db_engine()
SessionLocal = sessionmaker(
    autocommit = False,
    autoflush  = False,
    bind       = engine
)


def init_db() -> None:
    """Create all tables. Safe to call multiple times."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """
    FastAPI dependency — yields a database session.
    Automatically closes on request completion.

    Usage in FastAPI:
        @app.get("/endpoint")
        def endpoint(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
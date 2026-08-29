"""
Database connection setup for SmartETA.

Defaults to a local SQLite file for development. Set the DATABASE_URL
environment variable to point at PostgreSQL in production/Azure, e.g.:

    export DATABASE_URL="postgresql+psycopg2://user:password@host:5432/smarteta"

No code changes needed to switch — just set the env var.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///smarteta.db")

# SQLite needs this connect_arg for multi-threaded Streamlit access; Postgres doesn't.
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


def get_session():
    """Return a new DB session. Caller is responsible for closing it."""
    return SessionLocal()


def init_db():
    """Create all tables if they don't already exist."""
    from db import models  # noqa: F401 (ensures models are registered on Base)
    Base.metadata.create_all(bind=engine)

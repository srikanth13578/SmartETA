"""
Database connection setup for SmartETA.

Reads DATABASE_URL from a .env file in the project root (or from an
actual exported env var, which takes priority). Falls back to a local
SQLite file only if neither is set.
"""

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

load_dotenv()  # reads .env in the current working directory, if present

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
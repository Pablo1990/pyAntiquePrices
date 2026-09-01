"""Database connection and session management."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker

from .models import Base


def get_engine(database_url: str = "sqlite:///./data/antiquegpt.db"):
    _ensure_sqlite_parent_dir(database_url)
    connect_args = {"check_same_thread": False} if "sqlite" in database_url else {}
    return create_engine(database_url, connect_args=connect_args)


def _ensure_sqlite_parent_dir(database_url: str) -> None:
    url = make_url(database_url)
    if url.drivername != "sqlite":
        return
    database = url.database
    if not database or database == ":memory:":
        return
    Path(database).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def create_tables(engine) -> None:
    Base.metadata.create_all(engine)


def get_session_factory(engine):
    return sessionmaker(bind=engine)

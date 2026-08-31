"""Database connection and session management."""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .models import Base


def get_engine(database_url: str = "sqlite:///./data/antiquegpt.db"):
    connect_args = {"check_same_thread": False} if "sqlite" in database_url else {}
    return create_engine(database_url, connect_args=connect_args)


def create_tables(engine) -> None:
    Base.metadata.create_all(engine)


def get_session_factory(engine):
    return sessionmaker(bind=engine)

from __future__ import annotations

import os
import logging
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker
from dotenv import load_dotenv


load_dotenv()

logger = logging.getLogger(__name__)


def _default_database_url() -> str:
    return "sqlite:///./voice_os_phase5.db"


DATABASE_URL = (
    (os.getenv("POSTGRES_URL") or "").strip()
    or (os.getenv("DATABASE_URL") or "").strip()
    or _default_database_url()
)

Base = declarative_base()


def _engine_kwargs() -> dict:  # type: ignore[type-arg]
    is_sqlite = DATABASE_URL.startswith("sqlite")
    kwargs: dict = {
        "future": True,
        "pool_pre_ping": True,
    }
    if not is_sqlite:
        kwargs.update(
            {
                "pool_size": max(5, int((os.getenv("DB_POOL_SIZE") or "20").strip() or "20")),
                "max_overflow": max(0, int((os.getenv("DB_MAX_OVERFLOW") or "40").strip() or "40")),
                "pool_timeout": max(1, int((os.getenv("DB_POOL_TIMEOUT_SECONDS") or "30").strip() or "30")),
                "pool_recycle": max(60, int((os.getenv("DB_POOL_RECYCLE_SECONDS") or "1800").strip() or "1800")),
            }
        )
    return kwargs


engine = create_engine(DATABASE_URL, **_engine_kwargs())
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def init_db() -> None:
    # Import models so SQLAlchemy metadata is populated before create_all.
    from backend.infrastructure.database import db_models  # noqa: F401

    try:
        Base.metadata.create_all(bind=engine)
        logger.warning("DB connected")
    except Exception:
        logger.exception("DB failed")
        raise


@contextmanager
def db_session_scope() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

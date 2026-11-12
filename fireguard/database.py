from __future__ import annotations

from sqlmodel import Session, create_engine

from fireguard import models
from fireguard.config import settings

engine = create_engine(settings.database_url, echo=False)


def get_session() -> Session:
    return Session(engine)


def init_db() -> None:
    models.SQLModel.metadata.create_all(engine)

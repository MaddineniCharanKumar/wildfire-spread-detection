from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from sqlmodel import Session

from fireguard.models import Alert


class AlertService:
    def __init__(self, session: Session):
        self.session = session

    def create_alert(self, event_id: str, title: str, message: str, level: str = "MEDIUM", reason: str = "") -> Alert:
        alert = Alert(
            alert_id=f"alert-{uuid4().hex[:10]}",
            event_id=event_id,
            level=level,
            title=title,
            message=message,
            reason=reason,
            acknowledged=False,
            created_at=datetime.utcnow(),
        )
        self.session.add(alert)
        self.session.commit()
        self.session.refresh(alert)
        return alert

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class NotificationService:
    providers: list[str] = field(default_factory=lambda: ["EMAIL", "SMS", "WEBHOOK"])

    def send(self, channel: str, message: dict[str, Any]) -> dict[str, Any]:
        return {
            "provider": channel,
            "status": "queued",
            "payload": message,
        }


def send_all_notifications(message: dict[str, Any]):
    service = NotificationService()
    return [service.send(provider.lower(), message) for provider in service.providers]

from __future__ import annotations

import asyncio
from typing import Any, List


class WebSocketManager:
    """Simple live-stream broadcaster for detections, alerts and metrics."""

    def __init__(self):
        self.connections: List[Any] = []

    async def connect(self, websocket: Any) -> None:
        self.connections.append(websocket)
        await websocket.send_json({"type": "connected", "status": "ok"})

    async def broadcast(self, payload: dict[str, Any]) -> None:
        for ws in list(self.connections):
            try:
                await ws.send_json(payload)
            except Exception:
                self.connections.remove(ws)


ws_manager = WebSocketManager()

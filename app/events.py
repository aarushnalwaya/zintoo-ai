"""
Real-time event bus.

The previous dashboard faked "real-time" with a client-side setInterval cycling
through 7 hard-coded strings. This replaces that with a genuine server-driven
stream: domain actions (orchestration runs, transfers, forecasts, logins,
stock alerts) publish structured events that fan out to every connected client
over Server-Sent Events (SSE) and WebSocket.

SSE is the primary transport — it survives Render's proxy, needs no sticky
sessions, and auto-reconnects in the browser. WebSocket is offered too for
clients that want bidirectional use.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any

from .observability import get_logger, metrics

log = get_logger("zintoo.events")


@dataclass
class Event:
    type: str
    data: dict[str, Any]
    ts: float = field(default_factory=time.time)
    id: int = 0

    def to_sse(self) -> str:
        payload = {"type": self.type, "ts": self.ts, "data": self.data}
        return f"id: {self.id}\nevent: {self.type}\ndata: {json.dumps(payload, default=str)}\n\n"

    def to_json(self) -> str:
        return json.dumps(
            {"id": self.id, "type": self.type, "ts": self.ts, "data": self.data},
            default=str,
        )


class EventBus:
    def __init__(self, history: int = 50) -> None:
        self._subscribers: set[asyncio.Queue] = set()
        self._lock = asyncio.Lock()
        self._counter = 0
        self._history: list[Event] = []
        self._history_max = history

    async def publish(self, type_: str, data: dict[str, Any]) -> Event:
        self._counter += 1
        event = Event(type=type_, data=data, id=self._counter)
        self._history.append(event)
        if len(self._history) > self._history_max:
            self._history.pop(0)
        async with self._lock:
            subscribers = list(self._subscribers)
        for queue in subscribers:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # Slow consumer — drop the event for that client, don't block others.
                metrics.inc("events_dropped_total")
        metrics.inc("events_published_total")
        metrics.gauge("event_subscribers", len(subscribers))
        return event

    def publish_soon(self, type_: str, data: dict[str, Any]) -> None:
        """Fire-and-forget from sync code running inside the event loop."""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.publish(type_, data))
        except RuntimeError:
            # No running loop (e.g. during seeding) — safe to ignore.
            pass

    async def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        async with self._lock:
            self._subscribers.add(queue)
        metrics.gauge("event_subscribers", len(self._subscribers))
        return queue

    async def unsubscribe(self, queue: asyncio.Queue) -> None:
        async with self._lock:
            self._subscribers.discard(queue)
        metrics.gauge("event_subscribers", len(self._subscribers))

    def recent(self, limit: int = 20) -> list[Event]:
        return self._history[-limit:]

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)


bus = EventBus()


async def sse_stream(request, queue: asyncio.Queue):
    """Yield SSE frames for one subscriber, with heartbeats and backfill."""
    # Backfill recent history so a fresh client isn't staring at a blank feed.
    for event in bus.recent(10):
        yield event.to_sse()
    yield ": connected\n\n"
    try:
        while True:
            if await request.is_disconnected():
                break
            try:
                event = await asyncio.wait_for(queue.get(), timeout=15.0)
                yield event.to_sse()
            except asyncio.TimeoutError:
                # Heartbeat keeps intermediaries (and Render) from closing idle conns.
                yield ": heartbeat\n\n"
    finally:
        await bus.unsubscribe(queue)

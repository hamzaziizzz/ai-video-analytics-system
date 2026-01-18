import threading
from collections import deque
from typing import Deque, Dict, List

from .people_count import PeopleCountEvent


class EventStore:
    def __init__(self, max_events: int = 1000) -> None:
        self._events: Deque[PeopleCountEvent] = deque(maxlen=max_events)
        self._lock = threading.Lock()

    def add(self, event: PeopleCountEvent) -> None:
        with self._lock:
            self._events.append(event)

    def list(self, limit: int = 100) -> List[Dict[str, object]]:
        with self._lock:
            items = list(self._events)[-limit:]
        return [event.__dict__ for event in items]

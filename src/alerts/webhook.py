import json
import queue
import threading
import urllib.request
from typing import Optional

from ..events.people_count import PeopleCountEvent
from ..utils.logging import get_logger


class WebhookAlertDispatcher:
    def __init__(self, webhooks, timeout_seconds: float) -> None:
        self.webhooks = webhooks
        self.timeout_seconds = timeout_seconds
        self._queue: queue.Queue = queue.Queue(maxsize=500)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.logger = get_logger("alerts.webhook")

    def start(self) -> None:
        if not self.webhooks:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)

    def enqueue(self, event: PeopleCountEvent) -> None:
        if not self.webhooks:
            return
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            self.logger.warning("Alert queue full; dropping event")

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                event = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            payload = json.dumps(event.__dict__).encode("utf-8")
            for url in self.webhooks:
                try:
                    req = urllib.request.Request(
                        url,
                        data=payload,
                        headers={"Content-Type": "application/json"},
                    )
                    with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                        resp.read()
                except Exception as exc:
                    self.logger.warning("Webhook failed: %s", exc)

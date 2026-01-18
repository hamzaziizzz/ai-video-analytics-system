from dataclasses import dataclass
from typing import Optional


@dataclass
class PeopleCountEvent:
    camera_id: str
    timestamp_ms: int
    count: int
    event_type: str
    zone: Optional[str] = None


class PeopleCountEngine:
    def __init__(
        self,
        camera_id: str,
        min_count: int,
        stable_frames: int,
        cooldown_seconds: float,
        report_interval_seconds: float,
        zone: Optional[str] = None,
    ) -> None:
        self.camera_id = camera_id
        self.min_count = min_count
        self.stable_frames = max(stable_frames, 1)
        self.cooldown_seconds = cooldown_seconds
        self.report_interval_seconds = report_interval_seconds
        self.zone = zone
        self._stable_count = 0
        self._last_trigger_ts: Optional[float] = None
        self._last_report_ts: Optional[float] = None
        self._last_seen_count = 0

    def update(self, count: int, timestamp_ms: int) -> Optional[PeopleCountEvent]:
        self._last_seen_count = count
        if count >= self.min_count:
            self._stable_count += 1
        else:
            self._stable_count = 0

        ts_sec = timestamp_ms / 1000.0

        if self._stable_count >= self.stable_frames and (
            self._last_trigger_ts is None
            or (ts_sec - self._last_trigger_ts) >= self.cooldown_seconds
        ):
            self._last_trigger_ts = ts_sec
            return PeopleCountEvent(
                camera_id=self.camera_id,
                timestamp_ms=timestamp_ms,
                count=count,
                event_type="people_count_alert",
                zone=self.zone,
            )

        if self._last_report_ts is None:
            self._last_report_ts = ts_sec
            return None

        if (ts_sec - self._last_report_ts) >= self.report_interval_seconds:
            self._last_report_ts = ts_sec
            return PeopleCountEvent(
                camera_id=self.camera_id,
                timestamp_ms=timestamp_ms,
                count=count,
                event_type="people_count_report",
                zone=self.zone,
            )

        return None

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Detection:
    bbox: List[float]
    score: float
    class_id: int
    class_name: str
    keypoints: Optional[List[List[float]]] = None
    track_id: Optional[int] = None


class InferenceEngine:
    def load(self) -> None:
        raise NotImplementedError

    def infer(self, frame) -> List[Detection]:
        raise NotImplementedError

    def infer_batch(self, frames) -> List[List[Detection]]:
        return [self.infer(frame) for frame in frames]

    def close(self) -> None:
        pass

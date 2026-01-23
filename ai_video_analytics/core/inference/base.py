from dataclasses import dataclass
from typing import List


@dataclass
class Detection:
    bbox: List[float]
    score: float
    class_id: int
    class_name: str


class InferenceEngine:
    def load(self) -> None:
        raise NotImplementedError

    def infer(self, frame) -> List[Detection]:
        raise NotImplementedError

    def infer_batch(self, frames) -> List[List[Detection]]:
        return [self.infer(frame) for frame in frames]

    def close(self) -> None:
        pass

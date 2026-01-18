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

    def close(self) -> None:
        pass

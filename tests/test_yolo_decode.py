import numpy as np
import pytest

from src.inference.yolo import PreprocessMeta, decode_yolo_output


def test_decode_end_to_end_output():
    output = np.array(
        [[[10, 20, 50, 60, 0.9, 0], [15, 25, 55, 65, 0.2, 0]]],
        dtype=np.float32,
    )
    meta = PreprocessMeta(scale=1.0, pad_x=0.0, pad_y=0.0, orig_shape=(100, 200))

    detections = decode_yolo_output(
        output,
        labels=["person"],
        confidence_threshold=0.5,
        nms_iou_threshold=0.5,
        has_objectness=False,
        meta=meta,
        end_to_end=True,
    )

    assert len(detections) == 1
    det = detections[0]
    assert det.class_name == "person"
    assert det.score == pytest.approx(0.9)
    assert det.bbox == [10.0, 20.0, 50.0, 60.0]

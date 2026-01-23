from .onnx_engine import OnnxYoloEngine
from .openvino_engine import OpenVINOYoloEngine
from .tensorrt_engine import TensorRTYoloEngine
from .ultralytics_engine import UltralyticsYoloEngine
from ..utils.config import InferenceConfig


def create_engine(config: InferenceConfig):
    """Create a configured inference engine instance."""
    engine_name = config.engine.lower()
    if engine_name == "onnx":
        return OnnxYoloEngine(
            model_path=config.model_path,
            labels=config.labels,
            input_size=config.input_size,
            confidence_threshold=config.confidence_threshold,
            nms_iou_threshold=config.nms_iou_threshold,
            has_objectness=config.has_objectness,
            end_to_end=config.end_to_end,
            log_severity_level=config.onnx_log_severity_level,
            device=config.device,
            fp16=config.fp16,
            int8=config.int8,
            class_id_filter=config.class_id_filter,
            debug_log_raw_output=config.debug_log_raw_output,
            debug_log_raw_interval_seconds=config.debug_log_raw_interval_seconds,
            debug_log_raw_rows=config.debug_log_raw_rows,
            debug_log_raw_cols=config.debug_log_raw_cols,
        )
    if engine_name in {"trt", "tensorrt"}:
        return TensorRTYoloEngine(
            engine_path=config.model_path,
            labels=config.labels,
            input_size=config.input_size,
            confidence_threshold=config.confidence_threshold,
            nms_iou_threshold=config.nms_iou_threshold,
            has_objectness=config.has_objectness,
            end_to_end=config.end_to_end,
            device=config.device,
            fp16=config.fp16,
            int8=config.int8,
            class_id_filter=config.class_id_filter,
            debug_log_raw_output=config.debug_log_raw_output,
            debug_log_raw_interval_seconds=config.debug_log_raw_interval_seconds,
            debug_log_raw_rows=config.debug_log_raw_rows,
            debug_log_raw_cols=config.debug_log_raw_cols,
        )
    if config.algorithm.lower() == "yolo":
        return UltralyticsYoloEngine(
            model_path=config.model_path,
            labels=config.labels,
            input_size=config.input_size,
            confidence_threshold=config.confidence_threshold,
            nms_iou_threshold=config.nms_iou_threshold,
            device=config.device,
            fp16=config.fp16,
            int8=config.int8,
            class_id_filter=config.class_id_filter,
            batch_size=config.batch_size,
        )
    if engine_name == "openvino":
        return OpenVINOYoloEngine(
            model_path=config.model_path,
            labels=config.labels,
            input_size=config.input_size,
            confidence_threshold=config.confidence_threshold,
            nms_iou_threshold=config.nms_iou_threshold,
            has_objectness=config.has_objectness,
            end_to_end=config.end_to_end,
            device=config.device,
            fp16=config.fp16,
            int8=config.int8,
            class_id_filter=config.class_id_filter,
            debug_log_raw_output=config.debug_log_raw_output,
            debug_log_raw_interval_seconds=config.debug_log_raw_interval_seconds,
            debug_log_raw_rows=config.debug_log_raw_rows,
            debug_log_raw_cols=config.debug_log_raw_cols,
        )
    if engine_name in {"pt", "pytorch", "ultralytics"}:
        return UltralyticsYoloEngine(
            model_path=config.model_path,
            labels=config.labels,
            input_size=config.input_size,
            confidence_threshold=config.confidence_threshold,
            nms_iou_threshold=config.nms_iou_threshold,
            device=config.device,
            fp16=config.fp16,
            int8=config.int8,
            class_id_filter=config.class_id_filter,
            batch_size=config.batch_size,
        )
    raise ValueError(f"Unsupported inference engine: {config.engine}")

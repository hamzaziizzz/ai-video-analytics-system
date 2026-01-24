from .onnx_engine import OnnxYoloEngine
from .openvino_engine import OpenVINOYoloEngine
from .tensorrt_engine import TensorRTYoloEngine
from .ultralytics_engine import UltralyticsYoloEngine
from .ultralytics_pose_engine import UltralyticsPoseEngine
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
            use_cupy_nms=config.use_cupy_nms,
        )
    if engine_name in {"trt", "tensorrt"}:
        trt_impl = (config.trt_implementation or "custom").strip().lower()
        if trt_impl in {"ultralytics", "ultra", "yolo"}:
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
            use_cupy_nms=config.use_cupy_nms,
            use_gpu_preproc=config.trt_gpu_preproc,
            use_numba_decode=config.trt_numba_decode,
            dynamic_shapes=config.trt_dynamic_shapes,
            dynamic_min_size=config.trt_dynamic_min_size,
            dynamic_max_size=config.trt_dynamic_max_size,
            dynamic_stride=config.trt_dynamic_stride,
            no_letterbox=config.trt_no_letterbox,
            gpu_timing=config.trt_gpu_timing,
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


def create_pose_engine(config: InferenceConfig):
    """Create a configured pose inference engine instance."""
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
            return_keypoints=True,
        )
    if engine_name in {"trt", "tensorrt"}:
        trt_impl = (config.trt_implementation or "custom").strip().lower()
        if trt_impl in {"ultralytics", "ultra", "yolo"}:
            return UltralyticsPoseEngine(
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
            use_cupy_nms=config.use_cupy_nms,
            use_gpu_preproc=config.trt_gpu_preproc,
            use_numba_decode=config.trt_numba_decode,
            dynamic_shapes=config.trt_dynamic_shapes,
            dynamic_min_size=config.trt_dynamic_min_size,
            dynamic_max_size=config.trt_dynamic_max_size,
            dynamic_stride=config.trt_dynamic_stride,
            no_letterbox=config.trt_no_letterbox,
            gpu_timing=config.trt_gpu_timing,
            return_keypoints=True,
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
            return_keypoints=True,
        )
    if engine_name in {"ultralytics", "pt", "pytorch"}:
        return UltralyticsPoseEngine(
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
    raise ValueError(f"Unsupported pose inference engine: {config.engine}")

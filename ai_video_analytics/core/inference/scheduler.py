import queue
import threading
import time
from concurrent.futures import Future, TimeoutError
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from ..utils.logging import get_logger


@dataclass
class InferenceTask:
    camera_id: str
    frame: object
    timestamp_ms: int
    future: Future


@dataclass
class WorkerStats:
    batches: int = 0
    frames: int = 0
    total_infer_ms: float = 0.0
    last_infer_ms: float = 0.0
    last_batch_size: int = 0


class InferenceScheduler:
    def __init__(
        self,
        engine_factory: Callable[[], object],
        num_workers: int,
        queue_size: int,
        batch_size: int,
        max_batch_wait_ms: float,
        request_timeout_seconds: float,
        input_size: Optional[tuple[int, int]] = None,
        warmup_iters: int = 0,
        warmup_batch_size: int = 1,
    ) -> None:
        self.engine_factory = engine_factory
        self.num_workers = max(1, num_workers)
        self.queue_size = max(1, queue_size)
        self.batch_size = max(1, batch_size)
        self.max_batch_wait_ms = max(0.0, max_batch_wait_ms)
        self.request_timeout_seconds = max(0.1, request_timeout_seconds)
        self.input_size = input_size
        self.warmup_iters = max(0, warmup_iters)
        self.warmup_batch_size = max(1, warmup_batch_size)
        self._queue: queue.Queue = queue.Queue(maxsize=self.queue_size)
        self._stop_event = threading.Event()
        self._threads: List[threading.Thread] = []
        self.logger = get_logger("inference.scheduler")
        self._stats_lock = threading.Lock()
        self._frames_since_last = 0
        self._last_stats_ts = time.time()
        self._start_ts = time.time()
        self._total_frames = 0
        self._total_batches = 0
        self._total_infer_ms = 0.0
        self._worker_stats: dict[int, WorkerStats] = {}

    def start(self) -> None:
        for idx in range(self.num_workers):
            self._worker_stats[idx] = WorkerStats()
            thread = threading.Thread(target=self._worker_loop, args=(idx,), daemon=True)
            thread.start()
            self._threads.append(thread)
        self.logger.info("Inference scheduler started with %d workers", self.num_workers)

    def stop(self) -> None:
        self._stop_event.set()
        for thread in self._threads:
            thread.join(timeout=2)
        self._threads.clear()

    def infer(self, camera_id: str, frame, timestamp_ms: int) -> List:
        task = InferenceTask(
            camera_id=camera_id,
            frame=frame,
            timestamp_ms=timestamp_ms,
            future=Future(),
        )
        self._enqueue(task)
        try:
            return task.future.result(timeout=self.request_timeout_seconds)
        except TimeoutError:
            task.future.cancel()
            self.logger.warning("Inference timeout for camera_id=%s", camera_id)
            return []
        except Exception as exc:
            self.logger.warning("Inference failed for camera_id=%s: %s", camera_id, exc)
            return []

    def sample_fps(self) -> Optional[tuple[float, int]]:
        now = time.time()
        with self._stats_lock:
            elapsed = now - self._last_stats_ts
            if elapsed <= 0:
                return None
            fps = self._frames_since_last / elapsed
            self._frames_since_last = 0
            self._last_stats_ts = now
        return fps, self._queue.qsize()

    def get_stats(self) -> dict:
        now = time.time()
        with self._stats_lock:
            elapsed = max(0.001, now - self._start_ts)
            avg_fps = self._total_frames / elapsed
            workers = []
            for worker_id, stats in self._worker_stats.items():
                avg_batch_ms = stats.total_infer_ms / stats.batches if stats.batches else 0.0
                avg_frame_ms = stats.total_infer_ms / stats.frames if stats.frames else 0.0
                workers.append(
                    {
                        "worker_id": worker_id,
                        "batches": stats.batches,
                        "frames": stats.frames,
                        "last_batch_size": stats.last_batch_size,
                        "last_infer_ms": stats.last_infer_ms,
                        "avg_batch_ms": avg_batch_ms,
                        "avg_frame_ms": avg_frame_ms,
                    }
                )
            return {
                "workers": self.num_workers,
                "batch_size": self.batch_size,
                "max_batch_wait_ms": self.max_batch_wait_ms,
                "queue_size": self.queue_size,
                "queue_depth": self._queue.qsize(),
                "request_timeout_seconds": self.request_timeout_seconds,
                "total_frames": self._total_frames,
                "total_batches": self._total_batches,
                "total_infer_ms": self._total_infer_ms,
                "avg_throughput_fps": avg_fps,
                "workers_stats": workers,
            }

    def _enqueue(self, task: InferenceTask) -> None:
        try:
            self._queue.put_nowait(task)
        except queue.Full:
            try:
                dropped = self._queue.get_nowait()
                if isinstance(dropped, InferenceTask):
                    if not dropped.future.done():
                        dropped.future.set_exception(RuntimeError("Dropped by scheduler"))
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(task)
            except queue.Full:
                if not task.future.done():
                    task.future.set_exception(RuntimeError("Scheduler queue full"))

    def _worker_loop(self, worker_id: int) -> None:
        engine = self.engine_factory()
        try:
            engine.load()
        except Exception as exc:
            self.logger.error("Failed to load inference engine (worker=%d): %s", worker_id, exc)
            return
        if self.input_size and self.warmup_iters > 0:
            from .warmup import warmup_engine

            warmup_batch = min(self.warmup_batch_size, self.batch_size)
            max_batch = getattr(engine, "max_batch_size", warmup_batch)
            warmup_batch = min(warmup_batch, max_batch)
            warmup_engine(
                engine,
                self.input_size,
                warmup_batch,
                self.warmup_iters,
                logger_name=f"inference.warmup.worker{worker_id}",
            )

        try:
            while not self._stop_event.is_set():
                try:
                    task = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                tasks = [task]
                if self.batch_size > 1:
                    deadline = time.time() + (self.max_batch_wait_ms / 1000.0)
                    while len(tasks) < self.batch_size:
                        timeout = deadline - time.time()
                        if timeout <= 0:
                            break
                        try:
                            next_task = self._queue.get(timeout=timeout)
                        except queue.Empty:
                            break
                        tasks.append(next_task)

                frames = [entry.frame for entry in tasks]
                t0 = time.perf_counter()
                try:
                    results = engine.infer_batch(frames)
                except Exception as exc:
                    for entry in tasks:
                        if not entry.future.done():
                            entry.future.set_exception(exc)
                    continue
                t1 = time.perf_counter()

                for entry, result in zip(tasks, results):
                    if not entry.future.done():
                        entry.future.set_result(result)

                batch_size = len(tasks)
                infer_ms = (t1 - t0) * 1000.0
                with self._stats_lock:
                    self._frames_since_last += len(tasks)
                    self._total_frames += batch_size
                    self._total_batches += 1
                    self._total_infer_ms += infer_ms
                    stats = self._worker_stats.get(worker_id)
                    if stats:
                        stats.batches += 1
                        stats.frames += batch_size
                        stats.total_infer_ms += infer_ms
                        stats.last_infer_ms = infer_ms
                        stats.last_batch_size = batch_size

                for _ in tasks:
                    self._queue.task_done()
        finally:
            if hasattr(engine, "close"):
                engine.close()

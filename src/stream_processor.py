"""
Video stream processing module for RTSP streams.
Handles multi-stream ingestion with threading support.
"""
import cv2
import time
import threading
from queue import Queue, Empty
from typing import Optional, Callable
from loguru import logger


class RTSPStream:
    """
    RTSP Stream handler with frame buffering.
    Supports reconnection and error handling.
    """
    
    def __init__(self, stream_id: str, rtsp_url: str, frame_callback: Optional[Callable] = None):
        """
        Initialize RTSP stream.
        
        Args:
            stream_id: Unique identifier for the stream
            rtsp_url: RTSP URL of the camera
            frame_callback: Optional callback function for frame processing
        """
        self.stream_id = stream_id
        self.rtsp_url = rtsp_url
        self.frame_callback = frame_callback
        
        self.capture = None
        self.running = False
        self.thread = None
        self.frame_queue = Queue(maxsize=10)
        
        self.reconnect_delay = 5  # seconds
        self.max_reconnect_attempts = 10
        
    def start(self):
        """Start the video stream in a separate thread."""
        if self.running:
            logger.warning(f"Stream {self.stream_id} is already running")
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._capture_frames, daemon=True)
        self.thread.start()
        logger.info(f"Started stream {self.stream_id}")
        
    def stop(self):
        """Stop the video stream."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        if self.capture:
            self.capture.release()
        logger.info(f"Stopped stream {self.stream_id}")
        
    def _capture_frames(self):
        """Internal method to capture frames from RTSP stream."""
        reconnect_attempts = 0
        
        while self.running:
            try:
                # Connect to stream
                self.capture = cv2.VideoCapture(self.rtsp_url)
                
                if not self.capture.isOpened():
                    logger.error(f"Failed to open stream {self.stream_id}")
                    reconnect_attempts += 1
                    
                    if reconnect_attempts >= self.max_reconnect_attempts:
                        logger.error(f"Max reconnect attempts reached for {self.stream_id}")
                        break
                        
                    time.sleep(self.reconnect_delay)
                    continue
                
                # Reset reconnect counter on successful connection
                reconnect_attempts = 0
                logger.info(f"Connected to stream {self.stream_id}")
                
                # Read frames
                while self.running:
                    ret, frame = self.capture.read()
                    
                    if not ret:
                        logger.warning(f"Failed to read frame from {self.stream_id}")
                        break
                    
                    # Add frame to queue (non-blocking)
                    try:
                        self.frame_queue.put_nowait(frame)
                    except:
                        # Queue is full, remove oldest frame and add new one
                        try:
                            self.frame_queue.get_nowait()
                            self.frame_queue.put_nowait(frame)
                        except:
                            pass
                    
                    # Call callback if provided
                    if self.frame_callback:
                        self.frame_callback(self.stream_id, frame)
                        
            except Exception as e:
                logger.error(f"Error in stream {self.stream_id}: {e}")
                
            finally:
                if self.capture:
                    self.capture.release()
                    
                if self.running:
                    logger.info(f"Reconnecting to {self.stream_id} in {self.reconnect_delay}s...")
                    time.sleep(self.reconnect_delay)
    
    def get_frame(self, timeout: float = 1.0) -> Optional[cv2.Mat]:
        """
        Get the latest frame from the queue.
        
        Args:
            timeout: Maximum time to wait for a frame
            
        Returns:
            Frame if available, None otherwise
        """
        try:
            return self.frame_queue.get(timeout=timeout)
        except Empty:
            return None


class StreamManager:
    """
    Manages multiple RTSP streams.
    """
    
    def __init__(self):
        """Initialize stream manager."""
        self.streams = {}
        
    def add_stream(self, stream_id: str, rtsp_url: str, frame_callback: Optional[Callable] = None):
        """
        Add a new stream to the manager.
        
        Args:
            stream_id: Unique identifier for the stream
            rtsp_url: RTSP URL of the camera
            frame_callback: Optional callback for frame processing
        """
        if stream_id in self.streams:
            logger.warning(f"Stream {stream_id} already exists")
            return
            
        stream = RTSPStream(stream_id, rtsp_url, frame_callback)
        self.streams[stream_id] = stream
        logger.info(f"Added stream {stream_id}")
        
    def start_stream(self, stream_id: str):
        """Start a specific stream."""
        if stream_id in self.streams:
            self.streams[stream_id].start()
        else:
            logger.error(f"Stream {stream_id} not found")
            
    def stop_stream(self, stream_id: str):
        """Stop a specific stream."""
        if stream_id in self.streams:
            self.streams[stream_id].stop()
        else:
            logger.error(f"Stream {stream_id} not found")
            
    def start_all(self):
        """Start all streams."""
        for stream_id in self.streams:
            self.start_stream(stream_id)
            
    def stop_all(self):
        """Stop all streams."""
        for stream_id in self.streams:
            self.stop_stream(stream_id)
            
    def get_stream(self, stream_id: str) -> Optional[RTSPStream]:
        """Get a stream by ID."""
        return self.streams.get(stream_id)

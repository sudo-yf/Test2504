"""
Video processing module for EmotiSense.
Handles video capture and frame processing.
"""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

from .config import Config

logger = logging.getLogger(__name__)


class VideoCapture:
    """Manages video capture from camera."""
    
    def __init__(self, config: Config):
        """
        Initialize video capture.
        
        Args:
            config: Configuration instance
        """
        self.config = config
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_count = 0
        
        # Get video configuration
        video_config = config.video_config
        self.camera_index = video_config.get('camera_index', 0)
        self.frame_width = video_config.get('frame_width', 640)
        self.frame_height = video_config.get('frame_height', 360)
        self.fps = video_config.get('fps', 30)
        self.frame_skip = video_config.get('frame_skip', 2)
        
    def start(self) -> bool:
        """
        Start video capture.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.camera_index}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            logger.info(f"Video capture started: {self.frame_width}x{self.frame_height} @ {self.fps}fps")
            return True
            
        except Exception as e:
            logger.error(f"Error starting video capture: {e}")
            return False
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from camera.
        
        Returns:
            Tuple of (success, frame)
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None
            
        ret, frame = self.cap.read()
        
        if ret:
            self.frame_count += 1
            # Ensure frame is correct size
            if frame.shape[:2] != (self.frame_height, self.frame_width):
                frame = cv2.resize(frame, (self.frame_width, self.frame_height))
                
        return ret, frame
    
    def should_process_frame(self) -> bool:
        """
        Check if current frame should be processed (based on frame skip).
        
        Returns:
            True if frame should be processed
        """
        return self.frame_count % self.frame_skip == 0
    
    def release(self):
        """Release video capture resources."""
        if self.cap is not None:
            self.cap.release()
            logger.info("Video capture released")
            self.cap = None
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()


class FrameProcessor:
    """Processes video frames for face detection."""
    
    @staticmethod
    def to_grayscale(frame: np.ndarray) -> np.ndarray:
        """
        Convert frame to grayscale.
        
        Args:
            frame: BGR frame
            
        Returns:
            Grayscale frame
        """
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    @staticmethod
    def extract_roi(
        frame: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int
    ) -> np.ndarray:
        """
        Extract region of interest from frame.
        
        Args:
            frame: Source frame
            x, y, w, h: ROI coordinates and dimensions
            
        Returns:
            ROI frame
        """
        return frame[y:y+h, x:x+w].copy()
    
    @staticmethod
    def resize_frame(
        frame: np.ndarray,
        width: int,
        height: int
    ) -> np.ndarray:
        """
        Resize frame to specified dimensions.
        
        Args:
            frame: Source frame
            width: Target width
            height: Target height
            
        Returns:
            Resized frame
        """
        return cv2.resize(frame, (width, height))


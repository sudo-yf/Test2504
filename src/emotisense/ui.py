"""
UI rendering module for EmotiSense.
Handles display window and visual elements.
"""

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import Config

logger = logging.getLogger(__name__)


class UIRenderer:
    """Handles UI rendering for emotion detection display."""
    
    def __init__(self, config: Config):
        """
        Initialize UI renderer.
        
        Args:
            config: Configuration instance
        """
        self.config = config
        
        # Get UI configuration
        ui_config = config.ui_config
        self.window_name = ui_config.get('window_name', '情绪检测')
        self.window_mode = ui_config.get('window_mode', 'normal')
        
        # Font settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = ui_config.get('font_scale', 0.7)
        self.font_thickness = ui_config.get('font_thickness', 2)
        self.font_color = tuple(ui_config.get('font_color', [255, 255, 255]))
        self.font_outline_color = tuple(ui_config.get('font_outline_color', [0, 0, 0]))
        self.font_outline_thickness = ui_config.get('font_outline_thickness', 3)
        
        # Box settings
        self.face_box_color = tuple(ui_config.get('face_box_color', [255, 0, 0]))
        self.eye_box_color = tuple(ui_config.get('eye_box_color', [0, 255, 0]))
        self.box_thickness = ui_config.get('box_thickness', 2)
        
        self.window_created = False
        
    def create_window(self):
        """Create display window."""
        if not self.window_created:
            if self.window_mode == 'fullscreen':
                cv2.namedWindow(self.window_name, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            self.window_created = True
            logger.info(f"Created window: {self.window_name}")
    
    def draw_face_box(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int
    ):
        """
        Draw rectangle around detected face.
        
        Args:
            frame: Frame to draw on
            x, y, w, h: Face rectangle coordinates
        """
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            self.face_box_color,
            self.box_thickness
        )
    
    def draw_eye_boxes(
        self,
        roi_frame: np.ndarray,
        eyes: List[Tuple[int, int, int, int]]
    ):
        """
        Draw rectangles around detected eyes.
        
        Args:
            roi_frame: ROI frame containing face
            eyes: List of eye rectangles
        """
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(
                roi_frame,
                (ex, ey),
                (ex + ew, ey + eh),
                self.eye_box_color,
                self.box_thickness
            )
    
    def draw_emotion_text(
        self,
        frame: np.ndarray,
        emotion: str,
        probability: float,
        x: int,
        y: int
    ):
        """
        Draw emotion text with outline for better visibility.
        
        Args:
            frame: Frame to draw on
            emotion: Emotion name
            probability: Confidence percentage
            x, y: Text position (above face box)
        """
        text = f'{emotion}: {probability:.0f}%'
        position = (x, y - 10)
        
        # Draw outline (black)
        cv2.putText(
            frame,
            text,
            position,
            self.font,
            self.font_scale,
            self.font_outline_color,
            self.font_outline_thickness
        )
        
        # Draw text (white)
        cv2.putText(
            frame,
            text,
            position,
            self.font,
            self.font_scale,
            self.font_color,
            self.font_thickness
        )
    
    def draw_info_text(
        self,
        frame: np.ndarray,
        text: str,
        position: Tuple[int, int],
        color: Optional[Tuple[int, int, int]] = None
    ):
        """
        Draw informational text on frame.
        
        Args:
            frame: Frame to draw on
            text: Text to display
            position: Text position (x, y)
            color: Text color (defaults to font_color)
        """
        if color is None:
            color = self.font_color
            
        cv2.putText(
            frame,
            text,
            position,
            self.font,
            self.font_scale * 0.6,  # Smaller font for info
            color,
            self.font_thickness - 1
        )
    
    def show_frame(self, frame: np.ndarray):
        """
        Display frame in window.
        
        Args:
            frame: Frame to display
        """
        if not self.window_created:
            self.create_window()
            
        cv2.imshow(self.window_name, frame)
    
    def wait_key(self, delay: int = 1) -> int:
        """
        Wait for key press.
        
        Args:
            delay: Delay in milliseconds
            
        Returns:
            Key code (masked to 8 bits)
        """
        return cv2.waitKey(delay) & 0xFF
    
    def should_quit(self, key: int) -> bool:
        """
        Check if quit key was pressed.
        
        Args:
            key: Key code from wait_key()
            
        Returns:
            True if 'q' or ESC was pressed
        """
        return key == ord('q') or key == 27  # 27 is ESC
    
    def destroy_window(self):
        """Destroy display window."""
        if self.window_created:
            cv2.destroyAllWindows()
            self.window_created = False
            logger.info("Destroyed window")
    
    def __enter__(self):
        """Context manager entry."""
        self.create_window()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.destroy_window()


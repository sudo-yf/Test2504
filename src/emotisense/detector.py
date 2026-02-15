"""
Face and emotion detection module for EmotiSense.
Handles face detection, eye detection, and emotion analysis.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from deepface import DeepFace

from .config import Config

logger = logging.getLogger(__name__)


def create_emotion_detector(config: Config):
    """
    Factory function to create the appropriate emotion detector.

    Args:
        config: Configuration object

    Returns:
        Emotion detector instance
    """
    detector_type = config.get('emotion.detector_type', 'deepface')

    if detector_type == 'hsemotion':
        from .advanced_detectors import HSEmotionDetector
        logger.info("Using HSEmotion detector")
        return HSEmotionDetector(config)
    elif detector_type == 'fer':
        from .advanced_detectors import FERDetector
        logger.info("Using FER detector")
        return FERDetector(config)
    elif detector_type == 'ensemble':
        from .advanced_detectors import EnsembleEmotionDetector
        logger.info("Using Ensemble detector")
        return EnsembleEmotionDetector(config)
    else:  # 'deepface' or default
        logger.info("Using DeepFace detector")
        return EmotionDetector(config)


class FaceDetector:
    """Handles face and eye detection using OpenCV Haar Cascades."""
    
    def __init__(self, config: Config):
        """
        Initialize face detector.
        
        Args:
            config: Configuration instance
        """
        self.config = config
        
        # Load Haar Cascade classifiers
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        # Face tracking state
        self.last_face_rect: Optional[Tuple[int, int, int, int]] = None
        self.smoothing_factor = config.get('face_detection.smoothing_factor', 0.3)
        
    def detect_faces(self, gray_frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in grayscale frame.
        
        Args:
            gray_frame: Grayscale image frame
            
        Returns:
            List of face rectangles (x, y, w, h)
        """
        face_config = self.config.face_detection_config
        
        faces = self.face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=face_config.get('scale_factor', 1.1),
            minNeighbors=face_config.get('min_neighbors', 3),
            minSize=tuple(face_config.get('min_size', [80, 80])),
            maxSize=tuple(face_config.get('max_size', [300, 300]))
        )
        
        return [tuple(face) for face in faces]
    
    def detect_eyes(self, gray_roi: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect eyes in grayscale region of interest.
        
        Args:
            gray_roi: Grayscale ROI containing face
            
        Returns:
            List of eye rectangles (x, y, w, h)
        """
        eye_config = self.config.eye_detection_config
        
        eyes = self.eye_cascade.detectMultiScale(
            gray_roi,
            scaleFactor=eye_config.get('scale_factor', 1.1),
            minNeighbors=eye_config.get('min_neighbors', 3),
            minSize=tuple(eye_config.get('min_size', [20, 20]))
        )
        
        return [tuple(eye) for eye in eyes]
    
    def smooth_face_rect(
        self, 
        face_rect: Tuple[int, int, int, int]
    ) -> Tuple[int, int, int, int]:
        """
        Apply smoothing to face rectangle to reduce jitter.
        
        Args:
            face_rect: Current face rectangle (x, y, w, h)
            
        Returns:
            Smoothed face rectangle
        """
        if self.last_face_rect is None:
            self.last_face_rect = face_rect
            return face_rect
            
        x, y, w, h = face_rect
        last_x, last_y, last_w, last_h = self.last_face_rect
        
        # Apply exponential smoothing
        smooth_x = int(last_x * (1 - self.smoothing_factor) + x * self.smoothing_factor)
        smooth_y = int(last_y * (1 - self.smoothing_factor) + y * self.smoothing_factor)
        smooth_w = int(last_w * (1 - self.smoothing_factor) + w * self.smoothing_factor)
        smooth_h = int(last_h * (1 - self.smoothing_factor) + h * self.smoothing_factor)
        
        smoothed = (smooth_x, smooth_y, smooth_w, smooth_h)
        self.last_face_rect = smoothed
        
        return smoothed
    
    def reset_tracking(self):
        """Reset face tracking state."""
        self.last_face_rect = None


class EmotionDetector:
    """Handles emotion detection using DeepFace."""
    
    def __init__(self, config: Config):
        """
        Initialize emotion detector.
        
        Args:
            config: Configuration instance
        """
        self.config = config
        self.emotion_model = None
        self.anger_threshold = config.get('emotion.anger_threshold', 50)
        
    def analyze_emotion(self, face_img: np.ndarray) -> Tuple[str, float]:
        """
        Analyze emotion from face image.
        
        Args:
            face_img: Face image (BGR format)
            
        Returns:
            Tuple of (emotion_name, confidence_percentage)
        """
        try:
            # Analyze emotion using DeepFace
            result = DeepFace.analyze(
                face_img,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='skip' if self.emotion_model is not None else 'opencv'
            )
            
            # Get emotion results
            emotions = result[0]['emotion']
            
            # Initialize model flag after first successful analysis
            if self.emotion_model is None:
                self.emotion_model = emotions
            
            # Apply anger threshold filter
            if emotions.get('angry', 0) < self.anger_threshold:
                emotions['angry'] = 0
            
            # Get top emotion
            top_emotion = max(emotions.items(), key=lambda x: x[1])
            
            return top_emotion
            
        except Exception as e:
            logger.error(f"Emotion analysis error: {e}")
            return ('Unknown', 0.0)
    
    def is_high_confidence(self, confidence: float) -> bool:
        """
        Check if emotion confidence is high enough to log.
        
        Args:
            confidence: Confidence percentage
            
        Returns:
            True if confidence exceeds threshold
        """
        threshold = self.config.get('emotion.high_confidence_threshold', 95)
        return confidence > threshold


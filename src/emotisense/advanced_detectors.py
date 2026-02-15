"""
Advanced emotion detection models.
This module provides state-of-the-art emotion recognition models.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class HSEmotionDetector:
    """
    HSEmotion (EmotiEffLib) - High-Speed Emotion Recognition.
    
    State-of-the-art emotion recognition model with:
    - High accuracy (66%+ on AffectNet)
    - Fast inference (~60ms on mobile)
    - Pre-trained on VGGFace2 and AffectNet
    """
    
    def __init__(self, config: Any):
        """
        Initialize HSEmotion detector.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.model = None
        self._initialized = False
        
        # Emotion labels for 8-class model
        self.emotion_labels_8 = [
            'angry', 'contempt', 'disgust', 'fear',
            'happy', 'neutral', 'sad', 'surprise'
        ]
        
        # Emotion labels for 7-class model (without contempt)
        self.emotion_labels_7 = [
            'angry', 'disgust', 'fear', 'happy',
            'neutral', 'sad', 'surprise'
        ]
        
    def _lazy_init(self):
        """Lazy initialization of the model."""
        if self._initialized:
            return
        
        try:
            from hsemotion.facial_emotions import HSEmotionRecognizer
            
            # Use 8-class model by default (can be configured)
            model_name = self.config.get('emotion.hsemotion_model', 'enet_b0_8_best_afew')
            
            logger.info(f"Loading HSEmotion model: {model_name}")
            self.model = HSEmotionRecognizer(model_name=model_name)
            
            # Determine which labels to use
            if '8' in model_name:
                self.emotion_labels = self.emotion_labels_8
            else:
                self.emotion_labels = self.emotion_labels_7
            
            self._initialized = True
            logger.info("HSEmotion model loaded successfully")
            
        except ImportError:
            logger.error("HSEmotion library not installed. Install with: pip install hsemotion")
            raise
        except Exception as e:
            logger.error(f"Failed to load HSEmotion model: {e}")
            raise
    
    def analyze_emotion(self, face_img: np.ndarray) -> Tuple[str, float]:
        """
        Analyze emotion in a face image.
        
        Args:
            face_img: Face image (BGR format from OpenCV)
            
        Returns:
            Tuple of (emotion_name, confidence_percentage)
        """
        self._lazy_init()
        
        try:
            # HSEmotion expects RGB format
            import cv2
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            # Get emotion prediction
            emotion, scores = self.model.predict_emotions(face_rgb, logits=False)
            
            # Get the top emotion
            top_emotion_idx = np.argmax(scores)
            emotion_name = self.emotion_labels[top_emotion_idx]
            confidence = float(scores[top_emotion_idx] * 100)
            
            return emotion_name, confidence
            
        except Exception as e:
            logger.error(f"HSEmotion analysis error: {e}")
            return 'unknown', 0.0
    
    def get_all_emotions(self, face_img: np.ndarray) -> Dict[str, float]:
        """
        Get all emotion scores.
        
        Args:
            face_img: Face image (BGR format)
            
        Returns:
            Dictionary mapping emotion names to confidence scores
        """
        self._lazy_init()
        
        try:
            import cv2
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            emotion, scores = self.model.predict_emotions(face_rgb, logits=False)
            
            return {
                label: float(score * 100)
                for label, score in zip(self.emotion_labels, scores)
            }
            
        except Exception as e:
            logger.error(f"HSEmotion analysis error: {e}")
            return {}


class FERDetector:
    """
    FER (Facial Expression Recognition) library detector.
    
    Uses deep learning CNN for emotion recognition.
    Supports real-time video analysis.
    """
    
    def __init__(self, config: Any):
        """
        Initialize FER detector.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.detector = None
        self._initialized = False
        
    def _lazy_init(self):
        """Lazy initialization of the model."""
        if self._initialized:
            return
        
        try:
            from fer import FER as FERModel
            
            logger.info("Loading FER model...")
            # mtcnn=True uses MTCNN for face detection (more accurate but slower)
            use_mtcnn = self.config.get('emotion.fer_use_mtcnn', False)
            self.detector = FERModel(mtcnn=use_mtcnn)
            
            self._initialized = True
            logger.info("FER model loaded successfully")
            
        except ImportError:
            logger.error("FER library not installed. Install with: pip install fer")
            raise
        except Exception as e:
            logger.error(f"Failed to load FER model: {e}")
            raise
    
    def analyze_emotion(self, face_img: np.ndarray) -> Tuple[str, float]:
        """
        Analyze emotion in a face image.
        
        Args:
            face_img: Face image (BGR format from OpenCV)
            
        Returns:
            Tuple of (emotion_name, confidence_percentage)
        """
        self._lazy_init()
        
        try:
            # FER expects RGB format
            import cv2
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            # Detect emotions
            result = self.detector.detect_emotions(face_rgb)
            
            if not result:
                return 'unknown', 0.0
            
            # Get the first face's emotions
            emotions = result[0]['emotions']
            
            # Find top emotion
            top_emotion = max(emotions.items(), key=lambda x: x[1])
            emotion_name = top_emotion[0]
            confidence = float(top_emotion[1] * 100)
            
            return emotion_name, confidence
            
        except Exception as e:
            logger.error(f"FER analysis error: {e}")
            return 'unknown', 0.0
    
    def get_all_emotions(self, face_img: np.ndarray) -> Dict[str, float]:
        """
        Get all emotion scores.
        
        Args:
            face_img: Face image (BGR format)
            
        Returns:
            Dictionary mapping emotion names to confidence scores
        """
        self._lazy_init()
        
        try:
            import cv2
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            result = self.detector.detect_emotions(face_rgb)
            
            if not result:
                return {}
            
            emotions = result[0]['emotions']
            
            return {
                emotion: float(score * 100)
                for emotion, score in emotions.items()
            }
            
        except Exception as e:
            logger.error(f"FER analysis error: {e}")
            return {}


class EnsembleEmotionDetector:
    """
    Ensemble detector that combines multiple models for better accuracy.
    """
    
    def __init__(self, config: Any):
        """
        Initialize ensemble detector.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.detectors = []
        
        # Initialize available detectors
        enabled_models = config.get('emotion.ensemble_models', ['hsemotion'])
        
        for model_name in enabled_models:
            try:
                if model_name == 'hsemotion':
                    self.detectors.append(HSEmotionDetector(config))
                    logger.info("Added HSEmotion to ensemble")
                elif model_name == 'fer':
                    self.detectors.append(FERDetector(config))
                    logger.info("Added FER to ensemble")
            except Exception as e:
                logger.warning(f"Failed to add {model_name} to ensemble: {e}")
        
        if not self.detectors:
            raise RuntimeError("No detectors available for ensemble")
    
    def analyze_emotion(self, face_img: np.ndarray) -> Tuple[str, float]:
        """
        Analyze emotion using ensemble of models.
        
        Args:
            face_img: Face image (BGR format)
            
        Returns:
            Tuple of (emotion_name, confidence_percentage)
        """
        # Collect predictions from all models
        all_emotions = {}
        
        for detector in self.detectors:
            try:
                emotions = detector.get_all_emotions(face_img)
                for emotion, score in emotions.items():
                    if emotion not in all_emotions:
                        all_emotions[emotion] = []
                    all_emotions[emotion].append(score)
            except Exception as e:
                logger.warning(f"Detector failed: {e}")
        
        if not all_emotions:
            return 'unknown', 0.0
        
        # Average scores across models
        averaged_emotions = {
            emotion: np.mean(scores)
            for emotion, scores in all_emotions.items()
        }
        
        # Get top emotion
        top_emotion = max(averaged_emotions.items(), key=lambda x: x[1])
        
        return top_emotion[0], float(top_emotion[1])


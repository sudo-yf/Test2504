"""
Model comparison tool for emotion detection.
Compares different emotion recognition models on the same input.
"""

import cv2
import time
import numpy as np
from typing import Dict, List, Tuple
import logging

from src.config import get_config
from src.detector import EmotionDetector, create_emotion_detector
from src.advanced_detectors import HSEmotionDetector, FERDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelComparator:
    """Compare different emotion detection models."""
    
    def __init__(self):
        """Initialize comparator with all available models."""
        self.config = get_config()
        self.models = {}
        
        # Try to load all available models
        self._load_models()
    
    def _load_models(self):
        """Load all available emotion detection models."""
        # DeepFace
        try:
            logger.info("Loading DeepFace model...")
            self.models['DeepFace'] = EmotionDetector(self.config)
            logger.info("✓ DeepFace loaded")
        except Exception as e:
            logger.warning(f"✗ Failed to load DeepFace: {e}")
        
        # HSEmotion
        try:
            logger.info("Loading HSEmotion model...")
            self.models['HSEmotion'] = HSEmotionDetector(self.config)
            logger.info("✓ HSEmotion loaded")
        except Exception as e:
            logger.warning(f"✗ Failed to load HSEmotion: {e}")
        
        # FER
        try:
            logger.info("Loading FER model...")
            self.models['FER'] = FERDetector(self.config)
            logger.info("✓ FER loaded")
        except Exception as e:
            logger.warning(f"✗ Failed to load FER: {e}")
        
        if not self.models:
            raise RuntimeError("No models could be loaded!")
        
        logger.info(f"\nLoaded {len(self.models)} models: {list(self.models.keys())}")
    
    def compare_on_image(self, image_path: str) -> Dict[str, Tuple[str, float, float]]:
        """
        Compare all models on a single image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary mapping model name to (emotion, confidence, inference_time)
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Detect face (using OpenCV for consistency)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        faces = face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(80, 80))
        
        if len(faces) == 0:
            raise ValueError("No face detected in image")
        
        # Extract face region
        x, y, w, h = faces[0]
        face_img = img[y:y+h, x:x+w]
        
        # Compare models
        results = {}
        
        for model_name, model in self.models.items():
            try:
                start_time = time.time()
                emotion, confidence = model.analyze_emotion(face_img)
                inference_time = (time.time() - start_time) * 1000  # Convert to ms
                
                results[model_name] = (emotion, confidence, inference_time)
                
            except Exception as e:
                logger.error(f"Error with {model_name}: {e}")
                results[model_name] = ('error', 0.0, 0.0)
        
        return results
    
    def compare_on_webcam(self, duration: int = 30):
        """
        Compare models on webcam feed.
        
        Args:
            duration: Duration in seconds to run comparison
        """
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        start_time = time.time()
        frame_count = 0
        
        # Statistics
        stats = {name: {'times': [], 'emotions': []} for name in self.models.keys()}
        
        print(f"\nRunning comparison for {duration} seconds...")
        print("Press 'q' to quit early\n")
        
        try:
            while (time.time() - start_time) < duration:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Process every 10th frame
                if frame_count % 10 != 0:
                    cv2.imshow('Model Comparison', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue
                
                # Detect face
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(80, 80))
                
                if len(faces) == 0:
                    cv2.imshow('Model Comparison', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue
                
                # Extract face
                x, y, w, h = faces[0]
                face_img = frame[y:y+h, x:x+w]
                
                # Draw face box
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Compare models
                y_offset = y - 10
                
                for model_name, model in self.models.items():
                    try:
                        start = time.time()
                        emotion, confidence = model.analyze_emotion(face_img)
                        inference_time = (time.time() - start) * 1000
                        
                        # Store stats
                        stats[model_name]['times'].append(inference_time)
                        stats[model_name]['emotions'].append(emotion)
                        
                        # Display result
                        text = f"{model_name}: {emotion} ({confidence:.0f}%) {inference_time:.0f}ms"
                        cv2.putText(frame, text, (10, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        y_offset -= 20
                        
                    except Exception as e:
                        logger.error(f"Error with {model_name}: {e}")
                
                cv2.imshow('Model Comparison', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        # Print statistics
        self._print_statistics(stats)
    
    def _print_statistics(self, stats: Dict):
        """Print comparison statistics."""
        print("\n" + "="*70)
        print("MODEL COMPARISON STATISTICS")
        print("="*70)
        
        for model_name, data in stats.items():
            if not data['times']:
                continue
            
            times = data['times']
            emotions = data['emotions']
            
            print(f"\n{model_name}:")
            print(f"  Samples: {len(times)}")
            print(f"  Avg inference time: {np.mean(times):.1f}ms")
            print(f"  Min/Max time: {np.min(times):.1f}ms / {np.max(times):.1f}ms")
            
            # Emotion distribution
            unique, counts = np.unique(emotions, return_counts=True)
            print(f"  Emotion distribution:")
            for emotion, count in zip(unique, counts):
                percentage = (count / len(emotions)) * 100
                print(f"    {emotion}: {count} ({percentage:.1f}%)")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare emotion detection models')
    parser.add_argument('--mode', choices=['image', 'webcam'], default='webcam',
                       help='Comparison mode')
    parser.add_argument('--image', type=str, help='Path to image file (for image mode)')
    parser.add_argument('--duration', type=int, default=30,
                       help='Duration in seconds (for webcam mode)')
    
    args = parser.parse_args()
    
    try:
        comparator = ModelComparator()
        
        if args.mode == 'image':
            if not args.image:
                print("Error: --image required for image mode")
                return
            
            results = comparator.compare_on_image(args.image)
            
            print("\n" + "="*70)
            print("COMPARISON RESULTS")
            print("="*70)
            
            for model_name, (emotion, confidence, time_ms) in results.items():
                print(f"{model_name:15s}: {emotion:10s} ({confidence:5.1f}%) - {time_ms:6.1f}ms")
        
        else:  # webcam
            comparator.compare_on_webcam(args.duration)
    
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)


if __name__ == '__main__':
    main()


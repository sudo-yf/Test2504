"""
EmotiSense - Real-time Emotion Detection System
Main entry point for the application.
"""

import time
import logging
from typing import Optional

from src.config import get_config
from src.video_processor import VideoCapture, FrameProcessor
from src.detector import FaceDetector, create_emotion_detector
from src.data_manager import EmotionDataManager
from src.analyzer import EmotionAnalyzer
from src.ui import UIRenderer


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmotionDetectionApp:
    """Main application class for EmotiSense."""

    def __init__(self):
        """Initialize the application."""
        # Load configuration
        self.config = get_config()

        # Initialize components
        self.video_capture = VideoCapture(self.config)
        self.frame_processor = FrameProcessor()
        self.face_detector = FaceDetector(self.config)
        self.emotion_detector = create_emotion_detector(self.config)  # Use factory function
        self.data_manager = EmotionDataManager(self.config)
        self.analyzer = EmotionAnalyzer(self.config)
        self.ui = UIRenderer(self.config)

        # Timing
        self.last_detection_time = time.time()
        self.detection_interval = self.config.get('emotion.detection_interval', 3.0)

    def should_analyze_emotion(self) -> bool:
        """Check if enough time has passed for emotion analysis."""
        current_time = time.time()
        if (current_time - self.last_detection_time) >= self.detection_interval:
            self.last_detection_time = current_time
            return True
        return False
    def process_frame(self, frame):
        """
        Process a single frame for face and emotion detection.

        Args:
            frame: Video frame to process

        Returns:
            Processed frame with annotations
        """
        # Convert to grayscale for face detection
        gray = self.frame_processor.to_grayscale(frame)

        # Detect faces
        faces = self.face_detector.detect_faces(gray)

        if len(faces) > 0:
            # Process first detected face
            face_rect = faces[0]
            x, y, w, h = self.face_detector.smooth_face_rect(face_rect)

            # Analyze emotion if interval has passed
            if self.should_analyze_emotion():
                face_img = self.frame_processor.extract_roi(frame, x, y, w, h)
                emotion, probability = self.emotion_detector.analyze_emotion(face_img)

                # Store emotion data
                record = self.data_manager.add_record(emotion, probability)

                # Log high-confidence emotions
                if self.emotion_detector.is_high_confidence(probability):
                    self.data_manager.log_high_confidence_emotion(record)

                # Draw emotion text
                self.ui.draw_emotion_text(frame, emotion, probability, x, y)

        return frame

    def run(self):
        """Run the main application loop."""
        logger.info("Starting EmotiSense...")

        # Start video capture
        if not self.video_capture.start():
            logger.error("Failed to start video capture")
            return

        try:
            while True:
                # Read frame
                ret, frame = self.video_capture.read_frame()
                if not ret:
                    logger.error("Failed to read frame from camera")
                    break

                # Check if we should process this frame
                if self.video_capture.should_process_frame():
                    # Cleanup old data if needed
                    if self.data_manager.should_cleanup():
                        self.data_manager.cleanup_old_data()

                    # Process frame
                    frame = self.process_frame(frame)

                # Display frame
                self.ui.show_frame(frame)

                # Check for quit
                key = self.ui.wait_key(1)
                if self.ui.should_quit(key):
                    logger.info("Quit signal received")
                    break

        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources and perform final analysis."""
        logger.info("Cleaning up...")

        # Release video capture
        self.video_capture.release()

        # Destroy UI window
        self.ui.destroy_window()

        # Perform emotion analysis if we have data
        self.perform_final_analysis()

        logger.info("EmotiSense stopped")

    def perform_final_analysis(self):
        """Perform final emotion analysis using DeepSeek API."""
        records = self.data_manager.get_all_records()

        if not records:
            logger.info("No emotion data to analyze")
            return

        logger.info(f"Analyzing {len(records)} emotion records...")

        # Get statistics
        stats = self.data_manager.get_statistics()
        logger.info(f"Statistics: {stats}")

        # Read log file and analyze
        log_lines = self.data_manager.read_log_file()

        if log_lines:
            analysis = self.analyzer.analyze_emotion_logs(log_lines)

            if analysis:
                # Format and display analysis
                formatted = self.analyzer.format_analysis_result(analysis)
                print(formatted)

                # Save analysis to log file
                self.data_manager.append_to_log(formatted)
                logger.info("Analysis saved to log file")
            else:
                logger.warning("Failed to get analysis from DeepSeek API")


def main():
    """Main entry point."""
    try:
        app = EmotionDetectionApp()
        app.run()
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)


if __name__ == '__main__':
    main()
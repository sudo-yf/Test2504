"""Application orchestration for EmotiSense."""

from __future__ import annotations

import logging
import time
from typing import Optional

from .analyzer import EmotionAnalyzer
from .config import Config, get_config
from .data_manager import EmotionDataManager
from .detector import FaceDetector, create_emotion_detector
from .product_ops import SessionReporter
from .ui import UIRenderer
from .video_processor import FrameProcessor, VideoCapture

logger = logging.getLogger(__name__)


class EmotionDetectionApp:
    """Main real-time emotion detection application."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()

        self.video_capture = VideoCapture(self.config)
        self.frame_processor = FrameProcessor()
        self.face_detector = FaceDetector(self.config)
        self.emotion_detector = create_emotion_detector(self.config)
        self.data_manager = EmotionDataManager(self.config)
        self.analyzer = EmotionAnalyzer(self.config)
        self.reporter = SessionReporter(self.config, analyzer=self.analyzer)
        self.ui = UIRenderer(self.config)

        self.last_detection_time = time.time()
        self.detection_interval = self.config.get("emotion.detection_interval", 3.0)

    def should_analyze_emotion(self) -> bool:
        current_time = time.time()
        if (current_time - self.last_detection_time) >= self.detection_interval:
            self.last_detection_time = current_time
            return True
        return False

    def process_frame(self, frame):
        gray = self.frame_processor.to_grayscale(frame)
        faces = self.face_detector.detect_faces(gray)

        if not faces:
            return frame

        x, y, w, h = self.face_detector.smooth_face_rect(faces[0])

        if self.should_analyze_emotion():
            face_img = self.frame_processor.extract_roi(frame, x, y, w, h)
            emotion, probability = self.emotion_detector.analyze_emotion(face_img)
            record = self.data_manager.add_record(emotion, probability)

            if self.emotion_detector.is_high_confidence(probability):
                self.data_manager.log_high_confidence_emotion(record)

            self.ui.draw_emotion_text(frame, emotion, probability, x, y)

        return frame

    def run(self):
        logger.info("Starting EmotiSense...")

        if not self.video_capture.start():
            logger.error("Failed to start video capture")
            return

        try:
            while True:
                ok, frame = self.video_capture.read_frame()
                if not ok:
                    logger.error("Failed to read frame from camera")
                    break

                if self.video_capture.should_process_frame():
                    if self.data_manager.should_cleanup():
                        self.data_manager.cleanup_old_data()
                    frame = self.process_frame(frame)

                self.ui.show_frame(frame)
                key = self.ui.wait_key(1)
                if self.ui.should_quit(key):
                    logger.info("Quit signal received")
                    break
        finally:
            self.cleanup()

    def cleanup(self):
        logger.info("Cleaning up...")
        self.video_capture.release()
        self.ui.destroy_window()
        self.perform_final_analysis()
        logger.info("EmotiSense stopped")

    def perform_final_analysis(self):
        records = self.data_manager.get_all_records()
        logger.info("Collected %s in-memory emotion records", len(records))
        logger.info("Current statistics: %s", self.data_manager.get_statistics())

        exported = self.reporter.export(self.data_manager, mode="realtime")
        snapshot = exported["snapshot"]
        analysis = snapshot.get("analysis_text")
        if analysis:
            formatted = self.analyzer.format_analysis_result(analysis)
            print(formatted)
            self.data_manager.append_to_log(formatted)

        logger.info("Session report exported: %s", exported["json_path"])
        logger.info("Session report exported: %s", exported["md_path"])

"""
Data management module for EmotiSense.
Handles emotion data storage, logging, and cleanup.
"""

import gc
import logging
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import Config

logger = logging.getLogger(__name__)


@dataclass
class EmotionRecord:
    """Represents a single emotion detection record."""
    timestamp: float
    emotion: str
    probability: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def format_timestamp(self) -> str:
        """Format timestamp as readable string."""
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp))
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.format_timestamp()} - {self.emotion}: {self.probability:.0f}%"


class EmotionDataManager:
    """Manages emotion detection data and logging."""
    
    def __init__(self, config: Config):
        """
        Initialize data manager.
        
        Args:
            config: Configuration instance
        """
        self.config = config
        self.emotion_data: List[EmotionRecord] = []
        
        # Get configuration
        data_config = config.data_config
        self.log_file = Path(data_config.get('log_file', 'emotion_log.txt'))
        self.enable_logging = data_config.get('enable_logging', True)
        self.db_path = Path(data_config.get('database_path', 'data/emotions.db'))
        self.max_records = config.get('emotion.max_data_records', 1000)
        self.cleanup_interval = data_config.get('cleanup_interval', 30)
        
        # Cleanup tracking
        self.last_cleanup_time = time.time()
        self._init_db()

    def _init_db(self):
        """Initialize SQLite storage."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS emotion_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    emotion TEXT NOT NULL,
                    probability REAL NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_emotion_records_ts ON emotion_records(timestamp)"
            )
        
    def add_record(
        self,
        emotion: str,
        probability: float,
        timestamp: Optional[float] = None
    ) -> EmotionRecord:
        """
        Add emotion detection record.
        
        Args:
            emotion: Detected emotion name
            probability: Confidence percentage
            timestamp: Record timestamp (defaults to current time)
            
        Returns:
            Created EmotionRecord
        """
        if timestamp is None:
            timestamp = time.time()
            
        record = EmotionRecord(
            timestamp=timestamp,
            emotion=emotion,
            probability=probability
        )
        
        self.emotion_data.append(record)
        self._persist_record(record)
        
        return record

    def _persist_record(self, record: EmotionRecord):
        """Persist a record into SQLite."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO emotion_records (timestamp, emotion, probability) VALUES (?, ?, ?)",
                    (record.timestamp, record.emotion, record.probability),
                )
        except Exception as e:
            logger.error(f"Error persisting emotion record: {e}")
    
    def log_high_confidence_emotion(self, record: EmotionRecord):
        """
        Log high-confidence emotion to file.
        
        Args:
            record: Emotion record to log
        """
        if not self.enable_logging:
            return
            
        try:
            message = f"{record.format_timestamp()} 检测到高强度情绪：{record.emotion} ({record.probability:.0f}%)"
            
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(message + '\n')
                
            logger.info(f"Logged high-confidence emotion: {message}")
            
        except Exception as e:
            logger.error(f"Error logging emotion: {e}")
    
    def get_all_records(self) -> List[EmotionRecord]:
        """
        Get all emotion records.
        
        Returns:
            List of emotion records
        """
        return self.emotion_data.copy()
    
    def get_recent_records(self, count: int) -> List[EmotionRecord]:
        """
        Get most recent emotion records.
        
        Args:
            count: Number of records to retrieve
            
        Returns:
            List of recent emotion records
        """
        return self.emotion_data[-count:] if self.emotion_data else []
    
    def cleanup_old_data(self, force: bool = False):
        """
        Clean up old emotion data to manage memory.
        
        Args:
            force: Force cleanup regardless of interval
        """
        current_time = time.time()
        
        # Check if cleanup is needed
        if not force and (current_time - self.last_cleanup_time) < self.cleanup_interval:
            return
            
        # Trim data if exceeds max records
        if len(self.emotion_data) > self.max_records:
            old_count = len(self.emotion_data)
            self.emotion_data = self.emotion_data[-self.max_records:]
            removed = old_count - len(self.emotion_data)
            
            logger.info(f"Cleaned up {removed} old emotion records")
            
            # Trigger garbage collection
            gc.collect()
            
        self.last_cleanup_time = current_time
    
    def should_cleanup(self) -> bool:
        """
        Check if cleanup should be performed.
        
        Returns:
            True if cleanup interval has elapsed
        """
        return (time.time() - self.last_cleanup_time) >= self.cleanup_interval
    
    def read_log_file(self) -> List[str]:
        """
        Read all lines from log file.
        
        Returns:
            List of log lines
        """
        if not self.log_file.exists():
            return []
            
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                return f.readlines()
        except Exception as e:
            logger.error(f"Error reading log file: {e}")
            return []
    
    def append_to_log(self, content: str):
        """
        Append content to log file.
        
        Args:
            content: Content to append
        """
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(content + '\n')
        except Exception as e:
            logger.error(f"Error appending to log file: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about collected emotion data.
        
        Returns:
            Dictionary containing statistics
        """
        if not self.emotion_data:
            return {
                'total_records': 0,
                'emotions': {},
                'average_confidence': 0.0
            }
        
        # Count emotions
        emotion_counts: Dict[str, int] = {}
        total_confidence = 0.0
        
        for record in self.emotion_data:
            emotion_counts[record.emotion] = emotion_counts.get(record.emotion, 0) + 1
            total_confidence += record.probability
        
        return {
            'total_records': len(self.emotion_data),
            'emotions': emotion_counts,
            'average_confidence': total_confidence / len(self.emotion_data),
            'time_span': {
                'start': self.emotion_data[0].format_timestamp(),
                'end': self.emotion_data[-1].format_timestamp()
            }
        }

    def get_db_statistics(self) -> Dict[str, Any]:
        """Get historical statistics from SQLite."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                total = conn.execute("SELECT COUNT(*) FROM emotion_records").fetchone()[0]
                avg = conn.execute("SELECT AVG(probability) FROM emotion_records").fetchone()[0] or 0.0
                rows = conn.execute(
                    "SELECT emotion, COUNT(*) FROM emotion_records GROUP BY emotion"
                ).fetchall()
        except Exception as e:
            logger.error(f"Error getting db statistics: {e}")
            return {'total_records': 0, 'emotions': {}, 'average_confidence': 0.0}

        return {
            'total_records': int(total),
            'emotions': {emotion: int(count) for emotion, count in rows},
            'average_confidence': float(avg),
        }
    
    def clear_all_data(self):
        """Clear all emotion data from memory."""
        self.emotion_data.clear()
        gc.collect()
        logger.info("Cleared all emotion data")

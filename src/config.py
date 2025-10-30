"""
Configuration management module for EmotiSense.
Handles loading and accessing configuration from YAML file.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from dotenv import load_dotenv


class Config:
    """Configuration manager for EmotiSense application."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration YAML file. If None, uses default.
        """
        # Load environment variables
        load_dotenv()
        
        # Determine config file path
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config.yaml"
        else:
            config_path = Path(config_path)
            
        # Load configuration
        self._config: Dict[str, Any] = self._load_config(config_path)
        
    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Dictionary containing configuration
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        return config or {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'video.frame_width')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
                
        return value
    
    @property
    def deepseek_api_key(self) -> Optional[str]:
        """Get DeepSeek API key from environment variables."""
        api_key = os.getenv('DEEPSEEK_API_KEY')
        if api_key and api_key != 'YOUR_API_KEY_HERE':
            return api_key
        return None
    
    @property
    def video_config(self) -> Dict[str, Any]:
        """Get video capture configuration."""
        return self._config.get('video', {})
    
    @property
    def face_detection_config(self) -> Dict[str, Any]:
        """Get face detection configuration."""
        return self._config.get('face_detection', {})
    
    @property
    def eye_detection_config(self) -> Dict[str, Any]:
        """Get eye detection configuration."""
        return self._config.get('eye_detection', {})
    
    @property
    def emotion_config(self) -> Dict[str, Any]:
        """Get emotion detection configuration."""
        return self._config.get('emotion', {})
    
    @property
    def data_config(self) -> Dict[str, Any]:
        """Get data management configuration."""
        return self._config.get('data', {})
    
    @property
    def api_config(self) -> Dict[str, Any]:
        """Get API configuration."""
        return self._config.get('api', {})
    
    @property
    def ui_config(self) -> Dict[str, Any]:
        """Get UI configuration."""
        return self._config.get('ui', {})


# Global configuration instance
_config_instance: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """
    Get global configuration instance (singleton pattern).
    
    Args:
        config_path: Path to configuration file (only used on first call)
        
    Returns:
        Config instance
    """
    global _config_instance
    
    if _config_instance is None:
        _config_instance = Config(config_path)
        
    return _config_instance


"""
Emotion analysis module for EmotiSense.
Handles DeepSeek API integration for emotion trend analysis.
"""

import logging
from typing import List, Optional

import requests

from .config import Config

logger = logging.getLogger(__name__)


class EmotionAnalyzer:
    """Analyzes emotion trends using DeepSeek API."""
    
    def __init__(self, config: Config):
        """
        Initialize emotion analyzer.
        
        Args:
            config: Configuration instance
        """
        self.config = config
        self.api_key = config.deepseek_api_key
        
        # Get API configuration
        api_config = config.api_config
        self.enabled = api_config.get('enabled', True) and self.api_key is not None
        self.model = api_config.get('model', 'deepseek-chat')
        self.endpoint = api_config.get('endpoint', 'https://api.deepseek.com/v1/chat/completions')
        self.analysis_prompt = api_config.get(
            'analysis_prompt',
            '分析以下情绪检测数据，找出主要情绪趋势和重要变化点：\n'
        )
        
    def analyze_emotion_logs(self, log_lines: List[str]) -> Optional[str]:
        """
        Analyze emotion logs using DeepSeek API.
        
        Args:
            log_lines: List of log lines to analyze
            
        Returns:
            Analysis result text, or None if analysis failed
        """
        if not self.enabled:
            logger.warning("DeepSeek API is not enabled or API key is missing")
            return None
            
        if not log_lines:
            logger.info("No log lines to analyze")
            return None
        
        try:
            # Prepare request
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # Build analysis prompt
            prompt = self.analysis_prompt + ''.join(log_lines)
            
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
            
            # Make API request
            logger.info("Sending emotion analysis request to DeepSeek API...")
            response = requests.post(
                self.endpoint,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            # Check response
            if response.status_code == 200:
                result = response.json()
                analysis_content = result['choices'][0]['message']['content']
                logger.info("Successfully received analysis from DeepSeek API")
                return analysis_content
            else:
                logger.error(f"API request failed with status {response.status_code}: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            logger.error("API request timed out")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"API request error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during analysis: {e}")
            return None
    
    def format_analysis_result(self, analysis: str) -> str:
        """
        Format analysis result for logging.
        
        Args:
            analysis: Raw analysis text
            
        Returns:
            Formatted analysis text
        """
        return f"\n{'='*60}\nDeepSeek分析结果：\n{analysis}\n{'='*60}\n"


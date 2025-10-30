"""
Simple test script to verify all modules can be imported correctly.
Run this before running the main application to check for import errors.
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all modules can be imported."""
    print("Testing module imports...")
    
    try:
        print("  ✓ Importing src.config...")
        from src.config import Config, get_config
        
        print("  ✓ Importing src.detector...")
        from src.detector import FaceDetector, EmotionDetector
        
        print("  ✓ Importing src.video_processor...")
        from src.video_processor import VideoCapture, FrameProcessor
        
        print("  ✓ Importing src.data_manager...")
        from src.data_manager import EmotionDataManager, EmotionRecord
        
        print("  ✓ Importing src.analyzer...")
        from src.analyzer import EmotionAnalyzer
        
        print("  ✓ Importing src.ui...")
        from src.ui import UIRenderer
        
        print("\n✅ All modules imported successfully!")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False


def test_config():
    """Test configuration loading."""
    print("\nTesting configuration...")
    
    try:
        from src.config import get_config
        
        config = get_config()
        
        # Test some config values
        print(f"  ✓ Video config: {config.video_config}")
        print(f"  ✓ Emotion config: {config.emotion_config}")
        print(f"  ✓ API enabled: {config.api_config.get('enabled')}")
        
        print("\n✅ Configuration loaded successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Configuration error: {e}")
        return False


def test_dependencies():
    """Test that required dependencies are installed."""
    print("\nTesting dependencies...")
    
    dependencies = [
        ('cv2', 'opencv-python'),
        ('numpy', 'numpy'),
        ('deepface', 'deepface'),
        ('tensorflow', 'tensorflow'),
        ('yaml', 'pyyaml'),
        ('dotenv', 'python-dotenv'),
        ('requests', 'requests'),
    ]
    
    all_ok = True
    
    for module_name, package_name in dependencies:
        try:
            __import__(module_name)
            print(f"  ✓ {package_name}")
        except ImportError:
            print(f"  ❌ {package_name} - NOT INSTALLED")
            all_ok = False
    
    if all_ok:
        print("\n✅ All dependencies are installed!")
    else:
        print("\n❌ Some dependencies are missing. Run: pip install -r requirements.txt")
    
    return all_ok


def main():
    """Run all tests."""
    print("=" * 60)
    print("EmotiSense Module Test Suite")
    print("=" * 60)
    
    results = []
    
    # Test dependencies first
    results.append(("Dependencies", test_dependencies()))
    
    # Test imports
    results.append(("Module Imports", test_imports()))
    
    # Test configuration
    results.append(("Configuration", test_config()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n🎉 All tests passed! You can run the application with: python main.py")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix the issues before running the application.")
        return 1


if __name__ == '__main__':
    sys.exit(main())


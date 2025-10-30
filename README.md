# EmotiSense: Real-time Emotion Detection System

A professional, modular real-time emotion detection system powered by OpenCV and DeepFace, capable of analyzing facial expressions and emotional states through webcam. Features include real-time face detection, emotion tracking, high-confidence emotion logging (>95%), and emotion trend analysis with DeepSeek API integration.

## ✨ Features

- 🎥 **Real-time Face Detection** - Haar Cascade-based face and eye tracking
- 😊 **Multiple Emotion Models** - Choose from HSEmotion, FER, DeepFace, or Ensemble
- 🏆 **State-of-the-Art Accuracy** - HSEmotion achieves 66%+ accuracy on AffectNet
- 📊 **High-Confidence Logging** - Automatic logging of emotions >95% confidence
- 🤖 **AI-Powered Analysis** - DeepSeek API integration for emotion trend analysis
- ⚡ **Performance Optimized** - Frame skipping, memory management, smooth tracking
- 🏗️ **Modular Architecture** - Clean, maintainable, object-oriented design
- ⚙️ **Configurable** - YAML-based configuration for easy customization
- 🔬 **Model Comparison Tool** - Compare different models side-by-side

## 🏗️ Project Structure

```
EmotiSense/
├── src/
│   ├── __init__.py          # Package initialization
│   ├── config.py            # Configuration management
│   ├── detector.py          # Face and emotion detection
│   ├── video_processor.py   # Video capture and processing
│   ├── data_manager.py      # Data storage and logging
│   ├── analyzer.py          # DeepSeek API integration
│   └── ui.py                # UI rendering
├── main.py                  # Application entry point
├── config.yaml              # Configuration file
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variables template
└── README.md                # This file
```

## 📋 Requirements

- Python 3.7+
- Webcam
- (Optional) DeepSeek API key for emotion analysis

## 🚀 Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd EmotiSense-Real-time-Emotion-Detection-System
```

2. **Install dependencies**
```bash
# Basic installation (DeepFace only)
pip install -r requirements.txt

# Or use the interactive installer for advanced models
python install_models.py
```

3. **Install advanced models** (Optional but recommended)
```bash
# HSEmotion (Recommended - fast and accurate)
pip install hsemotion timm

# FER (Lightweight alternative)
pip install fer

# Or install all models
pip install hsemotion timm fer
```

4. **Configure environment variables**
```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your DeepSeek API key
DEEPSEEK_API_KEY=your_api_key_here
```

5. **Select your emotion model**
Edit `config.yaml`:
```yaml
emotion:
  detector_type: 'hsemotion'  # Options: 'hsemotion', 'fer', 'deepface', 'ensemble'
```

## 🎮 Usage

### Basic Usage

**Start the application:**
```bash
python main.py
```

**Controls:**
- Press `q` or `ESC` to quit
- The application will automatically:
  - Detect faces and emotions in real-time
  - Log high-confidence emotions to `emotion_log.txt`
  - Analyze emotion trends on exit (if API key is configured)

### Compare Models

**Compare different emotion recognition models:**
```bash
# Real-time webcam comparison
python compare_models.py --mode webcam --duration 30

# Compare on a single image
python compare_models.py --mode image --image path/to/image.jpg
```

This will show you:
- Emotion predictions from each model
- Inference time for each model
- Emotion distribution statistics

## 📊 Output

- **Real-time Display**: Video feed with face boxes, eye boxes, and emotion labels
- **Log File**: `emotion_log.txt` - High-confidence emotion records (>95%)
- **Analysis Report**: AI-generated emotion trend analysis (appended to log file)

## ⚙️ Configuration

The `config.yaml` file allows you to customize:

- **Video Settings**: Resolution, FPS, frame skip rate
- **Detection Parameters**: Face/eye detection thresholds, smoothing
- **Emotion Settings**: Detection interval, confidence thresholds
- **UI Appearance**: Colors, fonts, window mode
- **Data Management**: Log file location, cleanup intervals

Example configuration:
```yaml
video:
  frame_width: 640
  frame_height: 360
  frame_skip: 2

emotion:
  detection_interval: 3.0
  high_confidence_threshold: 95
```

## 🔧 Performance Optimization

- **Reduced Resolution**: 640x360 default (configurable)
- **Frame Skipping**: Process every Nth frame (default: 2)
- **Memory Management**: Automatic cleanup of old data (max 1000 records)
- **Smooth Tracking**: Exponential smoothing for stable face boxes
- **Lazy Loading**: Models loaded on first use

**Memory Usage**: ~150-200MB total
- DeepFace model: ~100MB
- Video buffers: ~1MB
- Emotion data: <1MB (with cleanup)

## 🏛️ Architecture

### Modular Design

The application follows a clean, modular architecture with separation of concerns:

1. **Config Module** (`config.py`)
   - Centralized configuration management
   - YAML-based settings
   - Environment variable handling

2. **Detector Module** (`detector.py`)
   - `FaceDetector`: Face and eye detection with Haar Cascades
   - `EmotionDetector`: Emotion analysis using DeepFace
   - Smooth tracking and confidence filtering

3. **Video Processor** (`video_processor.py`)
   - `VideoCapture`: Camera management with context manager
   - `FrameProcessor`: Frame manipulation utilities
   - Frame skipping logic

4. **Data Manager** (`data_manager.py`)
   - `EmotionRecord`: Data class for emotion records
   - `EmotionDataManager`: Storage, logging, and statistics
   - Automatic cleanup and memory management

5. **Analyzer** (`analyzer.py`)
   - `EmotionAnalyzer`: DeepSeek API integration
   - Emotion trend analysis
   - Result formatting

6. **UI Renderer** (`ui.py`)
   - `UIRenderer`: Display window and visual elements
   - Configurable colors, fonts, and styles
   - Drawing utilities for boxes and text

7. **Main Application** (`main.py`)
   - `EmotionDetectionApp`: Main application orchestrator
   - Event loop and lifecycle management
   - Component integration

### Design Patterns Used

- **Singleton Pattern**: Configuration management
- **Context Manager**: Resource cleanup (video capture, UI)
- **Data Class**: Structured emotion records
- **Dependency Injection**: Components receive config instance
- **Separation of Concerns**: Each module has single responsibility

## 🔍 Technical Details

### Emotion Detection Models

EmotiSense supports multiple state-of-the-art emotion recognition models:

| Model | Accuracy | Speed | Size | Emotions | Best For |
|-------|----------|-------|------|----------|----------|
| **HSEmotion** ⭐ | 66%+ | ~60ms | 16-30MB | 7 or 8 | Production, Real-time |
| **FER** | ~65% | ~150ms | ~5MB | 7 | Lightweight apps |
| **DeepFace** | ~60-65% | ~300ms | ~100MB | 7 | Multi-task analysis |
| **Ensemble** | Highest | Slowest | Combined | 7-8 | Maximum accuracy |

**Supported Emotions:**
- 7-class: angry, disgust, fear, happy, neutral, sad, surprise
- 8-class: + contempt (HSEmotion only)

**Model Details:**
- **HSEmotion**: Pre-trained on VGGFace2 + AffectNet, ABAW competition winner
- **FER**: CNN-based, trained on FER2013 dataset
- **DeepFace**: Hybrid framework with multiple backends
- **Ensemble**: Combines multiple models for better accuracy

See [ADVANCED_MODELS.md](ADVANCED_MODELS.md) for detailed comparison and usage guide.

### Face Detection
- **Method**: OpenCV Haar Cascade Classifiers
- **Features**: Face and eye detection
- **Smoothing**: Exponential smoothing to reduce jitter
- **Parameters**: Configurable scale factor, min/max size

### Data Management
- **Format**: Timestamped emotion records with confidence
- **Storage**: In-memory list with automatic cleanup
- **Logging**: High-confidence emotions to text file
- **Statistics**: Emotion counts, averages, time spans

## 📝 Notes

1. **Camera Access**: Ensure your webcam is accessible and not in use by other applications
2. **Lighting**: Good lighting conditions improve detection accuracy
3. **API Key**: DeepSeek API key is optional; the system works without it (no trend analysis)
4. **Privacy**: All processing is done locally; only emotion logs are sent to API (if enabled)

## 🤝 Contributing

This is a refactored version of an early learning project. Contributions are welcome!

## 📄 License

This project is open source and available for educational purposes.

## 🙏 Acknowledgments

- **DeepFace**: Emotion detection library
- **OpenCV**: Computer vision framework
- **DeepSeek**: AI-powered emotion analysis

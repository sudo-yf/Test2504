# EmotiSense 重构说明

## 概述

这是对原始 EmotiSense 项目的完整重构，采用现代软件工程最佳实践，提高了代码的可维护性、可扩展性和可测试性。

## 主要改进

### 1. 架构改进

#### 之前（单文件架构）
- 所有代码在 `main.py` 中（244行）
- 全局变量和函数
- 难以测试和维护
- 配置硬编码

#### 现在（模块化架构）
```
src/
├── config.py          # 配置管理
├── detector.py        # 检测逻辑
├── video_processor.py # 视频处理
├── data_manager.py    # 数据管理
├── analyzer.py        # API集成
└── ui.py              # UI渲染
```

### 2. 代码质量提升

#### 类型注解
```python
# 之前
def analyze_emotion(face_img):
    ...

# 现在
def analyze_emotion(self, face_img: np.ndarray) -> Tuple[str, float]:
    ...
```

#### 文档字符串
所有类和方法都有详细的文档字符串，说明参数、返回值和功能。

#### 错误处理
- 使用 Python logging 模块替代 print
- 更好的异常处理和错误消息
- 资源清理保证（使用 context managers）

### 3. 配置管理

#### 之前
```python
detection_interval = 3.0  # 硬编码
MAX_EMOTION_DATA = 1000   # 全局常量
```

#### 现在
```yaml
# config.yaml
emotion:
  detection_interval: 3.0
  max_data_records: 1000
```

### 4. 面向对象设计

#### 之前（过程式）
```python
def main():
    cap = cv2.VideoCapture(0)
    emotion_data = []
    # ... 200+ 行代码
```

#### 现在（面向对象）
```python
class EmotionDetectionApp:
    def __init__(self):
        self.video_capture = VideoCapture(config)
        self.data_manager = EmotionDataManager(config)
        # ...
    
    def run(self):
        # 清晰的应用生命周期
```

## 功能对比

| 功能 | 旧版本 | 新版本 |
|------|--------|--------|
| 人脸检测 | ✅ | ✅ |
| 眼睛检测 | ✅ | ✅ |
| 情绪分析 | ✅ | ✅ |
| 高置信度日志 | ✅ | ✅ |
| DeepSeek API | ✅ | ✅ |
| 配置文件 | ❌ | ✅ YAML |
| 类型注解 | ❌ | ✅ |
| 日志系统 | ❌ | ✅ |
| 模块化 | ❌ | ✅ |
| 单元测试支持 | ❌ | ✅ |
| 文档 | 基础 | 完整 |

## 迁移指南

### 如果你想使用旧版本
旧版本的代码已被重构，但功能完全保留。如果需要旧版本，可以从 git 历史中恢复。

### 使用新版本

1. **安装依赖**
```bash
pip install -r requirements.txt
```

2. **配置环境变量**
```bash
cp .env.example .env
# 编辑 .env 添加你的 API key
```

3. **（可选）自定义配置**
编辑 `config.yaml` 调整参数

4. **运行应用**
```bash
python main.py
```

## 配置选项说明

### 视频设置
```yaml
video:
  camera_index: 0        # 摄像头索引
  frame_width: 640       # 帧宽度
  frame_height: 360      # 帧高度
  fps: 30                # 帧率
  frame_skip: 2          # 跳帧数（处理每N帧）
```

### 人脸检测
```yaml
face_detection:
  scale_factor: 1.1      # 缩放因子
  min_neighbors: 3       # 最小邻居数
  min_size: [80, 80]     # 最小人脸尺寸
  max_size: [300, 300]   # 最大人脸尺寸
  smoothing_factor: 0.3  # 平滑因子
```

### 情绪检测
```yaml
emotion:
  detection_interval: 3.0           # 检测间隔（秒）
  high_confidence_threshold: 95     # 高置信度阈值
  anger_threshold: 50               # 愤怒情绪特殊阈值
  max_data_records: 1000            # 最大记录数
```

### UI 设置
```yaml
ui:
  window_name: "情绪检测"
  font_scale: 0.7
  face_box_color: [255, 0, 0]  # BGR 格式
  eye_box_color: [0, 255, 0]
```

## 代码示例

### 使用配置系统
```python
from src.config import get_config

config = get_config()
interval = config.get('emotion.detection_interval', 3.0)
```

### 使用数据管理器
```python
from src.data_manager import EmotionDataManager

manager = EmotionDataManager(config)
record = manager.add_record('happy', 98.5)
stats = manager.get_statistics()
```

### 使用检测器
```python
from src.detector import FaceDetector, EmotionDetector

face_detector = FaceDetector(config)
emotion_detector = EmotionDetector(config)

faces = face_detector.detect_faces(gray_frame)
emotion, confidence = emotion_detector.analyze_emotion(face_img)
```

## 性能对比

| 指标 | 旧版本 | 新版本 |
|------|--------|--------|
| 内存占用 | ~150-200MB | ~150-200MB |
| 启动时间 | 相同 | 相同 |
| 代码行数 | 244行（单文件） | ~800行（模块化） |
| 可维护性 | 低 | 高 |
| 可测试性 | 低 | 高 |
| 可扩展性 | 低 | 高 |

## 扩展建议

新的模块化架构使得扩展变得容易：

1. **添加新的情绪检测模型**
   - 在 `detector.py` 中创建新的检测器类
   - 实现相同的接口

2. **添加新的数据存储方式**
   - 扩展 `data_manager.py`
   - 添加数据库支持等

3. **添加新的 UI 功能**
   - 在 `ui.py` 中添加新的渲染方法
   - 支持图表、统计等

4. **添加单元测试**
   - 每个模块都可以独立测试
   - 使用 pytest 或 unittest

## 常见问题

### Q: 为什么代码行数增加了？
A: 模块化、文档字符串、类型注解和更好的错误处理增加了代码量，但大大提高了可维护性。

### Q: 性能有影响吗？
A: 没有。运行时性能基本相同，内存占用也相同。

### Q: 可以不使用配置文件吗？
A: 可以，配置系统有默认值。但使用配置文件更灵活。

### Q: 如何添加新功能？
A: 找到相关模块，添加新的类或方法。模块化设计使扩展变得简单。

## 总结

这次重构将一个学习项目转变为一个专业的、可维护的应用程序。主要改进包括：

✅ 模块化架构  
✅ 面向对象设计  
✅ 配置管理  
✅ 类型注解  
✅ 完整文档  
✅ 错误处理  
✅ 日志系统  
✅ 可测试性  

代码现在更容易理解、维护和扩展，同时保持了所有原有功能。


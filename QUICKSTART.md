# EmotiSense 快速开始指南

## 5分钟快速上手

### 1️⃣ 安装依赖（1-2分钟）

**基础安装**：
```bash
pip install -r requirements.txt
```

**推荐：安装高性能模型**（可选但强烈推荐）：
```bash
# 使用交互式安装器
python install_models.py

# 或手动安装 HSEmotion（推荐）
pip install hsemotion timm
```

### 2️⃣ 配置环境（可选，1分钟）

如果你想使用 DeepSeek API 进行情绪趋势分析：

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件，添加你的 API key
# DEEPSEEK_API_KEY=your_actual_api_key_here
```

> 💡 **提示**: 没有 API key 也可以运行，只是不会有最终的趋势分析报告。

### 3️⃣ 运行应用（30秒）

```bash
python main.py
```

### 4️⃣ 使用应用

- 📹 应用会自动打开摄像头
- 👤 将你的脸对准摄像头
- 😊 实时查看情绪检测结果
- 🔴 按 `q` 或 `ESC` 退出

### 5️⃣ 查看结果

- **实时显示**: 窗口中显示人脸框、眼睛框和情绪标签
- **日志文件**: `emotion_log.txt` 记录高置信度情绪（>95%）
- **分析报告**: 退出时自动生成（需要 API key）

---

## 测试安装

运行测试脚本验证所有模块：

```bash
python test_modules.py
```

应该看到：
```
🎉 All tests passed! You can run the application with: python main.py
```

---

## 自定义配置（可选）

编辑 `config.yaml` 调整参数：

```yaml
# 调整检测间隔（秒）
emotion:
  detection_interval: 3.0  # 改为 1.0 更频繁检测

# 调整视频分辨率
video:
  frame_width: 640   # 改为 1280 提高清晰度
  frame_height: 360  # 改为 720
```

---

## 常见问题

### ❓ 摄像头无法打开
- 检查摄像头是否被其他应用占用
- 尝试修改 `config.yaml` 中的 `camera_index`（0, 1, 2...）

### ❓ 检测不到人脸
- 确保光线充足
- 调整摄像头角度
- 降低 `config.yaml` 中的 `min_size` 参数

### ❓ 情绪检测不准确
- 保持正面对着摄像头
- 确保面部清晰可见
- 调整 `anger_threshold` 等参数

### ❓ 程序运行缓慢
- 增加 `frame_skip` 值（跳过更多帧）
- 降低分辨率
- 增加 `detection_interval`

---

## 下一步

- 📖 阅读 [README.md](README.md) 了解详细功能
- 🔧 查看 [REFACTORING_NOTES.md](REFACTORING_NOTES.md) 了解架构
- 🎨 自定义 `config.yaml` 调整参数
- 🚀 扩展功能（添加新的检测模型、UI 等）

---

## 获取帮助

遇到问题？

1. 运行 `python test_modules.py` 检查安装
2. 查看日志输出中的错误信息
3. 检查 `emotion_log.txt` 文件
4. 阅读完整文档

---

**享受使用 EmotiSense！** 😊


# 日志分析

## 功能说明

系统在运行过程中会记录高置信度情绪事件，并在应用退出时执行统计与趋势分析。

分析输入来源：

- 内存中的情绪记录序列
- 本地日志文件（`data.log_file`）
- SQLite 数据库（`data.database_path`）

## 配置项

`config.yaml` 中与日志分析相关的关键配置：

- `emotion.high_confidence_threshold`：高置信度阈值
- `data.log_file`：日志文件路径
- `data.database_path`：SQLite 数据库存储路径
- `data.max_records`：内存记录上限
- `data.cleanup_interval`：清理间隔

## 运行产物

- 高置信度事件文本日志
- SQLite `emotion_records` 表
- 退出时生成 `outputs/reports/*.json` 与 `outputs/reports/*.md` 报告

## 命令入口

运行前自检：

```bash
uv run python main.py doctor --check-camera
```

离线生成报告：

```bash
uv run python main.py report --mode offline
```

## API Key（可选）

若设置 `DEEPSEEK_API_KEY`，系统会在退出阶段调用分析接口生成更详细的文本分析结果。

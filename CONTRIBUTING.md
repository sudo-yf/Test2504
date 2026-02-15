# 贡献指南

## 开发环境

```bash
uv sync
cp .env.example .env
```

训练相关依赖：

```bash
uv sync --extra train --extra models --extra dev
```

## 本地检查

```bash
make lint
make test
make check
```

## 提交要求

- 单次 PR 聚焦单一目标
- 修改行为时必须更新 README/docs
- 提供最小复现与验证步骤

## PR 检查清单

- [ ] lint 通过
- [ ] test 通过
- [ ] 文档已更新
- [ ] 无无关文件改动


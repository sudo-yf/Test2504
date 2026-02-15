# GPU Requirements

## Training (Recommended)

- NVIDIA GPU with CUDA support
- CUDA driver/runtime compatible with installed PyTorch build
- VRAM recommendations:
  - Face detector training (YOLO): **8GB+**
  - Emotion classifier fine-tuning: **8GB+**
  - Multi-model experiments / larger batch sizes: **12GB+**

## Inference

- CPU is supported for real-time demo (lower throughput)
- GPU recommended for high-FPS and multi-stream processing

## Notes

- Ensure CUDA toolkit and driver versions are compatible with installed PyTorch.
- For production runs, pin PyTorch/CUDA versions in deployment environment.

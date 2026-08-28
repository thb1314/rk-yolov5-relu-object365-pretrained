# Pin the V1 consumer toolchain

V1 supports official YOLOv5 v6.2 for fine-tuning and aiRockchip YOLOv5 `d25a075` for
RKNN-friendly ONNX export. This exact pair has passed checkpoint loading, training, and
export smoke tests; pinning it prevents silent behavior changes from upstream releases.

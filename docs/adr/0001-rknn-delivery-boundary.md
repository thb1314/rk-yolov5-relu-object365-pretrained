# RKNN delivery boundary

V1 publishes official-format checkpoints and a verified RKNN-friendly ONNX export
path, not universal `.rknn` artifacts. RKNN binaries depend on target SoC, Toolkit2
version, quantization choices, and calibration data, so claiming a generic binary as
portable would be misleading; users perform the final target-specific conversion.

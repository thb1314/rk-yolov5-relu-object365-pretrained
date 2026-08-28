#!/usr/bin/env bash
set -euo pipefail

weights=${1:?usage: export_rknn_onnx.sh WEIGHTS ROCKCHIP_YOLOV5 OUTPUT_DIR}
rockchip_repo=${2:?usage: export_rknn_onnx.sh WEIGHTS ROCKCHIP_YOLOV5 OUTPUT_DIR}
output_dir=${3:?usage: export_rknn_onnx.sh WEIGHTS ROCKCHIP_YOLOV5 OUTPUT_DIR}

test -f "$weights"
test -f "$rockchip_repo/export.py"
mkdir -p "$output_dir"
local_weights="$output_dir/$(basename "$weights")"
cp "$weights" "$local_weights"

cd "$output_dir"
python "$rockchip_repo/export.py" --rknpu --weights "$local_weights" --include onnx --imgsz 640 --device cpu

onnx_file="${local_weights%.pt}.onnx"
anchors_file="$(dirname "$onnx_file")/RK_anchors.txt"
test -f "$onnx_file"
test -f "$anchors_file"

python - "$onnx_file" <<'PY'
import sys
import onnx
model = onnx.load(sys.argv[1])
onnx.checker.check_model(model)
print(f"ONNX checker passed: {sys.argv[1]}")
PY

printf 'RKNN-friendly ONNX export: %s\n' "$output_dir/$(basename "$onnx_file")"

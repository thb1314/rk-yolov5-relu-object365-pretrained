#!/usr/bin/env bash
set -euo pipefail

weights=${1:?usage: export_rknn_onnx.sh WEIGHTS ROCKCHIP_YOLOV5 OUTPUT_DIR}
rockchip_repo=${2:?usage: export_rknn_onnx.sh WEIGHTS ROCKCHIP_YOLOV5 OUTPUT_DIR}
output_dir=${3:?usage: export_rknn_onnx.sh WEIGHTS ROCKCHIP_YOLOV5 OUTPUT_DIR}

test -f "$weights"
test -f "$rockchip_repo/export.py"
mkdir -p "$output_dir"

cd "$rockchip_repo"
python export.py --rknpu --weights "$weights" --include onnx --imgsz 640 --device cpu

onnx_file="${weights%.pt}.onnx"
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

cp "$onnx_file" "$anchors_file" "$output_dir/"
printf 'RKNN-friendly ONNX export: %s\n' "$output_dir/$(basename "$onnx_file")"

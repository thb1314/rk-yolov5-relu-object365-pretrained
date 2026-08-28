#!/usr/bin/env bash
set -euo pipefail

weights=${1:?usage: smoke_finetune_official.sh /weights/model.pt}
workspace=${WORKSPACE:-/workspace}
subset_dir="$workspace/outputs/coco-smoke"

test -f "$weights"
test -d /data/coco/images/train2017
test -d /data/coco/images/val2017

python "$workspace/scripts/create_coco_smoke_subset.py" \
  --coco /data/coco --output "$subset_dir" --train-count 4 --val-count 2

cd /opt/yolov5
python train.py \
  --weights "$weights" \
  --data "$subset_dir/coco-smoke.yaml" \
  --epochs 1 --batch-size 2 --imgsz 64 --workers 0 --device cpu --noval \
  --project "$workspace/outputs/official-finetune-smoke" --name "$(basename "${weights%.pt}")" \
  --exist-ok

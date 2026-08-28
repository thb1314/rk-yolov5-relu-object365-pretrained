# Verification Record

Date: 2026-08-28

## Official YOLOv5 Fine-tuning

Environment: clean Docker image based on `pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime`,
official Ultralytics YOLOv5 v6.2, CPU execution.

Command shape:

```bash
bash scripts/smoke_finetune_official.sh \
  /weights/yolov5n_relu_coco_best_epoch300.pt
```

The test used four COCO train images, two COCO validation images, one epoch, image size
64, batch size 2, and zero data-loader workers. It completed successfully and reported:

```text
Transferred 349/349 items from /weights/yolov5n_relu_coco_best_epoch300.pt
1 epochs completed
```

The run wrote official YOLOv5 `last.pt` and `best.pt`, then performed the normal final
validation pass. This confirms that the release checkpoint loads as a pretrained weight
in the upstream training script.

## Rockchip ONNX Export

Environment: the same image, Rockchip YOLOv5 fork commit `d25a075`.

```bash
bash scripts/export_rknn_onnx.sh \
  /weights/yolov5n_relu_coco_best_epoch300.pt \
  /opt/yolov5-airockchip /workspace/outputs/rknn
```

The command completed successfully and generated:

- `yolov5n_relu_coco_best_epoch300.onnx` (7,501,403 bytes)
- `RK_anchors.txt` (96 bytes)

`onnx.checker.check_model` passed for the exported ONNX graph. This proves compatibility
with the Rockchip-friendly ONNX path. Producing a target `.rknn` still requires the
target-SoC RKNN Toolkit2 environment and, where appropriate, calibration data.

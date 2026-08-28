# RK YOLOv5 ReLU Objects365 Pretrained Weights

面向 RKNN/Rockchip 开发者的 YOLOv5 ReLU 预训练权重与微调起点。权重来自
MMYOLO 训练 checkpoint，并严格转换为官方 Ultralytics YOLOv5 格式；用户侧只需
使用官方 YOLOv5 微调，再通过 Rockchip YOLOv5 导出 RKNN-friendly ONNX。

所有模型均使用 **ReLU**，不是官方 YOLOv5 默认的 SiLU。加载 Release 权重时会保留
该激活函数，无需手动修改模型结构。

## 权重

| 模型 | Objects365 权重 | COCO 权重 |
| --- | --- | --- |
| YOLOv5-N ReLU | [下载](https://github.com/thb1314/rk-yolov5-relu-object365-pretrained/releases/download/v0.1.0/yolov5n_relu_objects365_best_epoch100.pt) | [下载](https://github.com/thb1314/rk-yolov5-relu-object365-pretrained/releases/download/v0.1.0/yolov5n_relu_coco_best_epoch300.pt) |
| YOLOv5-S ReLU | X | [下载](https://github.com/thb1314/rk-yolov5-relu-object365-pretrained/releases/download/v0.1.0/yolov5s_relu_coco_best_epoch300.pt) |
| YOLOv5-M ReLU | [下载](https://github.com/thb1314/rk-yolov5-relu-object365-pretrained/releases/download/v0.1.0/yolov5m_relu_objects365_best_epoch100.pt) | [下载](https://github.com/thb1314/rk-yolov5-relu-object365-pretrained/releases/download/v0.1.0/yolov5m_relu_coco_best_epoch290.pt) |
| YOLOv5-L ReLU | X | [下载](https://github.com/thb1314/rk-yolov5-relu-object365-pretrained/releases/download/v0.1.0/yolov5l_relu_coco_best_epoch255.pt) |

- Objects365 权重为 365 类检测头；COCO 权重为 80 类检测头。
- Release 资产：[v0.1.0](https://github.com/thb1314/rk-yolov5-relu-object365-pretrained/releases/tag/v0.1.0)。下载后使用其中的 `SHA256SUMS` 校验，`models.json` 提供机器可读的模型元数据。

```bash
sha256sum -c SHA256SUMS
```

<details>
<summary>训练记录</summary>

| 模型 | 数据集 | epoch | batch | 精度 | 增强路径 | 起始权重 |
| --- | --- | ---: | --- | --- | --- | --- |
| YOLOv5-N | Objects365 | 100 | 4 x 64 | FP32 | MMYOLO YOLOv5 标准 CPU Mosaic | scratch |
| YOLOv5-M | Objects365 | 100 (best) | 2 x 64, accum 2 | BF16 | GPU decode + batch GPU augment | scratch |
| YOLOv5-N | COCO | 300 | 1 x 256 | FP32 | MMYOLO YOLOv5 标准 CPU Mosaic | Objects365-N e100 |
| YOLOv5-S | COCO | 300 | 2 x 128 | FP32 | MMYOLO YOLOv5 标准 CPU Mosaic | Objects365-S e100 |
| YOLOv5-M | COCO | 290 (best) | 4 x 64, accum 2 | BF16 | GPU decode + batch GPU augment | Objects365-M e200 |
| YOLOv5-L | COCO | 255 (best) | 1 x 64 | BF16 | GPU decode + batch GPU augment | Objects365-L e100 |

</details>

## 1. 官方 YOLOv5 微调

已验证的用户训练入口为 [Ultralytics YOLOv5 v6.2](https://github.com/ultralytics/yolov5/tree/v6.2)。
将 `--weights` 换成上表中与你的任务最接近的权重，并将 `--data` 指向你的 YOLO 数据集 YAML。

```bash
git clone --branch v6.2 --depth 1 https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt

python train.py \
  --weights /path/to/yolov5n_relu_objects365_best_epoch100.pt \
  --data /path/to/your_dataset.yaml \
  --epochs 100 --img 640 --batch 16
```

N 权重已在干净环境的官方 v6.2 `train.py` 中完成一轮训练 smoke，日志显示
`Transferred 349/349 items`。完整记录见 [验证记录](docs/VERIFICATION.md)。

## 2. Rockchip ONNX 导出

已验证导出端为 [aiRockchip YOLOv5](https://github.com/airockchip/yolov5) commit
`d25a075`。仓库脚本调用其 `--rknpu` 模式，生成 ONNX 与 `RK_anchors.txt`：

```bash
bash scripts/export_rknn_onnx.sh \
  /path/to/yolov5n_relu_coco_best_epoch300.pt \
  /path/to/yolov5-airockchip \
  /path/to/output
```

该路径已通过 ONNX checker。输出的 ONNX 与 `RK_anchors.txt` 是 RKNN Toolkit2 的输入；
不同 SoC、Toolkit2 版本和量化校准数据需要在目标环境中完成最终 `.rknn` 转换与验证。

## 转换与验证

[`tools/convert_mmyolo_yolov5_to_ultralytics.py`](tools/convert_mmyolo_yolov5_to_ultralytics.py)
是本 Release 使用的严格转换器，支持 YOLOv5 P5 N/S/M/L。未映射 tensor 或形状不一致会
直接失败。转换一致性与干净 Docker 验证记录见 [MODEL_CARD.md](MODEL_CARD.md) 和
[docs/VERIFICATION.md](docs/VERIFICATION.md)。

## 许可证

仓库的转换、集成和文档以 [GPL-3.0-only](LICENSE) 发布。COCO 与 Objects365 数据集的
使用仍分别受其原始条款约束。

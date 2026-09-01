# RK YOLOv5 ReLU Objects365 Pretrained Weights

> 打个广告：模型部署优化、模型加速（云端、端侧、边缘侧）、计算机视觉相关需求欢迎联系 zhuilewang@163.com
> 面向行业与厂家：工业质检、表面缺陷检测、SOP 行为检测等

## 项目初衷

当前 YOLO 模型的发展越来越重视 GPU 推理速度与模型精度之间的平衡。然而在 NPU 端，
尤其是国产瑞芯微系列 NPU 上，许多面向 GPU 设计的新型 YOLO 网络结构并不能直接转化为
更高的端侧效率。随着 YOLO 版本不断迭代，模型结构日益复杂，实际部署在瑞芯微 NPU 上的
吞吐反而可能下降。在我们关注的 YOLO Small 规格中，YOLOv5 依然具有突出的推理速度优势。

但为了适配 NPU 而将激活函数替换为 ReLU 时，由于缺少大型数据集预训练，往往会造成一定的
精度损失。为此，本项目以 YOLOv5 为基础，将 SiLU 替换为 ReLU 激活函数，并在 Objects365
和 COCO 数据集上从零开始依次进行预训练，以重新建立更适合 NPU 部署的高精度基础模型。

我们开源了 YOLOv5-ReLU Nano、Small、Medium 和 Large 四种规格的预训练权重。开发者可以
基于这些权重继续微调自己的业务数据，在保留 YOLOv5-ReLU 高效 NPU 推理能力的同时，获得
更好的检测精度，实现端侧速度与精度的双重收益。

面向 RKNN/Rockchip 开发者的 YOLOv5 ReLU 预训练权重与微调起点。权重来自
MMYOLO 训练 checkpoint，并严格转换为官方 Ultralytics YOLOv5 格式；用户侧只需
使用官方 YOLOv5 微调，再通过 Rockchip YOLOv5 导出 RKNN-friendly ONNX。

所有模型均使用 **ReLU**，不是官方 YOLOv5 默认的 SiLU。加载 Release 权重时会保留
该激活函数，无需手动修改模型结构。

## 权重

| 模型 | Objects365 权重 | COCO 权重 | COCO mAP@0.5:0.95 | COCO mAP@0.5 |
| --- | --- | --- | ---: | ---: |
| YOLOv5-N ReLU | [下载](https://github.com/thb1314/rk-yolov5-relu-object365-pretrained/releases/download/v0.1.0/yolov5n_relu_objects365_best_epoch100.pt) | [下载](https://github.com/thb1314/rk-yolov5-relu-object365-pretrained/releases/download/v0.1.0/yolov5n_relu_coco_best_epoch300.pt) | 25.7% | 42.7% |
| YOLOv5-S ReLU | X | [下载](https://github.com/thb1314/rk-yolov5-relu-object365-pretrained/releases/download/v0.1.0/yolov5s_relu_coco_best_epoch300.pt) | 37.0% | 56.3% |
| YOLOv5-M ReLU | [下载](https://github.com/thb1314/rk-yolov5-relu-object365-pretrained/releases/download/v0.1.0/yolov5m_relu_objects365_best_epoch100.pt) | [下载](https://github.com/thb1314/rk-yolov5-relu-object365-pretrained/releases/download/v0.1.0/yolov5m_relu_coco_best_epoch290.pt) | 44.8% | 63.5% |
| YOLOv5-L ReLU | X | [下载](https://github.com/thb1314/rk-yolov5-relu-object365-pretrained/releases/download/v0.1.0/yolov5l_relu_coco_best_epoch255.pt) | 47.9% | 66.4% |

- Objects365 权重为 365 类检测头；COCO 权重为 80 类检测头。
- COCO 指标来自 MMYOLO 在 `COCO val2017` 上对最佳 checkpoint 的评测；`mAP@0.5:0.95`
  为 COCO AP，`mAP@0.5` 为 AP50。YOLOv5-N 转为官方格式后使用 Ultralytics YOLOv5
  v6.2 复评得到 25.6% / 42.7%，与训练日志中的 25.7% / 42.7% 基本一致。
- Release 资产：[v0.1.0](https://github.com/thb1314/rk-yolov5-relu-object365-pretrained/releases/tag/v0.1.0)。下载后使用其中的 `SHA256SUMS` 校验，`models.json` 提供机器可读的模型元数据。

```bash
sha256sum -c SHA256SUMS
```

<details>
<summary>训练记录</summary>

| 模型 | 数据集 | epoch | batch | 训练精度 | 增强路径 | 起始权重 |
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

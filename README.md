# RK YOLOv5 Pretrained Weights

COCO-pretrained YOLOv5 P5 detection weights for fine-tuning and Rockchip RKNN deployment.
The release files are converted from MMYOLO training checkpoints into the official
Ultralytics YOLOv5 checkpoint format. They use **ReLU**, not the upstream default
SiLU; this is intentional and the converted checkpoints preserve that activation.

## Models

| Model | COCO training | Asset | SHA256 |
| --- | --- | --- | --- |
| YOLOv5-N ReLU | 300 epochs | `yolov5n_relu_coco_best_epoch300.pt` | `af918e856d43f166b5611be6c107a2deb916c8a9ba7767dcfd515780e687c877` |
| YOLOv5-S ReLU | 300 epochs | `yolov5s_relu_coco_best_epoch300.pt` | `a667e70170f444b2078fdd12de74e24b22439d0ddf846058ab46ca95b5283570` |
| YOLOv5-M ReLU | 290 epochs (best) | `yolov5m_relu_coco_best_epoch290.pt` | `6510733b15a93478a3d0388dff1ad6a5c09d39775662f7bc6299ed2aec6ab2df` |
| YOLOv5-L ReLU | 255 epochs (best) | `yolov5l_relu_coco_best_epoch255.pt` | `2c66e03f278b0e26ad7188fd03e141d6824932ff20861c7e1dd4ef58f660a718` |

Download a release asset and verify it with the accompanying `SHA256SUMS` file:

```bash
sha256sum -c SHA256SUMS
```

## Official YOLOv5 Fine-tuning

The checkpoints target [Ultralytics YOLOv5 v6.2](https://github.com/ultralytics/yolov5/tree/v6.2).
They are ordinary official-format `.pt` checkpoints and can be used by the upstream
scripts. Use the image supplied here, or install the v6.2 requirements locally.

```bash
git clone --branch v6.2 --depth 1 https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt
python train.py --weights /path/to/yolov5n_relu_coco_best_epoch300.pt \
  --data /path/to/your_dataset.yaml --epochs 100 --img 640 --batch 16
```

The supplied `scripts/smoke_finetune_official.sh` makes a tiny COCO subset and runs a
one-epoch official-training loading test. It is intentionally a wiring check, not a
meaningful accuracy run.

## Docker Smoke Test

```bash
docker build -t rk-yolov5-pretrained:smoke -f docker/Dockerfile .
docker run --rm \
  -v /path/to/coco:/data/coco:ro \
  -v /path/to/downloaded-weights:/weights:ro \
  -v "$PWD":/workspace \
  rk-yolov5-pretrained:smoke \
  bash scripts/smoke_finetune_official.sh /weights/yolov5n_relu_coco_best_epoch300.pt
```

`/path/to/coco` must contain `images/train2017`, `images/val2017`, and COCO labels.
The smoke test needs only a few images and runs on CPU by default.

## Rockchip RKNN

The local Rockchip fork is based on YOLOv5 v7.0 and accepts these official-format
checkpoints. Its `--rknpu` mode exports an RKNN-friendly ONNX graph and writes
`RK_anchors.txt`; it does **not** itself produce a `.rknn` file. Use the target-SoC
version of RKNN Toolkit2 / RKNN Model Zoo to compile that ONNX graph into `.rknn`.

```bash
git clone https://github.com/airockchip/yolov5.git external/yolov5-airockchip
bash scripts/export_rknn_onnx.sh \
  /weights/yolov5n_relu_coco_best_epoch300.pt \
  external/yolov5-airockchip /workspace/outputs/rknn
```

The export script checks the ONNX model and requires `RK_anchors.txt` to be produced.

## Conversion

`tools/convert_mmyolo_yolov5_to_ultralytics.py` is the strict MMYOLO-to-official
converter used for these assets. It supports P5 YOLOv5 N/S/M/L models and fails on
unmapped or shape-mismatched tensors. See [MODEL_CARD.md](MODEL_CARD.md) for the
conversion and validation evidence.

## License

The converter and integration material are distributed under [GPL-3.0-only](LICENSE),
to remain compatible with the official YOLOv5 codebase. Dataset terms remain governed
by their respective dataset providers.

# Model Card

## Scope

These are YOLOv5 P5, 80-class COCO object detectors trained in MMYOLO and converted
losslessly to the official Ultralytics YOLOv5 checkpoint layout. All models use ReLU
activations. This differs from standard upstream YOLOv5 models, which default to SiLU.

| Model | Best COCO epoch | File size | Source training shape |
| --- | ---: | ---: | --- |
| N | 300 | 7.4 MiB | 1 x 256 |
| S | 300 | 28 MiB | 2 x 128 |
| M | 290 | 82 MiB | COCO best checkpoint |
| L | 255 | 179 MiB | 1 x 64, BF16 + GPU augmentation |

## Conversion Fidelity

The converter reverses MMYOLO's official YOLOv5 checkpoint mapping and reconstructs
the model with ReLU before strict loading. It uses the saved EMA `state_dict`, which
is what MMEngine's `EMAHook` writes into an evaluation checkpoint.

For YOLOv5-N, the converted model was validated with the official v6.2 code:

| Evaluation | MMYOLO source | Converted official checkpoint |
| --- | ---: | ---: |
| COCO AP | 0.257 | 0.256 |
| COCO AP50 | 0.427 | 0.427 |

P3/P4/P5 raw outputs had `max_abs=0` on seeded tensors and on a real COCO image after
matching preprocessing. The 0.001 AP reporting difference is within independent
evaluator rounding/implementation variation; AP50 matches exactly.

The remaining release checkpoints use the same strict architecture-aware conversion.

## Intended Use

Use the models as initialization for fine-tuning on a downstream detection dataset,
or export them through the Rockchip-friendly ONNX path before running RKNN Toolkit2.
Validate accuracy on the target data and hardware before deployment.

## Limitations

- COCO labels and image distributions may not resemble the deployment domain.
- ReLU is a model property. Do not rebuild a SiLU YAML model and load these weights
  while expecting identical behavior.
- The Rockchip ONNX export is only the first deployment step; target-specific RKNN
  conversion and calibration remain necessary.

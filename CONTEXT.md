# Context

## Terms

**Target user**
: A developer who fine-tunes an object detector for, exports to, or deploys on a
  Rockchip RKNN/NPU workflow.

**Release model**
: An official YOLOv5-format ReLU checkpoint published as a GitHub Release asset.
  It is identified by architecture, source dataset, class count, and selected epoch.

**COCO model**
: A Release model whose detection head has 80 COCO classes. It is a general
  fine-tuning starting point and a compatibility reference.

**Objects365 model**
: A Release model whose detection head has 365 Objects365 classes. It is a broader
  pretraining starting point and must not be treated as an 80-class COCO checkpoint.

**RKNN-friendly ONNX**
: An ONNX model exported by the Rockchip YOLOv5 fork with `--rknpu`, accompanied by
  `RK_anchors.txt`. It is an input to RKNN Toolkit2, not itself a `.rknn` model.

**Primary training interface**
: The official Ultralytics YOLOv5 v6.2 `train.py` workflow used by Release consumers.
  MMYOLO is retained only as the provenance of training and conversion records.

**Repository license**
: GPL-3.0-only for this repository's scripts, Docker integration, and documentation.
  COCO and Objects365 dataset terms remain separate from this code license.

**V1 release**
: The mutable initial GitHub Release tag `v0.1.0`. Before it is announced, its assets
  and manifests are corrected in place so one tag describes one self-consistent set.

**Availability matrix**
: The README table with the columns Model, Objects365 weight, and COCO weight. A
  downloadable Release asset is linked by name; an unavailable combination is shown as
  `X` without a narrative about its history.

**Training record**
: A collapsed README table containing only model, dataset, epochs, batch, precision,
  augmentation path, and initialization weight. It describes provenance and is not a
  second, user-facing MMYOLO training workflow.

**README language**
: Chinese-first documentation for the initial release. Commands, asset filenames, and
  upstream project names remain in their original English forms.

**Release companion assets**
: `SHA256SUMS` for download verification and `models.json` for machine-readable model
  metadata accompany the six checkpoint assets in V1.

**Supported toolchain**
: Official Ultralytics YOLOv5 v6.2 for consumer fine-tuning and aiRockchip YOLOv5
  commit `d25a075` for RKNN-friendly ONNX export.

**V1 deployment claim**
: RKNN-friendly ONNX export has been verified. V1 makes no claim about a target-SoC
  `.rknn` binary, NPU latency, INT8 accuracy, or on-device performance.

**Primary user paths**
: The two README workflows are fine-tuning with official YOLOv5 and exporting
RKNN-friendly ONNX with the pinned Rockchip fork. MMYOLO and RKNN Toolkit2 details
are outside the V1 getting-started path.

## Scope

Version 1 is a set of COCO and Objects365 ReLU pretrained weights and fine-tuning
starting points for RKNN/Rockchip users. It provides conversion, official YOLOv5
fine-tuning, and Rockchip-friendly ONNX export guidance; target-SoC RKNN conversion
and production accuracy validation remain downstream user responsibilities.

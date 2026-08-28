#!/usr/bin/env python3
"""Convert a MMYOLO YOLOv5 P5 checkpoint into an Ultralytics YOLOv5 v6.2 .pt.

The mapping is the exact inverse of MMYOLO's official-to-MMYOLO converter.
The destination is constructed from the official model first, so every
parameter and buffer must map one-to-one before a checkpoint is written.
"""

import argparse
import copy
import sys
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn


P5_MAP = {
    'model.0': 'backbone.stem',
    'model.1': 'backbone.stage1.0',
    'model.2': 'backbone.stage1.1',
    'model.3': 'backbone.stage2.0',
    'model.4': 'backbone.stage2.1',
    'model.5': 'backbone.stage3.0',
    'model.6': 'backbone.stage3.1',
    'model.7': 'backbone.stage4.0',
    'model.8': 'backbone.stage4.1',
    'model.9.cv1': 'backbone.stage4.2.conv1',
    'model.9.cv2': 'backbone.stage4.2.conv2',
    'model.10': 'neck.reduce_layers.2',
    'model.13': 'neck.top_down_layers.0.0',
    'model.14': 'neck.top_down_layers.0.1',
    'model.17': 'neck.top_down_layers.1',
    'model.18': 'neck.downsample_layers.0',
    'model.20': 'neck.bottom_up_layers.0',
    'model.21': 'neck.downsample_layers.1',
    'model.23': 'neck.bottom_up_layers.1',
    'model.24.m': 'bbox_head.head_module.convs_pred',
}


def official_to_mmyolo_key(key: str) -> str:
    """Apply MMYOLO's published YOLOv5 key conversion to one official key."""
    parts = key.split('.')
    if len(parts) < 3 or parts[0] != 'model':
        raise ValueError(f'Unexpected official YOLOv5 state-dict key: {key}')

    number, module = parts[1], parts[2]
    prefix = f'model.{number}.{module}' if number in ('9', '24') else f'model.{number}'
    try:
        converted = key.replace(prefix, P5_MAP[prefix], 1)
    except KeyError as exc:
        raise KeyError(f'No P5 mapping for {prefix} (from {key})') from exc

    if '.m.' in converted:
        converted = converted.replace('.m.', '.blocks.')
        converted = converted.replace('.cv', '.conv')
    else:
        converted = converted.replace('.cv1', '.main_conv')
        converted = converted.replace('.cv2', '.short_conv')
        converted = converted.replace('.cv3', '.final_conv')
    return converted


def replace_silu_with_relu(module: nn.Module) -> None:
    """MMYOLO checkpoints here were trained with ReLU, unlike official defaults."""
    for name, child in module.named_children():
        if isinstance(child, nn.SiLU):
            setattr(module, name, nn.ReLU(inplace=True))
        else:
            replace_silu_with_relu(child)


def load_mmyolo_state(path: Path, use_ema: bool) -> OrderedDict:
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    key = 'ema_state_dict' if use_ema else 'state_dict'
    if key not in checkpoint or not isinstance(checkpoint[key], dict):
        raise KeyError(f'{path} does not contain a {key!r} dictionary')
    state = checkpoint[key]
    if not use_ema:
        return state

    # MMEngine's EMA wrapper stores ``steps`` plus ``module.``-prefixed
    # tensors. Expose it in the same key space as the regular model state.
    normalized = OrderedDict()
    for name, value in state.items():
        if name == 'steps':
            continue
        if not name.startswith('module.'):
            raise KeyError(f'Unexpected EMA key in {path}: {name}')
        normalized[name[7:]] = value
    return normalized


def build_official_model(repo: Path, cfg: Path, num_classes: int) -> nn.Module:
    repo = repo.resolve()
    if not (repo / 'models' / 'yolo.py').is_file():
        raise FileNotFoundError(f'Not an Ultralytics YOLOv5 checkout: {repo}')
    sys.path.insert(0, str(repo))
    from models.yolo import Model  # pylint: disable=import-outside-toplevel

    model = Model(str(cfg.resolve()), nc=num_classes)
    replace_silu_with_relu(model)
    return model


def convert(args: argparse.Namespace) -> None:
    source_state = load_mmyolo_state(args.src, args.use_ema)
    model = build_official_model(args.yolov5_repo, args.cfg, args.num_classes)
    destination_state = model.state_dict()
    mapped_state = OrderedDict()
    missing_source = []
    mismatched_shape = []

    for official_key, destination_tensor in destination_state.items():
        # MMYOLO keeps anchor geometry in the prior generator rather than its
        # checkpoint state dict. The official model initialized this buffer
        # from the same YAML anchors, so retain that deterministic value.
        if official_key == 'model.24.anchors':
            mapped_state[official_key] = destination_tensor
            continue
        mmyolo_key = official_to_mmyolo_key(official_key)
        source_tensor = source_state.get(mmyolo_key)
        if source_tensor is None:
            missing_source.append((official_key, mmyolo_key))
            continue
        if tuple(source_tensor.shape) != tuple(destination_tensor.shape):
            mismatched_shape.append(
                (official_key, tuple(destination_tensor.shape), mmyolo_key,
                 tuple(source_tensor.shape)))
            continue
        mapped_state[official_key] = source_tensor.detach().cpu().clone()

    expected_source = {
        official_to_mmyolo_key(key) for key in destination_state
        if key != 'model.24.anchors'
    }
    extra_source = sorted(set(source_state) - expected_source)
    if missing_source or mismatched_shape or extra_source:
        details = [
            f'missing source tensors: {len(missing_source)}',
            f'shape mismatches: {len(mismatched_shape)}',
            f'unmapped source tensors: {len(extra_source)}',
        ]
        for official_key, mmyolo_key in missing_source[:5]:
            details.append(f'  missing {mmyolo_key} for {official_key}')
        for item in mismatched_shape[:5]:
            details.append(f'  shape mismatch: {item}')
        for key in extra_source[:5]:
            details.append(f'  extra source tensor: {key}')
        raise RuntimeError('\n'.join(details))

    result = model.load_state_dict(mapped_state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f'Unexpected load result: {result}')

    model.float().eval()
    # YOLOv5 v6.2's val.py reads this top-level compatibility attribute,
    # while DetectionModel itself stores the class count in Detect.
    model.nc = args.num_classes
    checkpoint = {
        'epoch': -1,
        'best_fitness': None,
        'model': copy.deepcopy(model),
        'ema': None,
        'updates': None,
        'optimizer': None,
        'wandb_id': None,
        'date': None,
        'source_checkpoint': str(args.src.resolve()),
        'source_state_key': 'ema_state_dict' if args.use_ema else 'state_dict',
        'activation': 'ReLU',
    }
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.dst)
    print(f'Converted {len(mapped_state)} tensors to {args.dst}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=Path, required=True)
    parser.add_argument('--dst', type=Path, required=True)
    parser.add_argument('--cfg', type=Path, required=True,
                        help='Official YOLOv5 P5 yaml, e.g. models/yolov5n.yaml')
    parser.add_argument('--yolov5-repo', type=Path, required=True)
    parser.add_argument('--num-classes', type=int, required=True)
    parser.add_argument('--use-ema', action='store_true',
                        help='Use MMEngine ema_state_dict instead of state_dict')
    return parser.parse_args()


if __name__ == '__main__':
    convert(parse_args())

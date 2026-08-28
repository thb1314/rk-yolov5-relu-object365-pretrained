# 模型卡

## 范围

本仓库发布 YOLOv5 P5 ReLU 检测权重，覆盖 COCO 80 类与 Objects365 365 类。所有
Release checkpoint 均由 MMYOLO checkpoint 严格转换为官方 Ultralytics YOLOv5 格式。

| 数据集 | 模型 | Release 训练结果 |
| --- | --- | --- |
| COCO | N / S / M / L | N e300, S e300, M e290, L e255 |
| Objects365 | N / M | N e100, M e100 |

## 转换一致性

转换器反转 MMYOLO 官方 YOLOv5 checkpoint 映射，在模型构建时将默认 SiLU 替换为 ReLU，
再执行严格 state dict 加载。checkpoint 保存的 `state_dict` 已按 MMEngine `EMAHook`
语义包含评估所用的 EMA 值。

对 YOLOv5-N COCO 权重，官方 YOLOv5 v6.2 的 COCO 评估结果为 AP 0.256、AP50 0.427；
MMYOLO 原日志为 0.257、0.427。随机张量与真实 COCO 图片的 P3/P4/P5 原始输出
`max_abs=0`。详情见 [验证记录](docs/VERIFICATION.md)。

## 限制

- ReLU 是模型属性，不能将权重加载进默认 SiLU 模型后期待相同行为。
- COCO 与 Objects365 的检测头类别数不同；下游任务须按自己的数据 YAML 重新适配。
- RKNN-friendly ONNX 导出已验证；目标 SoC 的 `.rknn`、量化精度和部署性能需在目标环境验证。

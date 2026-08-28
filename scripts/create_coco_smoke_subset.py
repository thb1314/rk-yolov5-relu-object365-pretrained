#!/usr/bin/env python3
"""Create list-based, tiny COCO subsets without copying images."""
import argparse
from pathlib import Path


def collect(images_dir: Path, count: int) -> list[Path]:
    images = sorted(images_dir.glob("*.jpg"))[:count]
    if len(images) < count:
        raise RuntimeError(f"need {count} images under {images_dir}, found {len(images)}")
    return images


def write_list(path: Path, images: list[Path]) -> None:
    path.write_text("".join(f"{image}\n" for image in images), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=4)
    parser.add_argument("--val-count", type=int, default=2)
    args = parser.parse_args()

    train_dir = args.coco / "images" / "train2017"
    val_dir = args.coco / "images" / "val2017"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise RuntimeError("expected COCO images/train2017 and images/val2017")

    args.output.mkdir(parents=True, exist_ok=True)
    write_list(args.output / "train.txt", collect(train_dir, args.train_count))
    write_list(args.output / "val.txt", collect(val_dir, args.val_count))
    (args.output / "coco-smoke.yaml").write_text(
        "path: /data/coco\n"
        f"train: {args.output / 'train.txt'}\n"
        f"val: {args.output / 'val.txt'}\n"
        "nc: 80\n"
        "names: [person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic_light, fire_hydrant, stop_sign, parking_meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports_ball, kite, baseball_bat, baseball_glove, skateboard, surfboard, tennis_racket, bottle, wine_glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot_dog, pizza, donut, cake, chair, couch, potted_plant, bed, dining_table, toilet, tv, laptop, mouse, remote, keyboard, cell_phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy_bear, hair_drier, toothbrush]\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

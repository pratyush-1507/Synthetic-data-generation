"""Stage 5 - Convert COCO annotations to an Ultralytics YOLO detection dataset.

Writes, for a dataset directory containing coco_data/:
    <dataset>/yolo/images/{train,val,test}/*.jpg   (symlinks to coco_data/images)
    <dataset>/yolo/labels/{train,val,test}/*.txt   (class cx cy w h, normalised)
    <dataset>/yolo/dataset.yaml
YOLO class ids are COCO category id - 1.

Synthetic datasets are split by *scene* (both views of a scene land in the same
split, so there is no leakage). The split depends only on the image ids and the
seed, so datasets A and B get exactly the same train/val/test scenes.

Usage:
    python src/coco_to_yolo.py --dataset data/synthetic/B_deformed
    python src/coco_to_yolo.py --dataset data/real/taco --all-test     # real test set
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from config import load_config


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="directory containing coco_data/coco_annotations.json")
    ap.add_argument("--seed", type=int, default=cfg["seed"])
    ap.add_argument("--group-size", type=int, default=cfg["render"]["views_per_scene"],
                    help="consecutive images that belong to one scene")
    ap.add_argument("--all-test", action="store_true", help="put every image in the test split (real data)")
    ap.add_argument("--min-area", type=float, default=16, help="skip boxes smaller than this (px^2)")
    args = ap.parse_args()

    root = Path(args.dataset)
    coco_path = root / "coco_data" / "coco_annotations.json"
    out = root / "yolo"
    coco = json.loads(coco_path.read_text())
    cats = sorted(coco["categories"], key=lambda c: c["id"])
    cat_to_yolo = {c["id"]: i for i, c in enumerate(cats)}
    anns = defaultdict(list)
    for a in coco["annotations"]:
        anns[a["image_id"]].append(a)

    images = sorted(coco["images"], key=lambda im: im["id"])
    if args.all_test:
        splits = {"test": images}
    else:
        groups = sorted({im["id"] // args.group_size for im in images})
        random.Random(args.seed).shuffle(groups)
        frac = cfg["render"]["split"]
        n_val, n_test = round(len(groups) * frac["val"]), round(len(groups) * frac["test"])
        group_split = {g: "val" for g in groups[:n_val]}
        group_split.update({g: "test" for g in groups[n_val:n_val + n_test]})
        splits = {"train": [], "val": [], "test": []}
        for im in images:
            splits[group_split.get(im["id"] // args.group_size, "train")].append(im)

    n_boxes = 0
    for split, ims in splits.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for im in ims:
            src = (coco_path.parent / im["file_name"]).resolve()
            link = out / "images" / split / f"{im['id']:06d}{src.suffix}"
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(src)
            W, H = im["width"], im["height"]
            lines = []
            for a in anns[im["id"]]:
                x, y, w, h = a["bbox"]
                if w * h < args.min_area or a["category_id"] not in cat_to_yolo:
                    continue
                lines.append(f"{cat_to_yolo[a['category_id']]} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} "
                             f"{w / W:.6f} {h / H:.6f}")
            (out / "labels" / split / f"{im['id']:06d}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            n_boxes += len(lines)

    yaml = [f"path: {out.resolve()}"]
    for split in ("train", "val", "test"):
        if split in splits:
            yaml.append(f"{split}: images/{split}")
    if "train" not in splits:  # Ultralytics requires train/val keys even for test-only sets
        yaml += ["train: images/test", "val: images/test"]
    yaml.append("names:")
    yaml += [f"  {i}: {c['name']}" for i, c in enumerate(cats)]
    (out / "dataset.yaml").write_text("\n".join(yaml) + "\n")
    summary = ", ".join(f"{len(v)} {k}" for k, v in splits.items())
    print(f"{root.name}: {summary} images, {n_boxes} boxes -> {out}")


if __name__ == "__main__":
    main()

"""Step 6 - Visual sanity check of the synthetic annotations.

Draws instance masks + bounding boxes (with class and deformation label) on a
sample of rendered images, and builds contact sheets for the presentation:
    outputs/figures/annotations_<dataset>/<image>.jpg   individual overlays
    outputs/figures/annotation_grid_<dataset>.jpg       overlays, 4 x 3 grid
    outputs/figures/render_grid_<dataset>.jpg           raw renders, 4 x 3 grid
It also prints simple dataset statistics.

Usage:
    python src/visualize_annotations.py --dataset data/synthetic/B_deformed --num 12
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

from config import FIGURES as FIG
from config import SYNTHETIC

# RGB, reference categorical palette order (blue, orange, aqua, yellow)
PALETTE = [(42, 120, 214), (235, 104, 52), (27, 175, 122), (237, 161, 0)]
CLASS_COLOURS = {}


def ann_mask(ann, h, w):
    seg = ann["segmentation"]
    if isinstance(seg, list):
        rles = mask_utils.frPyObjects(seg, h, w)
        rle = mask_utils.merge(rles)
    elif isinstance(seg.get("counts"), list):
        rle = mask_utils.frPyObjects(seg, h, w)
    else:
        rle = seg
    return mask_utils.decode(rle).astype(bool)


def label_for(ann, cat_names):
    d = ann.get("deformation") or {}
    state = f"{d['type']}-{d['level']}" if d.get("deformed") else "intact"
    return f"{cat_names[ann['category_id']]} | {state}"


def draw(img, anns, cat_names, rng):
    out = img.copy()
    h, w = img.shape[:2]
    for ann in anns:
        cls = cat_names[ann["category_id"]]
        base = np.array(CLASS_COLOURS[cls], float)
        colour = np.clip(base + rng.uniform(-40, 40, 3), 0, 255)
        m = ann_mask(ann, h, w)
        out[m] = (0.5 * out[m] + 0.5 * colour).astype(np.uint8)
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, colour.tolist(), 1)
    for ann in anns:  # boxes and labels on top of all masks
        cls = cat_names[ann["category_id"]]
        x, y, bw, bh = map(int, ann["bbox"])
        c = CLASS_COLOURS[cls]
        cv2.rectangle(out, (x, y), (x + bw, y + bh), c, 2)
        text = label_for(ann, cat_names)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = max(y, th + 4)
        cv2.rectangle(out, (x, ty - th - 4), (x + tw + 4, ty), c, -1)
        cv2.putText(out, text, (x + 2, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def grid(images, cols, cell_w=480):
    cell_h = int(images[0].shape[0] * cell_w / images[0].shape[1])
    rows = (len(images) + cols - 1) // cols
    sheet = np.full((rows * cell_h + (rows + 1) * 6, cols * cell_w + (cols + 1) * 6, 3), 255, np.uint8)
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        y, x = 6 + r * (cell_h + 6), 6 + c * (cell_w + 6)
        sheet[y:y + cell_h, x:x + cell_w] = cv2.resize(im, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
    return sheet


def save_rgb(path, rgb):
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(SYNTHETIC / "B_deformed"))
    ap.add_argument("--num", type=int, default=12)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ds = Path(args.dataset)
    coco_path = ds / "coco_data" / "coco_annotations.json"
    coco = json.loads(coco_path.read_text())
    cat_names = {c["id"]: c["name"] for c in coco["categories"]}
    for c in sorted(coco["categories"], key=lambda c: c["id"]):
        CLASS_COLOURS[c["name"]] = PALETTE[(c["id"] - 1) % len(PALETTE)]
    tag = ds.name
    by_image = {im["id"]: [] for im in coco["images"]}
    for a in coco["annotations"]:
        by_image[a["image_id"]].append(a)

    rng = np.random.default_rng(args.seed)
    candidates = [im for im in coco["images"] if by_image[im["id"]]]
    sample = sorted(random.Random(args.seed).sample(candidates, min(args.num, len(candidates))), key=lambda i: i["id"])

    (FIG / f"annotations_{tag}").mkdir(parents=True, exist_ok=True)
    raws, overlays = [], []
    for im in sample:
        rgb = cv2.cvtColor(cv2.imread(str(coco_path.parent / im["file_name"])), cv2.COLOR_BGR2RGB)
        ov = draw(rgb, by_image[im["id"]], cat_names, rng)
        save_rgb(FIG / f"annotations_{tag}" / Path(im["file_name"]).name, ov)
        raws.append(rgb)
        overlays.append(ov)
    save_rgb(FIG / f"annotation_grid_{tag}.jpg", grid(overlays, args.cols))
    save_rgb(FIG / f"render_grid_{tag}.jpg", grid(raws, args.cols))

    # Dataset statistics
    anns = coco["annotations"]
    per_class = Counter(cat_names[a["category_id"]] for a in anns)
    per_state = Counter(
        (cat_names[a["category_id"]],
         f"{a['deformation']['type']}-{a['deformation']['level']}" if a.get("deformation", {}).get("deformed") else "intact")
        for a in anns)
    print(f"images: {len(coco['images'])}, annotations: {len(anns)}, "
          f"mean objects/image: {len(anns) / max(1, len(coco['images'])):.2f}")
    print("per class:", dict(per_class))
    for (cls, state), n in sorted(per_state.items()):
        print(f"  {cls:6s} {state:14s} {n}")
    print(f"saved {len(overlays)} overlays, {FIG / f'annotation_grid_{tag}.jpg'}, {FIG / f'render_grid_{tag}.jpg'}")


if __name__ == "__main__":
    main()

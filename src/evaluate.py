"""Stage 8 - Evaluate the trained detectors (A = intact only, B = with deformations).

For every model x test set:
    * synth_B : held-out test scenes of the deformed dataset (contains deformed objects)
    * synth_A : the same test scenes with every object intact
    * taco    : real-world TACO photos mapped to our classes (sim-to-real)
it records precision, recall, mAP50 and mAP50-95 overall and per class
(Ultralytics val; plots and confusion matrices in outputs/results/val/).

On synth_B it also matches predictions to ground truth (same class, IoU >= 0.5,
conf >= 0.25) to measure recall per deformation type and level.

Writes outputs/results/{metrics_overall,metrics_per_class,recall_by_deformation}.csv
and prediction examples to outputs/figures/predictions_{taco,synthetic}.jpg.

Usage:
    python src/evaluate.py
"""
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from config import FIGURES, REAL, RESULTS, SYNTHETIC, TRAIN, class_names, load_config

MODELS = ["A_intact", "B_deformed"]
TEST_SETS = {
    "synth_B": SYNTHETIC / "B_deformed",
    "synth_A": SYNTHETIC / "A_intact",
    "taco": REAL / "taco",
}
CONF, IOU_MATCH = 0.25, 0.5
SHOW_CONF = 0.4  # only for the example figures, to keep them readable
BOX_COLOURS = [(42, 120, 214), (235, 104, 52), (27, 175, 122), (237, 161, 0)]  # RGB per class


def iou(a, b):
    """a, b: [x1, y1, x2, y2]."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def test_images(dataset_dir):
    return sorted((dataset_dir / "yolo" / "images" / "test").glob("*.jpg"))


def gt_by_image(dataset_dir):
    coco = json.loads((dataset_dir / "coco_data" / "coco_annotations.json").read_text())
    gt = defaultdict(list)
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        if w * h >= 16:
            gt[a["image_id"]].append({"cls": a["category_id"] - 1, "box": [x, y, x + w, y + h],
                                      "deformation": a.get("deformation") or {}})
    return gt


def predict(model, paths, device, imgsz):
    preds = {}
    for i in range(0, len(paths), 32):
        batch = paths[i:i + 32]
        for p, r in zip(batch, model.predict([str(p) for p in batch], conf=CONF, imgsz=imgsz,
                                             device=device, verbose=False)):
            b = r.boxes
            preds[int(p.stem)] = [{"cls": int(c), "conf": float(s), "box": xyxy.tolist()}
                                  for c, s, xyxy in zip(b.cls.cpu(), b.conf.cpu(), b.xyxy.cpu())]
    return preds


def match_recall(gt, preds):
    """Greedy same-class matching; returns list of (gt object, matched?)."""
    out = []
    for img_id, objs in gt.items():
        cand = sorted(preds.get(img_id, []), key=lambda p: -p["conf"])
        used = set()
        for o in objs:
            best, best_j = 0, None
            for j, p in enumerate(cand):
                if j in used or p["cls"] != o["cls"]:
                    continue
                v = iou(o["box"], p["box"])
                if v > best:
                    best, best_j = v, j
            hit = best >= IOU_MATCH
            if hit:
                used.add(best_j)
            out.append((o, hit))
    return out


def draw(img_path, gts, preds, names, title):
    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    for o in gts:  # ground truth: thin white box
        x1, y1, x2, y2 = map(int, o["box"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)
    for p in preds:
        x1, y1, x2, y2 = map(int, p["box"])
        c = BOX_COLOURS[p["cls"] % len(BOX_COLOURS)]
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
        text = f"{names[p['cls']]} {p['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = max(y1, th + 4)
        cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 4, ty), c, -1)
        cv2.putText(img, text, (x1 + 2, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(img, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def example_sheet(paths, gt, preds_by_model, names, out_path, cell_w=420):
    rows = []
    for p in paths:
        cells = [draw(p, gt.get(int(p.stem), []),
                      [q for q in preds_by_model[m].get(int(p.stem), []) if q["conf"] >= SHOW_CONF], names,
                      f"model {m.split('_')[0]} ({m})  |  white = ground truth") for m in MODELS]
        cells = [cv2.resize(c, (cell_w, int(c.shape[0] * cell_w / c.shape[1]))) for c in cells]
        h = max(c.shape[0] for c in cells)
        row = np.full((h, cell_w * len(cells) + 6 * (len(cells) - 1), 3), 255, np.uint8)
        for i, c in enumerate(cells):
            row[:c.shape[0], i * (cell_w + 6):i * (cell_w + 6) + cell_w] = c
        rows.append(row)
    width = max(r.shape[1] for r in rows)
    sheet = np.full((sum(r.shape[0] for r in rows) + 6 * (len(rows) - 1), width, 3), 255, np.uint8)
    y = 0
    for r in rows:
        sheet[y:y + r.shape[0], :r.shape[1]] = r
        y += r.shape[0] + 6
    cv2.imwrite(str(out_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=cfg["train"]["device"])
    ap.add_argument("--imgsz", type=int, default=cfg["train"]["imgsz"])
    ap.add_argument("--examples", type=int, default=6)
    args = ap.parse_args()
    names = class_names(cfg)
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    overall, per_class, deform_rows = [], [], []
    preds_cache = {}
    for m in MODELS:
        weights = TRAIN / "weights" / f"{m}.pt"
        model = YOLO(str(weights))
        for ts, ds_dir in TEST_SETS.items():
            r = model.val(data=str(ds_dir / "yolo" / "dataset.yaml"), split="test", imgsz=args.imgsz,
                          device=args.device, project=str(RESULTS / "val"), name=f"{m}_on_{ts}",
                          exist_ok=True, plots=True, verbose=False)
            b = r.box
            overall.append({"model": m, "test_set": ts, "precision": round(float(b.mp), 4),
                            "recall": round(float(b.mr), 4), "mAP50": round(float(b.map50), 4),
                            "mAP50_95": round(float(b.map), 4)})
            for k, ci in enumerate(b.ap_class_index):
                p, rc, ap50, ap = b.class_result(k)
                per_class.append({"model": m, "test_set": ts, "class": names[int(ci)],
                                  "precision": round(float(p), 4), "recall": round(float(rc), 4),
                                  "AP50": round(float(ap50), 4), "AP50_95": round(float(ap), 4)})
            print(f"{m} on {ts}: {overall[-1]}")

        # Recall per deformation type / level on the deformed synthetic test scenes.
        gt = gt_by_image(TEST_SETS["synth_B"])
        paths = test_images(TEST_SETS["synth_B"])
        gt = {int(p.stem): gt.get(int(p.stem), []) for p in paths}
        preds = predict(model, paths, args.device, args.imgsz)
        preds_cache[(m, "synth_B")] = preds
        groups = defaultdict(lambda: [0, 0])
        for o, hit in match_recall(gt, preds):
            d = o["deformation"]
            key = (names[o["cls"]], d.get("type") or "intact", d.get("level") or "intact") \
                if d.get("deformed") else (names[o["cls"]], "intact", "intact")
            for k in (key, ("all", key[1], key[2]), ("all", key[1], "all") if key[1] != "intact" else None):
                if k:
                    groups[k][0] += hit
                    groups[k][1] += 1
        for (cls, dtype, level), (hits, n) in sorted(groups.items()):
            deform_rows.append({"model": m, "class": cls, "deform_type": dtype, "deform_level": level,
                                "objects": n, "detected": hits, "recall": round(hits / n, 4) if n else ""})

        taco_paths = test_images(TEST_SETS["taco"])
        preds_cache[(m, "taco")] = predict(model, taco_paths, args.device, args.imgsz)

    def write(name, rows):
        with open(RESULTS / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    write("metrics_overall.csv", overall)
    write("metrics_per_class.csv", per_class)
    write("recall_by_deformation.csv", deform_rows)

    # Prediction examples (same images for both models; white = ground truth).
    rng = random.Random(cfg["seed"])
    for ts, fig in (("taco", "predictions_taco.jpg"), ("synth_B", "predictions_synthetic.jpg")):
        gt = gt_by_image(TEST_SETS[ts])
        paths = [p for p in test_images(TEST_SETS[ts]) if gt.get(int(p.stem))]
        sample = sorted(rng.sample(paths, min(args.examples, len(paths))))
        example_sheet(sample, gt, {m: preds_cache[(m, ts)] for m in MODELS}, names, FIGURES / fig)
    print(f"results written to {RESULTS}")


if __name__ == "__main__":
    main()

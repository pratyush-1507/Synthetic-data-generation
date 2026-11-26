"""Stage 6 - Prepare the real-world test set from TACO (http://tacodataset.org).

Downloads the official TACO annotations and images (640-px renditions, or the
original downscaled to 640 px with EXIF orientation removed), maps
TACO categories onto our classes (the `taco` lists in configs/pipeline.yaml),
drops every other category, rescales boxes to the downloaded resolution and
writes a COCO file. All 1500 images are kept: images without any of our
classes still count for false positives.

Output:
    data/real/taco/coco_data/images/<id>.jpg
    data/real/taco/coco_data/coco_annotations.json
then `coco_to_yolo.py --dataset data/real/taco --all-test` builds the YOLO test set.

Usage:
    python src/prepare_taco.py
"""
import argparse
import json
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageOps

from config import REAL, category_ids, load_config

ANNOTATIONS_URL = "https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations.json"
USER_AGENT = "AI512-course-project/1.0"


def fetch(url, dst, retries=3):
    if dst.exists() and dst.stat().st_size > 0:
        return True
    for _ in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as r:
                dst.write_bytes(r.read())
            return True
        except Exception:
            continue
    return False


def normalise_image(path, max_side):
    """Apply the EXIF orientation (TACO annotates the displayed image), downscale
    to max_side and drop EXIF so every loader sees the same pixels."""
    with Image.open(path) as im:
        needs = max(im.size) > max_side or "exif" in im.info or im.mode != "RGB"
        if not needs:
            return
        im = ImageOps.exif_transpose(im).convert("RGB")
        if max(im.size) > max_side:
            im.thumbnail((max_side, max_side), Image.LANCZOS)
        im.save(path, quality=92)


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REAL / "taco"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-side", type=int, default=640, help="downscale larger images to this size")
    args = ap.parse_args()

    out = Path(args.out)
    img_dir = out / "coco_data" / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    raw_ann = out / "annotations_taco.json"
    if not raw_ann.exists():
        fetch(ANNOTATIONS_URL, raw_ann)
    taco = json.loads(raw_ann.read_text())

    ids = category_ids(cfg)
    taco_names = {c["id"]: c["name"] for c in taco["categories"]}
    name_to_class = {t: cls for cls, c in cfg["classes"].items() for t in c["taco"]}
    unknown = set(name_to_class) - set(taco_names.values())
    if unknown:
        raise SystemExit(f"TACO categories not found: {unknown}")

    def download(im):
        # Prefer the 640 px rendition; some images only have the full-size original.
        dst = img_dir / f"{im['id']:06d}.jpg"
        url = im.get("flickr_640_url") or im.get("flickr_url")
        if not fetch(url, dst):
            return im["id"], False
        normalise_image(dst, args.max_side)
        return im["id"], True

    with ThreadPoolExecutor(args.workers) as pool:
        ok = dict(pool.map(download, taco["images"]))
    print(f"downloaded {sum(ok.values())}/{len(taco['images'])} images")

    images, annotations, skipped = [], [], Counter()
    by_image = {}
    for a in taco["annotations"]:
        by_image.setdefault(a["image_id"], []).append(a)
    for im in taco["images"]:
        path = img_dir / f"{im['id']:06d}.jpg"
        if not ok[im["id"]]:
            skipped["download failed"] += 1
            continue
        with Image.open(path) as pil:
            w, h = pil.size
        # The 640 rendition must have the annotation's orientation; rotated EXIF copies are skipped.
        if (w >= h) != (im["width"] >= im["height"]) and im["width"] != im["height"]:
            skipped["orientation mismatch"] += 1
            continue
        sx, sy = w / im["width"], h / im["height"]
        images.append({"id": im["id"], "file_name": f"images/{im['id']:06d}.jpg", "width": w, "height": h,
                       "taco_file": im["file_name"]})
        for a in by_image.get(im["id"], []):
            cls = name_to_class.get(taco_names[a["category_id"]])
            if cls is None:
                continue
            x, y, bw, bh = a["bbox"]
            annotations.append({"id": len(annotations) + 1, "image_id": im["id"], "category_id": ids[cls],
                                "bbox": [x * sx, y * sy, bw * sx, bh * sy], "area": bw * bh * sx * sy,
                                "iscrowd": 0, "taco_category": taco_names[a["category_id"]]})

    coco = {"info": {"description": "TACO mapped to project classes", "url": "http://tacodataset.org"},
            "images": images, "annotations": annotations,
            "categories": [{"id": i, "name": n, "supercategory": "waste"}
                           for n, i in sorted(ids.items(), key=lambda kv: kv[1])]}
    (out / "coco_data" / "coco_annotations.json").write_text(json.dumps(coco))
    per_class = Counter(next(n for n, i in ids.items() if i == a["category_id"]) for a in annotations)
    with_targets = len({a["image_id"] for a in annotations})
    print(f"{len(images)} images ({with_targets} contain our classes), {len(annotations)} objects: {dict(per_class)}")
    if skipped:
        print(f"skipped: {dict(skipped)}")


if __name__ == "__main__":
    main()

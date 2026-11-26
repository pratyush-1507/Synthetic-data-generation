"""Stage 1 - Download meshes for every class from Objaverse (LVIS subset).

For each class in configs/pipeline.yaml: collect candidates from its LVIS
categories, keep permissive licences (CC-BY / CC0), filter by name, file size
and polygon count, download, and keep only models that pass the class shape
check (mesh_utils.shape_check). GLBs go to assets/models/raw/<class>/ and
attribution to assets/models/LICENSES.csv.

Incremental: models already in raw/<class>/ are kept, only missing ones are fetched.

Usage:
    python src/download_objaverse.py                 # all classes, per_class from config
    python src/download_objaverse.py --classes cup carton --per-class 8
"""
import argparse
import csv
import shutil
from pathlib import Path

import objaverse

from config import LICENSE_CSV, RAW, ROOT, load_config
from mesh_utils import load_single_mesh, shape_check

LICENSE_NAMES = {"by": "CC-BY 4.0", "cc0": "CC0 1.0", "by-sa": "CC-BY-SA 4.0"}
FIELDS = ["uid", "class", "lvis_category", "name", "author", "source_url", "license"]


def candidate_uids(lvis, categories):
    uids, seen = [], set()
    for cat in categories:
        for uid in lvis.get(cat, []):
            if uid not in seen:
                seen.add(uid)
                uids.append((uid, cat))
    return uids


_ALL_ANNOTATIONS = None


def name_search(keywords):
    """Candidates found by name anywhere in Objaverse (not only in the LVIS subset)."""
    global _ALL_ANNOTATIONS
    if _ALL_ANNOTATIONS is None:
        print("  loading all Objaverse annotations for name search (cached after first run)...")
        _ALL_ANNOTATIONS = objaverse.load_annotations()
    return [(u, "name-search") for u, a in _ALL_ANNOTATIONS.items()
            if any(k in (a.get("name") or "").lower() for k in keywords)]


def name_ok(name, cls_cfg, cfg):
    name = (name or "").lower()
    if any(w in name for w in cfg["exclude_words_all"] + cls_cfg.get("exclude_words", [])):
        return False
    include = cls_cfg.get("include_words")
    return not include or any(w in name for w in include)


def acceptable(ann, cls_cfg, cfg):
    dl = cfg["download"]
    if ann.get("license") not in dl["licenses"] or ann.get("uid") in cfg["exclude_uids"]:
        return False
    if not name_ok(ann.get("name"), cls_cfg, cfg):
        return False
    glb = (ann.get("archives") or {}).get("glb") or {}
    size = glb.get("size") or 0
    faces = glb.get("faceCount") or ann.get("faceCount") or 0
    return 0 < size <= dl["max_mb"] * 1e6 and 100 <= faces <= dl["max_faces"]


def loads_ok(path, cls_cfg):
    """Reject unparsable files, multi-object scenes and implausible shapes."""
    try:
        ok, reason, _ = shape_check(load_single_mesh(path), cls_cfg)
    except Exception as e:
        ok, reason = False, f"load failed: {e}"
    if not ok:
        print(f"  - skip {Path(path).stem}: {reason}")
    return ok


def read_license_rows():
    if not LICENSE_CSV.exists():
        return {}
    with open(LICENSE_CSV) as f:
        return {r["uid"]: r for r in csv.DictReader(f)}


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", nargs="+", default=list(cfg["classes"]))
    ap.add_argument("--per-class", type=int, default=cfg["download"]["per_class"])
    ap.add_argument("--processes", type=int, default=4)
    args = ap.parse_args()

    lvis = objaverse.load_lvis_annotations()
    rows = read_license_rows()
    for cls in args.classes:
        cls_cfg = cfg["classes"][cls]
        out_dir = RAW / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        have = sorted(p.stem for p in out_dir.glob("*.glb") if p.stem not in cfg["exclude_uids"])
        for p in out_dir.glob("*.glb"):  # drop models excluded after download
            if p.stem in cfg["exclude_uids"]:
                p.unlink()
                rows.pop(p.stem, None)
        if len(have) >= args.per_class:
            print(f"[{cls}] already has {len(have)} models, skipping")
            continue

        cands = candidate_uids(lvis, cls_cfg["lvis"])
        if cls_cfg.get("search_names"):
            seen = {u for u, _ in cands}
            cands += [c for c in name_search(cls_cfg["search_names"]) if c[0] not in seen]
        anns = objaverse.load_annotations([u for u, _ in cands])
        cat_of = dict(cands)
        ranked = sorted(
            (u for u, _ in cands if u in anns and u not in have and acceptable(anns[u], cls_cfg, cfg)),
            key=lambda u: (-(anns[u].get("likeCount") or 0), u),  # popular first, deterministic
        )
        print(f"[{cls}] {len(cands)} candidates, {len(ranked)} pass licence/name/size filters, "
              f"{len(have)} already downloaded")

        need = args.per_class - len(have)
        for start in range(0, len(ranked), need):
            batch = ranked[start:start + need]
            paths = objaverse.load_objects(uids=batch, download_processes=args.processes)
            for uid in batch:
                p = paths.get(uid)
                if p and len(have) < args.per_class and loads_ok(p, cls_cfg):
                    shutil.copy(p, out_dir / f"{uid}.glb")
                    have.append(uid)
                    a = anns[uid]
                    rows[uid] = {"uid": uid, "class": cls, "lvis_category": cat_of[uid],
                                 "name": a.get("name", ""),
                                 "author": (a.get("user") or {}).get("displayName", ""),
                                 "source_url": a.get("viewerUrl", ""),
                                 "license": LICENSE_NAMES.get(a.get("license"), a.get("license"))}
                    print(f"  + {uid}  {a.get('name')!r}  ({rows[uid]['license']})")
            if len(have) >= args.per_class:
                break
        status = "" if len(have) >= args.per_class else "  (WARNING: fewer than requested)"
        print(f"[{cls}] {len(have)} models in {out_dir.relative_to(ROOT)}{status}")

    present = {p.stem for p in RAW.glob("*/*.glb")}
    with open(LICENSE_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in sorted(rows.values(), key=lambda r: (r["class"], r["uid"])):
            if r["uid"] in present:
                w.writerow(r)
    print(f"wrote {LICENSE_CSV.relative_to(ROOT)} ({len(present)} models)")


if __name__ == "__main__":
    main()

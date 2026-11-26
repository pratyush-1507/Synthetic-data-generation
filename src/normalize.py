"""Step 2 - Normalise raw Objaverse GLBs to real-world scale.

For every assets/models/raw/<class>/<uid>.glb:
  * flatten the glTF scene graph into a single mesh (textures are packed
    into one atlas when the parts are textured),
  * rotate so the object's symmetry axis (round cross-section) is +Y
    (OBJ / glTF convention; Blender converts Y-up to Z-up on import),
  * scale uniformly so the height matches a real object
    (height_m per class in configs/pipeline.yaml, metres),
  * move the origin to the centre of the base (x, z centred, min y = 0),
  * export assets/models/normalized/<class>/<uid>/model.obj (+ .mtl, texture).

Usage:
    python src/normalize.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

from config import NORM, RAW, ROOT, load_config
from mesh_utils import load_single_mesh, shape_check


def upright(mesh, cls_cfg):
    """Rotate so the height axis (from the class shape rule) becomes +Y."""
    ok, reason, axis = shape_check(mesh, cls_cfg)
    if axis == 0:
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1]))
    elif axis == 2:
        mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    return axis, ok, reason


def normalize(mesh, height):
    mesh.apply_scale(height / mesh.extents[1])
    lo, hi = mesh.bounds
    centre = (lo + hi) / 2
    mesh.apply_translation([-centre[0], -lo[1], -centre[2]])
    return mesh


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", nargs="+", default=list(cfg["classes"]))
    ap.add_argument("--force", action="store_true", help="re-normalise models that already exist")
    args = ap.parse_args()

    summary = []
    for cls in args.classes:
        for glb in sorted((RAW / cls).glob("*.glb")):
            uid = glb.stem
            if (NORM / cls / uid / "model.obj").exists() and not args.force:
                continue
            mesh = load_single_mesh(glb)
            raw_ext = mesh.extents.copy()
            axis, ok, reason = upright(mesh, cfg["classes"][cls])
            if not ok:
                print(f"[{cls}] {uid}: WARNING {reason}")
            normalize(mesh, cfg["classes"][cls]["height_m"])
            out = NORM / cls / uid
            out.mkdir(parents=True, exist_ok=True)
            mesh.export(out / "model.obj", include_texture=True, mtl_name="model.mtl")
            textured = isinstance(mesh.visual, trimesh.visual.TextureVisuals)
            info = {
                "uid": uid, "class": cls,
                "raw_extents": raw_ext.round(4).tolist(),
                "height_axis_in_raw": "xyz"[axis],
                "extents_m": mesh.extents.round(4).tolist(),
                "faces": int(len(mesh.faces)),
                "textured": textured,
            }
            (out / "normalize.json").write_text(json.dumps(info, indent=2))
            summary.append(info)
            print(f"[{cls}] {uid}: {info['extents_m']} m, {info['faces']} faces, "
                  f"textured={textured}, height axis={info['height_axis_in_raw']}")
    print(f"normalised {len(summary)} models into {NORM.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

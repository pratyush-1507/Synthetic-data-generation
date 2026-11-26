import blenderproc as bproc  # must be the first line for `blenderproc run`
# Step 3 - Procedural deformation of the normalised meshes (runs inside Blender).
#
# Two deformation types, each with three severity levels sampled from fixed ranges:
#
#   crush : Lattice squash along the height (default height loss 10-25 / 25-50 / 50-75 %),
#           barrel bulge + random per-ring offsets for asymmetric folding, a slight
#           tilt of the top, diagonal accordion folds, and Displace-modifier
#           noise for crumples.
#   bend  : SimpleDeform BEND about a pivot at a random height and azimuth
#           (default 15-35 / 35-60 / 60-90 degrees). Ranges live in configs/pipeline.yaml.
#
# Every variant is deterministic given (--seed, uid, type, level, variant). Output:
#     assets/models/deformed/<class>/<uid>/<type>_<level>_<k>.obj  (+ .mtl, textures
#         are referenced from normalized/ by relative path)
#     assets/models/deformed/<class>/<uid>/<type>_<level>_<k>.json (parameters)
#
# Usage:
#     ./run_blenderproc.sh src/deform.py            (settings from configs/pipeline.yaml)
import argparse
import json
import math
import sys
import zlib
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DEFORMED, NORM, load_config  # noqa: E402

CFG = load_config()
LEVELS = {t: {lvl: tuple(r) for lvl, r in levels.items()} for t, levels in CFG["deform"]["levels"].items()}
MIN_FACES = 8000  # subdivide sparse meshes so deformations stay smooth


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=CFG["seed"])
    ap.add_argument("--variants", type=int, default=CFG["deform"]["variants"], help="variants per type and level")
    ap.add_argument("--types", nargs="+", default=list(LEVELS))
    ap.add_argument("--classes", nargs="+", default=list(CFG["classes"]))
    ap.add_argument("--force", action="store_true", help="regenerate variants that already exist")
    ap.add_argument("--uids", nargs="*", help="restrict to these model uids")
    return ap.parse_args(argv)


def rng_for(seed, *keys):
    return np.random.default_rng(zlib.crc32(("|".join(map(str, (seed,) + keys))).encode()))


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for coll in (bpy.data.meshes, bpy.data.lattices, bpy.data.textures, bpy.data.objects):
        for block in list(coll):
            if block.users == 0:
                coll.remove(block)


def import_model(obj_path):
    bpy.ops.wm.obj_import(filepath=str(obj_path))  # Y-up OBJ -> Z-up Blender
    objs = [o for o in bpy.context.selected_objects if o.type == "MESH"]
    bpy.context.view_layer.objects.active = objs[0]
    if len(objs) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    # Weld UV-seam duplicates (prevents cracks) and add resolution for smooth bending.
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=1e-5)
    faces = len(obj.data.polygons)
    cuts = min(8, max(0, math.ceil(math.sqrt(MIN_FACES / max(faces, 1))) - 1))
    if cuts:
        bpy.ops.mesh.subdivide(number_cuts=cuts)
    bpy.ops.object.mode_set(mode="OBJECT")
    return obj


def bbox(obj):
    co = np.array([v.co[:] for v in obj.data.vertices])
    return co.min(0), co.max(0)


def add_crush(obj, rng, lo, hi):
    loss = float(rng.uniform(lo, hi))
    mn, mx = bbox(obj)
    size = mx - mn
    radius = float(max(size[0], size[1]) / 2)
    bulge = float(loss * rng.uniform(0.35, 0.6))
    tilt = float(rng.uniform(-1, 1) * 15 * loss)  # degrees, grows with severity
    tilt_dir = float(rng.uniform(0, 2 * math.pi))
    ring_jitter = float(0.12 * loss)
    noise_strength = float(radius * rng.uniform(0.12, 0.2) * (0.4 + loss))
    noise_scale = float(radius * rng.uniform(0.25, 0.45))
    n_folds = int(rng.integers(2, 5)) + int(round(4 * loss))
    fold_depth = float(radius * rng.uniform(0.15, 0.25) * loss)
    fold_skew = float(rng.uniform(0.5, 2.0))
    fold_phase = float(rng.uniform(0, 2 * math.pi))

    # Accordion folds: radial dents in rings along the height whose phase varies
    # with the azimuth, so the pleats run diagonally like a real crushed can.
    for v in obj.data.vertices:
        x, y, z = v.co
        t = (z - mn[2]) / max(size[2], 1e-9)
        theta = math.atan2(y, x)
        wave = math.sin(n_folds * math.pi * t + fold_skew * math.sin(theta + fold_phase))
        dent = fold_depth * (wave ** 2) * math.sin(math.pi * t) ** 0.5
        r = math.hypot(x, y)
        if r > 1e-6:
            k = max(0.0, (r - dent) / r)
            v.co.x, v.co.y = x * k, y * k
    obj.data.update()

    # Lattice: 3 x 3 x 6 control points spanning the bounding box.
    lat_data = bpy.data.lattices.new("crush_lattice")
    lat_data.points_u, lat_data.points_v, lat_data.points_w = 3, 3, 6
    lat = bpy.data.objects.new("crush_lattice", lat_data)
    bpy.context.collection.objects.link(lat)
    lat.location = Vector(((mn + mx) / 2).tolist())
    lat.scale = Vector((size * 1.02).tolist())
    nw = lat_data.points_w
    ring_offsets = rng.uniform(-ring_jitter, ring_jitter, size=(nw, 2))
    ring_offsets[0] = 0  # keep the base footprint
    for i, p in enumerate(lat_data.points):
        u, v, w = p.co  # local coords in [-0.5, 0.5]
        t = w + 0.5     # normalised height
        k = i // 9
        radial = 1 + bulge * math.sin(math.pi * t) + rng.uniform(-ring_jitter, ring_jitter)
        x = u * radial + ring_offsets[k, 0]
        y = v * radial + ring_offsets[k, 1]
        z = -0.5 + t * (1 - loss)
        z += math.tan(math.radians(tilt)) * t * (u * math.cos(tilt_dir) + v * math.sin(tilt_dir)) * (size[0] / size[2])
        p.co_deform = Vector((x, y, z))
    lat.hide_render = True

    # Crumples: displacement along normals driven by a cloud texture whose
    # coordinates come from a randomly placed empty (deterministic per seed).
    tex = bpy.data.textures.new("crumple", type="CLOUDS")
    tex.noise_scale = noise_scale
    tex.noise_basis = "VORONOI_F2_F1"
    tex.noise_depth = 2
    empty = bpy.data.objects.new("crumple_coords", None)
    bpy.context.collection.objects.link(empty)
    empty.location = Vector(rng.uniform(-10, 10, 3).tolist())
    disp = obj.modifiers.new("crumple", "DISPLACE")
    disp.texture, disp.texture_coords, disp.texture_coords_object = tex, "OBJECT", empty
    disp.strength, disp.mid_level = noise_strength, 0.5
    lm = obj.modifiers.new("crush", "LATTICE")
    lm.object = lat
    return {"height_loss_target": round(loss, 4), "bulge": round(bulge, 4), "tilt_deg": round(tilt, 3),
            "tilt_dir_deg": round(math.degrees(tilt_dir), 2), "ring_jitter": round(ring_jitter, 4),
            "noise_strength_m": round(noise_strength, 5), "noise_scale_m": round(noise_scale, 5),
            "n_folds": n_folds, "fold_depth_m": round(fold_depth, 5), "fold_skew": round(fold_skew, 3)}


def add_bend(obj, rng, lo, hi):
    angle = float(rng.uniform(lo, hi))
    mn, mx = bbox(obj)
    pivot = float(rng.uniform(0.35, 0.65))
    azimuth = float(rng.uniform(0, 360))
    empty = bpy.data.objects.new("bend_origin", None)
    bpy.context.collection.objects.link(empty)
    empty.location = Vector((0.0, 0.0, mn[2] + pivot * (mx[2] - mn[2])))
    # SimpleDeform bends the X axis around Z; rotate the frame so the object's
    # height (Z) plays the role of X and the bend plane has a random azimuth.
    empty.rotation_euler = (0.0, math.radians(-90), math.radians(azimuth))
    m = obj.modifiers.new("bend", "SIMPLE_DEFORM")
    m.deform_method, m.deform_axis, m.origin = "BEND", "Z", empty
    m.angle = math.radians(angle)
    return {"angle_deg": round(angle, 3), "pivot_height_frac": round(pivot, 4), "azimuth_deg": round(azimuth, 2)}


def finalize(obj):
    """Apply modifiers and put the origin back at the centre of the base."""
    bpy.context.view_layer.objects.active = obj
    for m in list(obj.modifiers):
        bpy.ops.object.modifier_apply(modifier=m.name)
    mn, mx = bbox(obj)
    shift = np.array([-(mn[0] + mx[0]) / 2, -(mn[1] + mx[1]) / 2, -mn[2]])
    for v in obj.data.vertices:
        v.co += Vector(shift.tolist())
    obj.data.update()
    return bbox(obj)


def export(obj, path):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.ops.wm.obj_export(filepath=str(path), export_selected_objects=True, export_materials=True,
                          path_mode="RELATIVE", export_uv=True, export_normals=True)


def main():
    args = parse_args()
    bproc.init()
    total = 0
    for cls in args.classes:
        for model_dir in sorted((NORM / cls).iterdir()):
            src = model_dir / "model.obj"
            if not src.exists():
                continue
            uid = model_dir.name
            if args.uids and uid not in args.uids:
                continue
            out_dir = DEFORMED / cls / uid
            out_dir.mkdir(parents=True, exist_ok=True)
            for dtype in args.types:
                for level, (lo, hi) in LEVELS[dtype].items():
                    for k in range(args.variants):
                        if (out_dir / f"{dtype}_{level}_{k}.json").exists() and not args.force:
                            continue
                        clear_scene()
                        obj = import_model(src)
                        mn0, mx0 = bbox(obj)
                        rng = rng_for(args.seed, uid, dtype, level, k)
                        params = (add_crush if dtype == "crush" else add_bend)(obj, rng, lo, hi)
                        mn1, mx1 = finalize(obj)
                        name = f"{dtype}_{level}_{k}"
                        export(obj, out_dir / f"{name}.obj")
                        h0, h1 = mx0[2] - mn0[2], mx1[2] - mn1[2]
                        meta = {
                            "uid": uid, "class": cls, "type": dtype, "level": level, "variant": k,
                            "seed": args.seed, "level_range": [lo, hi], "params": params,
                            "height_before_m": round(float(h0), 4), "height_after_m": round(float(h1), 4),
                            "measured_height_loss": round(float(1 - h1 / h0), 4),
                            "extents_after_m": [round(float(e), 4) for e in (mx1 - mn1)],
                            "source": f"normalized/{cls}/{uid}/model.obj",
                        }
                        (out_dir / f"{name}.json").write_text(json.dumps(meta, indent=2))
                        total += 1
            print(f"[{cls}] {uid}: done")
    print(f"wrote {total} deformed meshes to {DEFORMED}")


if __name__ == "__main__":
    main()

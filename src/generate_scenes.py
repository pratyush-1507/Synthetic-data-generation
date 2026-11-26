import blenderproc as bproc  # must be the first line for `blenderproc run`
# Stage 4 - Render randomised synthetic scenes with COCO annotations (BlenderProc).
#
# Each scene contains
#   * a ground plane with a random procedural texture and colours,
#   * N target objects (classes from configs/pipeline.yaml), each intact or a
#     deformed variant, with randomised roughness / metallic and sometimes a
#     random colour,
#   * distractor primitives (background, category 0),
#   * 1-3 random point lights,
# objects are dropped with rigid-body physics, then rendered from several camera
# poses sampled from top-down or oblique viewpoints.
#
# Every scene is seeded by (seed, scene index), so
#   * dataset A (all intact) and dataset B (deformed) contain the same scenes,
#     except that deformed targets in B are intact in A (controlled experiment);
#   * an interrupted run can be resumed with --resume.
#
# Outputs (data/synthetic/<dataset>/):
#   coco_data/images/*.jpg, coco_data/coco_annotations.json
#       categories from the config; each annotation carries a "deformation" dict
#   instance_metadata.csv   one row per annotation (source model, deformed,
#                           type, level, parameters)
#   scene_metadata.jsonl    one line per image (camera, lights, scene content)
#
# Usage:
#     ./run_blenderproc.sh src/generate_scenes.py --dataset B_deformed --overwrite
#     ./run_blenderproc.sh src/generate_scenes.py --dataset A_intact --resume
import argparse
import csv
import json
import random
import shutil
import sys
import zlib
from pathlib import Path

import bpy
import numpy as np
from blenderproc.python.utility.BlenderUtility import get_all_blender_mesh_objects
from blenderproc.python.utility.LabelIdMapping import LabelIdMapping
from blenderproc.python.writer.CocoWriterUtility import _CocoWriterUtility

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DEFORMED, NORM, ROOT, SYNTHETIC, category_ids, load_config  # noqa: E402

CFG = load_config()
RCFG = CFG["render"]
CATEGORIES = category_ids(CFG)
DISTRACTOR_SHAPES = ["CUBE", "SPHERE", "CONE", "MONKEY"]
META_FIELDS = ["image_id", "file_name", "annotation_id", "category", "bbox", "area", "source_uid",
               "deformed", "deform_type", "deform_level", "deform_variant", "deform_params", "mesh"]
METALLIC = {"can": (0.3, 1.0)}  # other classes: (0.0, 0.2)


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="B_deformed", help="dataset name from configs/pipeline.yaml render.datasets")
    ap.add_argument("--num-images", type=int, default=RCFG["num_images"])
    ap.add_argument("--seed", type=int, default=CFG["seed"])
    ap.add_argument("--p-deformed", type=float, default=None, help="override the dataset's deformed probability")
    ap.add_argument("--samples", type=int, default=RCFG["samples"], help="Cycles samples (low = fast)")
    ap.add_argument("--out", default=None, help="output directory (default data/synthetic/<dataset>)")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--overwrite", action="store_true", help="delete previous output first")
    group.add_argument("--resume", action="store_true", help="continue an interrupted run")
    args = ap.parse_args(argv)
    if args.p_deformed is None:
        args.p_deformed = RCFG["datasets"][args.dataset]
    args.out = Path(args.out) if args.out else SYNTHETIC / args.dataset
    return args


# --------------------------------------------------------------------------- assets
def build_catalog():
    """Every usable mesh per (class, uid): the intact model and its deformed variants."""
    catalog = {}
    for cls in CATEGORIES:
        catalog[cls] = {}
        for model_dir in sorted((NORM / cls).iterdir()):
            if not (model_dir / "model.obj").exists() or model_dir.name in CFG["exclude_uids"]:
                continue
            uid = model_dir.name
            intact = {"class": cls, "uid": uid, "deformed": False, "mesh": model_dir / "model.obj"}
            variants = []
            for meta_path in sorted((DEFORMED / cls / uid).glob("*.json")):
                meta = json.loads(meta_path.read_text())
                variants.append({"class": cls, "uid": uid, "deformed": True, "mesh": meta_path.with_suffix(".obj"),
                                 "type": meta["type"], "level": meta["level"], "variant": meta["variant"],
                                 "params": meta["params"]})
            catalog[cls][uid] = (intact, variants)
    return catalog


def pick_target(catalog, rng, p_deformed):
    # The same random numbers are drawn whatever p_deformed is, so datasets with
    # different deformation ratios share identical scenes.
    cls = rng.choice(sorted(catalog))
    uid = rng.choice(sorted(catalog[cls]))
    intact, variants = catalog[cls][uid]
    u = rng.random()
    variant = rng.choice(variants) if variants else intact
    return variant if u < p_deformed else intact


def load_mesh(path):
    parts = bproc.loader.load_obj(str(path))
    if len(parts) > 1:
        parts[0].join_with_other_objects(parts[1:])
    obj = parts[0]
    # Give the object its own material copies so randomisation does not leak.
    for slot in obj.blender_obj.material_slots:
        if slot.material is not None:
            slot.material = slot.material.copy()
    return obj


def randomize_target_material(obj, cls, rng):
    info = {"recoloured": False}
    lo, hi = METALLIC.get(cls, (0.0, 0.2))
    roughness, metallic = float(rng.uniform(0.1, 0.8)), float(rng.uniform(lo, hi))
    recolour = rng.random() < 0.3  # sometimes replace the texture by a flat random colour
    colour = [rng.uniform(0.05, 1.0) for _ in range(3)] + [1.0]
    for slot in obj.blender_obj.material_slots:
        if slot.material is None:
            continue
        mat = bproc.types.Material(slot.material)
        mat.set_principled_shader_value("Roughness", roughness)
        mat.set_principled_shader_value("Metallic", metallic)
        if recolour:
            mat.set_principled_shader_value("Base Color", colour)
    info.update(roughness=round(roughness, 3), metallic=round(metallic, 3), recoloured=recolour)
    return info


def procedural_material(name, rng):
    """Random two-colour procedural texture (noise / voronoi / checker / wave)."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes["Principled BSDF"]
    kind = rng.choice(["ShaderNodeTexNoise", "ShaderNodeTexVoronoi", "ShaderNodeTexChecker", "ShaderNodeTexWave"])
    tex = nodes.new(kind)
    tex.inputs["Scale"].default_value = float(rng.uniform(2, 40))
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = [rng.random(), rng.random(), rng.random(), 1]
    ramp.color_ramp.elements[1].color = [rng.random(), rng.random(), rng.random(), 1]
    out = tex.outputs["Fac"] if "Fac" in tex.outputs else tex.outputs["Distance"]
    links.new(out, ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = float(rng.uniform(0.3, 1.0))
    return mat, kind.replace("ShaderNodeTex", "").lower()


# --------------------------------------------------------------------------- scene
def build_scene(catalog, rng, p_deformed):
    plane = bproc.object.create_primitive("PLANE", scale=[1.5, 1.5, 1])
    plane_mat, plane_tex = procedural_material("ground", rng)
    plane.replace_materials(bproc.types.Material(plane_mat))
    plane.set_cp("category_id", 0)
    plane.enable_rigidbody(active=False, collision_shape="BOX")

    targets = []
    for i in range(rng.randint(*RCFG["targets_per_scene"])):
        entry = pick_target(catalog, rng, p_deformed)
        obj = load_mesh(entry["mesh"])
        obj.set_name(f"target_{i}")
        obj.set_cp("category_id", CATEGORIES[entry["class"]])
        mat_info = randomize_target_material(obj, entry["class"], rng)
        obj.enable_rigidbody(active=True, collision_shape="CONVEX_HULL", mass=0.1)
        targets.append((obj, entry, mat_info))

    distractors = []
    for i in range(rng.randint(*RCFG["distractors_per_scene"])):
        d = bproc.object.create_primitive(rng.choice(DISTRACTOR_SHAPES))
        d.set_scale([float(rng.uniform(0.015, 0.06))] * 3)
        d.set_name(f"distractor_{i}")
        d.set_cp("category_id", 0)
        m, _ = procedural_material(f"distractor_{i}", rng)
        d.replace_materials(bproc.types.Material(m))
        d.enable_rigidbody(active=True, collision_shape="CONVEX_HULL", mass=0.05)
        distractors.append(d)

    dynamic = [t[0] for t in targets] + distractors

    def sample_pose(obj):
        obj.set_location(np.random.uniform([-0.22, -0.22, 0.05], [0.22, 0.22, 0.35]))
        obj.set_rotation_euler(np.random.uniform([0, 0, 0], [2 * np.pi] * 3))

    bproc.object.sample_poses(dynamic, sample_pose_func=sample_pose, objects_to_check_collisions=dynamic + [plane])
    bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=1, max_simulation_time=4,
                                                      check_object_interval=1, substeps_per_frame=20,
                                                      solver_iters=25)

    lights = []
    for _ in range(rng.randint(1, 3)):
        light = bproc.types.Light()
        light.set_type("POINT")
        loc = [rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8), rng.uniform(0.5, 1.5)]
        energy = float(rng.uniform(40, 300))
        light.set_location(loc)
        light.set_energy(energy)
        light.set_color([rng.uniform(0.8, 1.0) for _ in range(3)])
        lights.append({"location": [round(v, 3) for v in loc], "energy_W": round(energy, 1)})
    bproc.renderer.set_world_background([1, 1, 1], strength=float(rng.uniform(0.1, 0.5)))

    scene = {"ground_texture": plane_tex, "num_targets": len(targets), "num_distractors": len(distractors),
             "lights": lights}
    return plane, targets, distractors, scene


def sample_cameras(targets, n_views, rng):
    target_objs = [t[0] for t in targets]
    poi = bproc.object.compute_poi(target_objs)
    views = []
    for _ in range(n_views):
        for _attempt in range(30):
            view = "top_down" if rng.random() < 0.5 else "oblique"
            elev = np.radians(rng.uniform(70, 88) if view == "top_down" else rng.uniform(25, 60))
            azim = rng.uniform(0, 2 * np.pi)
            dist = rng.uniform(0.45, 0.9)
            loc = poi + dist * np.array([np.cos(azim) * np.cos(elev), np.sin(azim) * np.cos(elev), np.sin(elev)])
            look = poi + np.array([rng.uniform(-0.05, 0.05) for _ in range(3)])
            inplane = rng.uniform(-np.pi, np.pi) if view == "top_down" else rng.uniform(-0.3, 0.3)
            rot = bproc.camera.rotation_from_forward_vec(look - loc, inplane_rot=inplane)
            cam2world = bproc.math.build_transformation_mat(loc, rot)
            if set(target_objs) & bproc.camera.visible_objects(cam2world):
                break
        bproc.camera.add_camera_pose(cam2world)
        views.append({"view": view, "elevation_deg": round(float(np.degrees(elev)), 1),
                      "azimuth_deg": round(float(np.degrees(azim)), 1), "distance_m": round(float(dist), 3)})
    return views


# --------------------------------------------------------------------------- metadata
def attach_metadata(coco_path, first_new_image_id, data, targets):
    """Match each new COCO annotation to its scene instance (identical mask bbox),
    store the deformation metadata on the annotation and return CSV rows."""
    coco = json.loads(coco_path.read_text())
    entry_by_name = {obj.get_name(): entry for obj, entry, _ in targets}
    images = {im["id"]: im for im in coco["images"] if im["id"] >= first_new_image_id}
    rows = []
    for ann in coco["annotations"]:
        if ann["image_id"] not in images:
            continue
        frame = ann["image_id"] - first_new_image_id
        segmap = data["instance_segmaps"][frame]
        for inst in data["instance_attribute_maps"][frame]:
            entry = entry_by_name.get(inst.get("name"))
            if entry is None:
                continue
            mask = (segmap == inst["idx"]).astype(np.uint8)
            if mask.any() and _CocoWriterUtility.bbox_from_binary_mask(mask) == ann["bbox"]:
                ann["deformation"] = {"source_uid": entry["uid"], "deformed": entry["deformed"],
                                      "type": entry.get("type"), "level": entry.get("level"),
                                      "variant": entry.get("variant"), "params": entry.get("params")}
                rows.append({"image_id": ann["image_id"], "file_name": images[ann["image_id"]]["file_name"],
                             "annotation_id": ann["id"], "category": entry["class"], "bbox": json.dumps(ann["bbox"]),
                             "area": ann["area"], "source_uid": entry["uid"], "deformed": entry["deformed"],
                             "deform_type": entry.get("type") or "none", "deform_level": entry.get("level") or "none",
                             "deform_variant": entry.get("variant", ""),
                             "deform_params": json.dumps(entry.get("params") or {}),
                             "mesh": str(Path(entry["mesh"]).relative_to(ROOT))})
                break
    # Fixed, sorted category list (the writer only lists categories seen so far).
    coco["categories"] = [{"id": i, "name": n, "supercategory": "waste"}
                          for n, i in sorted(CATEGORIES.items(), key=lambda kv: kv[1])]
    coco_path.write_text(json.dumps(coco))
    return rows


def append_csv(path, rows):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=META_FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)


def scene_seed(seed, scene_idx):
    return zlib.crc32(f"{seed}-{scene_idx}".encode())


# --------------------------------------------------------------------------- main
def main():
    args = parse_args()
    out = args.out
    coco_dir, coco_path = out / "coco_data", out / "coco_data" / "coco_annotations.json"
    meta_csv, scene_jsonl = out / "instance_metadata.csv", out / "scene_metadata.jsonl"
    if args.overwrite and out.exists():
        shutil.rmtree(out)
    if coco_path.exists() and not args.resume:
        sys.exit(f"{out} already contains a dataset; pass --overwrite to regenerate or --resume to continue.")
    out.mkdir(parents=True, exist_ok=True)

    n_done = 0
    if args.resume and coco_path.exists():
        n_done = len(json.loads(coco_path.read_text())["images"])
    views_per_scene = RCFG["views_per_scene"]
    if n_done % views_per_scene:
        sys.exit("cannot resume from a partial scene; regenerate with --overwrite")
    scene_idx = n_done // views_per_scene

    bproc.init()
    bproc.camera.set_resolution(RCFG["width"], RCFG["height"])
    bproc.camera.set_intrinsics_from_blender_params(lens=np.radians(60), lens_unit="FOV")
    bproc.renderer.set_max_amount_of_samples(args.samples)
    bproc.renderer.set_noise_threshold(0.05)
    bproc.renderer.set_light_bounces(diffuse_bounces=2, glossy_bounces=2, max_bounces=4,
                                     transmission_bounces=2, transparent_max_bounces=4)
    bproc.renderer.enable_segmentation_output(map_by=["class", "instance", "name"], default_values={"class": 0})
    label_map = LabelIdMapping.from_dict(CATEGORIES)

    catalog = build_catalog()
    n_models = sum(len(v) for v in catalog.values())
    n_def = sum(len(var) for v in catalog.values() for _, var in v.values())
    print(f"[{args.dataset}] catalog: {n_models} models, {n_def} deformed variants, p_deformed={args.p_deformed}; "
          f"starting at image {n_done}")

    while n_done < args.num_images:
        seed = scene_seed(args.seed, scene_idx)
        rng = random.Random(seed)
        np.random.seed(seed % (2 ** 32))
        n_views = min(views_per_scene, args.num_images - n_done)
        bproc.utility.reset_keyframes()
        plane, targets, distractors, scene = build_scene(catalog, rng, args.p_deformed)
        views = sample_cameras(targets, n_views, rng)
        # BlenderProc assigns object pass indices only when segmentation is enabled,
        # so objects created for this scene must be (re-)indexed before rendering.
        for index, obj in enumerate(get_all_blender_mesh_objects()):
            obj.pass_index = index + 1
        data = bproc.renderer.render()
        bproc.writer.write_coco_annotations(str(coco_dir), instance_segmaps=data["instance_segmaps"],
                                            instance_attribute_maps=data["instance_attribute_maps"],
                                            colors=data["colors"], color_file_format="JPEG",
                                            mask_encoding_format="rle", label_mapping=label_map,
                                            append_to_existing_output=True)
        append_csv(meta_csv, attach_metadata(coco_path, n_done, data, targets))
        with open(scene_jsonl, "a") as f:
            for k, v in enumerate(views):
                f.write(json.dumps({"image_id": n_done + k, "scene": scene_idx, "scene_seed": seed, **v, **scene,
                                    "targets": [{"name": o.get_name(), "class": e["class"], "uid": e["uid"],
                                                 "deformed": e["deformed"], "type": e.get("type"),
                                                 "level": e.get("level"), "material": m}
                                                for o, e, m in targets]}) + "\n")
        n_done += n_views
        scene_idx += 1
        for obj in [plane] + [t[0] for t in targets] + distractors:
            obj.delete()
        for light in list(bpy.data.lights):
            bpy.data.lights.remove(light)
        bpy.ops.outliner.orphans_purge(do_recursive=True)
        print(f"[{args.dataset}] scene {scene_idx}: {len(targets)} targets, {len(distractors)} distractors "
              f"-> {n_done}/{args.num_images} images", flush=True)
    print(f"[{args.dataset}] done: {n_done} images -> {out}")


if __name__ == "__main__":
    main()

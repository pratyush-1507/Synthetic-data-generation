import blenderproc as bproc  # must be the first line for `blenderproc run`
# Render a contact sheet of every model at every deformation level
# (columns: intact | crush low/medium/high | bend low/medium/high, variant 0).
#
# Output: outputs/figures/model_grid_<class>.png
#
# Usage:
#     ./run_blenderproc.sh src/render_model_grid.py
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DEFORMED, NORM, load_config  # noqa: E402
from config import FIGURES as FIG  # noqa: E402

CFG = load_config()
COLUMNS = ["intact"] + [f"{t}_{l}" for t in ("crush", "bend") for l in ("low", "medium", "high")]
FOV = 0.45  # rad


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", nargs="+", default=list(CFG["classes"]))
    ap.add_argument("--cell", type=int, default=224)
    ap.add_argument("--samples", type=int, default=24)
    return ap.parse_args(argv)


def column_path(cls, uid, col):
    if col == "intact":
        return NORM / cls / uid / "model.obj"
    return DEFORMED / cls / uid / f"{col}_0.obj"


def look_at_pose(target, distance, azimuth=np.radians(35), elevation=np.radians(20)):
    d = np.array([np.cos(azimuth) * np.cos(elevation), -np.sin(azimuth) * np.cos(elevation), np.sin(elevation)])
    loc = target + distance * d
    rot = bproc.camera.rotation_from_forward_vec(target - loc)
    return bproc.math.build_transformation_mat(loc, rot)


def main():
    args = parse_args()
    bproc.init()
    bproc.camera.set_resolution(args.cell, args.cell)
    bproc.camera.set_intrinsics_from_blender_params(lens=FOV, lens_unit="FOV")
    bproc.renderer.set_max_amount_of_samples(args.samples)
    bproc.renderer.set_world_background([1.0, 1.0, 1.0], strength=0.6)
    for loc, energy in (([1, -1, 2], 4.0), ([-2, 1, 1], 1.5)):
        light = bproc.types.Light(light_type="SUN")
        light.set_location(loc)
        light.set_rotation_mat(bproc.camera.rotation_from_forward_vec(-np.array(loc, float)))
        light.set_energy(energy)

    FIG.mkdir(parents=True, exist_ok=True)
    for cls in args.classes:
        uids = sorted(p.name for p in (NORM / cls).iterdir() if (p / "model.obj").exists())
        sheet = Image.new("RGB", (args.cell * len(COLUMNS) + 90, args.cell * len(uids) + 28), "white")
        draw = ImageDraw.Draw(sheet)
        for c, col in enumerate(COLUMNS):
            draw.text((90 + c * args.cell + 6, 8), col.replace("_", " "), fill="black")
        for r, uid in enumerate(uids):
            bproc.utility.reset_keyframes()
            objs = []
            for c, col in enumerate(COLUMNS):
                path = column_path(cls, uid, col)
                if not path.exists():
                    continue
                parts = bproc.loader.load_obj(str(path))
                for p in parts:
                    p.set_location([c * 2.0, 0, 0])  # 2 m apart: neighbours stay out of frame
                objs.append(parts)
                pts = np.concatenate([p.get_bound_box() for p in parts])
                lo, hi = pts.min(0), pts.max(0)
                radius = np.linalg.norm(hi - lo) / 2
                bproc.camera.add_camera_pose(look_at_pose((lo + hi) / 2, 1.15 * radius / np.tan(FOV / 2)))
            data = bproc.renderer.render()
            for c, img in enumerate(data["colors"]):
                sheet.paste(Image.fromarray(img[..., :3]), (90 + c * args.cell, 28 + r * args.cell))
            draw.text((6, 28 + r * args.cell + args.cell // 2), uid[:8], fill="black")
            for parts in objs:
                for p in parts:
                    p.delete()
            print(f"[{cls}] rendered row {r + 1}/{len(uids)} ({uid})")
        out = FIG / f"model_grid_{cls}.png"
        sheet.save(out)
        print(f"saved {out}")


if __name__ == "__main__":
    main()

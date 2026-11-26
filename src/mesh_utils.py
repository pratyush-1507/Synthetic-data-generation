"""Shared mesh helpers for the download and normalisation steps."""
import numpy as np
import trimesh

MAX_ROUNDNESS_ERR = 0.15  # |width_a - width_b| / max(width) for round objects
BOX_SIDE_RATIO = 0.35     # min(width) / max(width) for box-shaped objects


def load_single_mesh(path):
    """Flatten a GLB scene (node transforms applied) into one Trimesh."""
    scene = trimesh.load(path, force="scene")
    meshes = [m for m in scene.dump(concatenate=False)
              if isinstance(m, trimesh.Trimesh) and len(m.faces)]
    if len(meshes) == 1:
        return meshes[0]
    try:
        return trimesh.util.concatenate(meshes)  # packs textures into an atlas
    except Exception:
        for m in meshes:  # mixed visuals that cannot be packed
            m.visual = m.visual.to_color()
        return trimesh.util.concatenate(meshes)


def _cross_section(extents, axis):
    return [extents[i] for i in range(3) if i != axis]


def height_axis(extents, shape="round"):
    """Pick the object's vertical axis.

    round: the axis whose perpendicular cross-section is closest to circular.
    box:   the longest axis (cartons stand on their smallest face).
    glTF's +Y wins near-ties in both cases.
    """
    if shape == "box":
        best = int(np.argmax(extents))
        if extents[1] * 1.15 >= extents[best]:
            best = 1
        u, v = _cross_section(extents, best)
        return best, 1 - min(u, v) / max(u, v, 1e-9)
    errs = []
    for a in range(3):
        u, v = _cross_section(extents, a)
        errs.append(abs(u - v) / max(u, v, 1e-9))
    best = int(np.argmin(errs))
    if errs[1] <= errs[best] + 0.05:
        best = 1
    return best, errs[best]


def shape_check(mesh, class_cfg):
    """Return (ok, reason, axis). Rejects multi-object scenes and implausible shapes."""
    shape = class_cfg.get("shape", "round")
    ext = mesh.extents
    axis, err = height_axis(ext, shape)
    h = ext[axis]
    w = max(_cross_section(ext, axis))
    lo, hi = class_cfg["aspect"]
    if shape == "round" and err > MAX_ROUNDNESS_ERR:
        return False, f"cross-section not round (err={err:.2f})", axis
    if shape == "box" and 1 - err < BOX_SIDE_RATIO:
        return False, f"footprint too flat (side ratio={1 - err:.2f})", axis
    if not lo <= w / h <= hi:
        return False, f"aspect {w / h:.2f} outside {lo}-{hi}", axis
    return True, "ok", axis

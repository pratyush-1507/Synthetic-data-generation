#!/usr/bin/env python3
"""Run the whole project pipeline (or part of it) with one command.

Stages, in order:
    download    src/download_objaverse.py      fetch models per class from Objaverse
    normalize   src/normalize.py               upright, real scale, OBJ
    deform      src/deform.py                  crush / bend variants (Blender)
    model_grid  src/render_model_grid.py       figure: every model at every level
    render      src/generate_scenes.py         synthetic datasets A_intact and B_deformed
    yolo        src/coco_to_yolo.py            YOLO labels + train/val/test split
    visualize   src/visualize_annotations.py   annotation overlays + statistics
    taco        src/prepare_taco.py            real test set (TACO) in COCO + YOLO
    train       src/train_yolo.py              YOLO11n on A_intact and B_deformed
    evaluate    src/evaluate.py                metrics on synthetic + real test sets
    report      src/make_report.py             result figures + outputs/results/results.md

Every stage reads configs/pipeline.yaml. Stages that are already complete skip
existing work (download, normalize, deform) or can be resumed (render --resume).
Logs are written to logs/<stage>.log.

Usage:
    .venv/bin/python run_pipeline.py --stages all
    .venv/bin/python run_pipeline.py --from render --to yolo
    .venv/bin/python run_pipeline.py --stages train evaluate report
    .venv/bin/python run_pipeline.py --stages all --dry-run
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from config import load_config  # noqa: E402

PY = str(ROOT / ".venv" / "bin" / "python")
BP = str(ROOT / "run_blenderproc.sh")
STAGES = ["download", "normalize", "deform", "model_grid", "render", "yolo", "visualize", "taco",
          "train", "evaluate", "report"]


def commands(stage, cfg, args):
    datasets = list(cfg["render"]["datasets"])
    if stage == "download":
        return [[PY, "src/download_objaverse.py"]]
    if stage == "normalize":
        return [[PY, "src/normalize.py"]]
    if stage == "deform":
        return [[BP, "src/deform.py"]]
    if stage == "model_grid":
        return [[BP, "src/render_model_grid.py"]]
    if stage == "render":
        mode = "--resume" if args.resume else "--overwrite"
        return [[BP, "src/generate_scenes.py", "--dataset", d, mode] for d in datasets]
    if stage == "yolo":
        return [[PY, "src/coco_to_yolo.py", "--dataset", f"data/synthetic/{d}"] for d in datasets]
    if stage == "visualize":
        return [[PY, "src/visualize_annotations.py", "--dataset", f"data/synthetic/{d}"] for d in datasets]
    if stage == "taco":
        return [[PY, "src/prepare_taco.py"],
                [PY, "src/coco_to_yolo.py", "--dataset", "data/real/taco", "--all-test"]]
    if stage == "train":
        return [[PY, "src/train_yolo.py", "--dataset", d] for d in datasets]
    if stage == "evaluate":
        return [[PY, "src/evaluate.py"]]
    if stage == "report":
        return [[PY, "src/make_report.py"]]
    raise ValueError(stage)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", nargs="+", help="stages to run, or 'all'")
    ap.add_argument("--from", dest="start", choices=STAGES, help="first stage to run")
    ap.add_argument("--to", dest="end", choices=STAGES, help="last stage to run")
    ap.add_argument("--resume", action="store_true", help="resume interrupted renders instead of overwriting")
    ap.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    args = ap.parse_args()

    if args.stages and args.stages != ["all"]:
        unknown = set(args.stages) - set(STAGES)
        if unknown:
            ap.error(f"unknown stages: {unknown}")
        stages = [s for s in STAGES if s in args.stages]
    else:
        i = STAGES.index(args.start) if args.start else 0
        j = STAGES.index(args.end) + 1 if args.end else len(STAGES)
        if not (args.stages or args.start or args.end):
            ap.error("pass --stages all, --stages <names...>, or --from/--to")
        stages = STAGES[i:j]

    cfg = load_config()
    (ROOT / "logs").mkdir(exist_ok=True)
    t0 = time.time()
    for stage in stages:
        log = ROOT / "logs" / f"{stage}.log"
        print(f"\n=== {stage} ===  (log: {log.relative_to(ROOT)})", flush=True)
        with open(os.devnull if args.dry_run else log, "w") as f:
            for cmd in commands(stage, cfg, args):
                pretty = " ".join(c.replace(str(ROOT) + "/", "") for c in cmd)
                print(f"  $ {pretty}", flush=True)
                if args.dry_run:
                    continue
                start = time.time()
                f.write(f"$ {pretty}\n")
                f.flush()
                result = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
                if result.returncode != 0:
                    print(f"  FAILED ({result.returncode}) - see {log.relative_to(ROOT)}")
                    sys.exit(result.returncode)
                print(f"    done in {(time.time() - start) / 60:.1f} min", flush=True)
    print(f"\npipeline finished in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()

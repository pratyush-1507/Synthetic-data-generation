"""Stage 7 - Train a YOLO11n detector on one synthetic dataset.

Trains with the settings in configs/pipeline.yaml (`train:`) on
data/synthetic/<dataset>/yolo/dataset.yaml. Ultralytics writes curves, confusion
matrices and weights to train/runs/<dataset>/; the best weights are also copied
to train/weights/<dataset>.pt.

Usage:
    python src/train_yolo.py --dataset B_deformed
    python src/train_yolo.py --dataset A_intact --epochs 60 --device cpu
"""
import argparse
import json
import shutil
import time

from ultralytics import YOLO

from config import SYNTHETIC, TRAIN, load_config


def main():
    cfg = load_config()
    t = cfg["train"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="dataset name under data/synthetic/")
    ap.add_argument("--model", default=t["model"])
    ap.add_argument("--epochs", type=int, default=t["epochs"])
    ap.add_argument("--imgsz", type=int, default=t["imgsz"])
    ap.add_argument("--batch", type=int, default=t["batch"])
    ap.add_argument("--device", default=t["device"])
    ap.add_argument("--patience", type=int, default=t["patience"])
    args = ap.parse_args()

    data = SYNTHETIC / args.dataset / "yolo" / "dataset.yaml"
    if not data.exists():
        raise SystemExit(f"{data} not found - run coco_to_yolo.py first")
    runs = TRAIN / "runs"
    start = time.time()
    model = YOLO(args.model)
    model.train(data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device,
                patience=args.patience, seed=cfg["seed"], deterministic=True, project=str(runs),
                name=args.dataset, exist_ok=True, plots=True, verbose=False)

    run_dir = runs / args.dataset
    (TRAIN / "weights").mkdir(parents=True, exist_ok=True)
    shutil.copy(run_dir / "weights" / "best.pt", TRAIN / "weights" / f"{args.dataset}.pt")
    summary = {"dataset": args.dataset, "model": args.model, "epochs": args.epochs, "imgsz": args.imgsz,
               "batch": args.batch, "device": args.device, "minutes": round((time.time() - start) / 60, 1),
               "weights": f"train/weights/{args.dataset}.pt", "run_dir": f"train/runs/{args.dataset}"}
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

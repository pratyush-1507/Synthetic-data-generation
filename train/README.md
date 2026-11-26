# Training

Detector: **YOLO11n** (Ultralytics, 2.6 M parameters), input 640 px, settings in
`configs/pipeline.yaml` (`train:`).

| Run | Training data | Purpose |
|-----|---------------|---------|
| **A_intact** | 1200 synthetic images, every object intact | baseline |
| **B_deformed** | the *same* 1200 scenes, ~60 % of objects crushed/bent | effect of deformation |

Both use the same scenes, cameras, lights and train/val/test split, so the only difference
is whether the objects are deformed.

```bash
.venv/bin/python run_pipeline.py --stages train          # both runs
.venv/bin/python src/train_yolo.py --dataset B_deformed  # one run
```

Outputs:
- `train/runs/<run>/` – Ultralytics run: `results.csv`, loss/mAP curves (`results.png`),
  confusion matrix, PR curves, `weights/best.pt`, `train_summary.json`
- `train/weights/<run>.pt` – copy of the best weights used by `src/evaluate.py`

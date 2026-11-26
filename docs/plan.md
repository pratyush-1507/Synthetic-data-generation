# Project plan

Project 2 (AI512): *Synthetic Deformable Waste Data Generation for Sim-to-Real Perception*

| Phase | Goal | Deliverables | Status |
|------|------|--------------|--------|
| 1 | Assets: download and normalise models from Objaverse | `download_objaverse.py`, `normalize.py`, `LICENSES.csv` | Done (4 classes × 8 models) |
| 2 | Procedural deformation (crush/bend × 3 levels, seeded) | `deform.py`, 384 deformed meshes + JSON, model grids | Done |
| 3 | Scene generation + automatic annotation (pilot: 50 images, 2 classes) | `generate_scenes.py`, `coco_to_yolo.py`, `visualize_annotations.py` | Done |
| 4 | Pipeline upgrade: central config, 4 classes, class-aware shape checks, seeded/resumable scenes, scene-level split, one-command `run_pipeline.py` | `configs/pipeline.yaml`, updated scripts | Done |
| 5 | Datasets: A_intact and B_deformed, 1200 images each (each class in ~530-600 images) | `data/synthetic/{A_intact,B_deformed}` | Done |
| 6 | Real test set: TACO mapped to the 4 classes | `prepare_taco.py`, `data/real/taco` | Done |
| 7 | Training: YOLO11n on A and on B (identical settings) | `train_yolo.py`, `train/` | Done (45 + 30 min) |
| 8 | Evaluation + report: synthetic and real test sets, recall by deformation, figures | `evaluate.py`, `make_report.py`, `outputs/` | Done |

## Experimental design

- **Hypothesis**: training on procedurally deformed objects improves detection of real
  deformed waste (TACO) compared with training on intact objects only.
- **Control**: dataset A – identical scenes (same seeds, cameras, lights, models) with all
  objects intact. Only the deformation differs between A and B.
- **Test sets**: synthetic test split of B (deformed objects), of A (intact objects), and the
  real TACO photos (never used for training).
- **Metrics**: precision, recall, mAP50, mAP50-95 overall and per class; recall per
  deformation type × level on the synthetic test scenes.
- **Possible follow-ups**: more models per class, realistic backgrounds (HDRI / photo
  textures), dirt/label-damage textures, deformation ratio ablation, larger detector.

## Risks

| Risk | Mitigation |
|------|------------|
| Few clean Objaverse models for cups/cartons | name search beyond LVIS, manual visual review, exclusion list in config |
| Large sim-to-real gap (clean renders vs. real litter) | domain randomisation; report gap honestly; A/B comparison is relative |
| Small TACO counts for cup/carton | report per-class AP with counts; use overall mAP50 as main metric |
| Training time on a laptop | YOLO11n, 60 epochs, early stopping (patience 20) |

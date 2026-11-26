# Real data: TACO test set

Real-world evaluation set: **TACO – Trash Annotations in Context** (http://tacodataset.org,
https://github.com/pedropro/TACO), 1500 photos of litter in the wild, many of them crushed
or bent.

Built by `src/prepare_taco.py` (or `run_pipeline.py --stages taco`):

1. downloads the official `annotations.json` and every image (the 640 px rendition, or the
   original with its EXIF rotation applied and downscaled to 640 px),
2. maps TACO categories onto the project classes (from `configs/pipeline.yaml`):

   | Class | TACO categories |
   |---|---|
   | bottle | Clear plastic bottle, Glass bottle, Other plastic bottle |
   | can | Drink can, Food Can |
   | cup | Disposable plastic cup, Foam cup, Paper cup, Other plastic cup |
   | carton | Drink carton, Other carton |

   All other categories (cigarettes, wrappers, film, …) are dropped. Images without any of
   our classes are kept, so false positives on other litter are counted.
3. writes `taco/coco_data/coco_annotations.json` (+ `images/`), then
   `coco_to_yolo.py --all-test` builds `taco/yolo/` with every image in the **test** split.

TACO is used **only for testing**, never for training: it measures the sim-to-real gap.
TACO images are licensed by their Flickr owners; see the TACO repository for terms.

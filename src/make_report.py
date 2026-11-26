"""Stage 9 - Turn the evaluation CSVs into presentation figures and a results table.

Reads outputs/results/*.csv (from evaluate.py) and the dataset metadata, writes
    outputs/figures/results_map50.png             A vs B mAP50 on each test set
    outputs/figures/results_taco_per_class.png    A vs B AP50 per class on real TACO
    outputs/figures/results_recall_deformation.png  recall by deformation level (synthetic)
    outputs/figures/results_training_curves.png   validation mAP50-95 and box loss per epoch
    outputs/results/results.md                    tables for the README / slides

Usage:
    python src/make_report.py
"""
import csv
import json
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from config import FIGURES, REAL, RESULTS, SYNTHETIC, TRAIN, class_names, load_config  # noqa: E402

# Reference categorical palette (validated): slot 1 blue, slot 2 orange.
SERIES = {"A_intact": "#2a78d6", "B_deformed": "#eb6834"}
LABELS = {"A_intact": "A: intact only", "B_deformed": "B: intact + deformed"}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
TEST_LABELS = {"synth_B": "Synthetic test\n(deformed objects)", "synth_A": "Synthetic test\n(intact objects)",
               "taco": "Real photos\n(TACO)"}


def read(name):
    with open(RESULTS / name) as f:
        return list(csv.DictReader(f))


def style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=30)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.tick_params(colors=INK_2, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def grouped_bars(groups, values, title, ylabel, path, ylim=1.0):
    """values: {model: [v per group]}; two series, labelled directly + legend."""
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    n, width = len(groups), 0.36
    for i, (m, vals) in enumerate(values.items()):
        xs = [g + (i - 0.5) * (width + 0.02) for g in range(n)]
        bars = ax.bar(xs, vals, width, color=SERIES[m], label=LABELS[m], edgecolor=SURFACE, linewidth=2)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + ylim * 0.015, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=9, color=INK)
    ax.set_xticks(range(n), groups)
    ax.set_ylim(0, ylim)
    style(ax, title, ylabel)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, labelcolor=INK,
              borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def training_curves(models, path):
    """Validation mAP50-95 and training box loss per epoch (two panels, one axis each)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), facecolor=SURFACE)
    panels = [("metrics/mAP50-95(B)", "Validation mAP50-95 (own synthetic val split)", "mAP50-95"),
              ("train/box_loss", "Training box loss", "box loss")]
    for ax, (col, title, ylabel) in zip(axes, panels):
        for m in models:
            p = TRAIN / "runs" / m / "results.csv"
            if not p.exists():
                continue
            with open(p) as f:
                rows = list(csv.DictReader(f))
            ep = [int(r["epoch"]) for r in rows]
            ax.plot(ep, [float(r[col]) for r in rows], color=SERIES[m], linewidth=2, label=LABELS[m])
        style(ax, title, ylabel)
        ax.set_xlabel("epoch", color=INK_2)
    axes[0].legend(frameon=False, loc="lower right", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def dataset_stats(cfg):
    names = class_names(cfg)
    rows = []
    for ds in cfg["render"]["datasets"]:
        coco = json.loads((SYNTHETIC / ds / "coco_data" / "coco_annotations.json").read_text())
        cat = {c["id"]: c["name"] for c in coco["categories"]}
        inst = Counter(cat[a["category_id"]] for a in coco["annotations"])
        imgs = defaultdict(set)
        deformed = Counter()
        for a in coco["annotations"]:
            imgs[cat[a["category_id"]]].add(a["image_id"])
            deformed[cat[a["category_id"]]] += bool((a.get("deformation") or {}).get("deformed"))
        for n in names:
            rows.append({"dataset": ds, "class": n, "images": len(imgs[n]), "instances": inst[n],
                         "deformed": deformed[n], "total_images": len(coco["images"])})
    taco = json.loads((REAL / "taco" / "coco_data" / "coco_annotations.json").read_text())
    tcat = {c["id"]: c["name"] for c in taco["categories"]}
    tinst = Counter(tcat[a["category_id"]] for a in taco["annotations"])
    timgs = defaultdict(set)
    for a in taco["annotations"]:
        timgs[tcat[a["category_id"]]].add(a["image_id"])
    for n in names:
        rows.append({"dataset": "taco (real)", "class": n, "images": len(timgs[n]), "instances": tinst[n],
                     "deformed": "", "total_images": len(taco["images"])})
    return rows


def md_table(rows, cols):
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(out)


def main():
    cfg = load_config()
    names = class_names(cfg)
    overall, per_class, deform = read("metrics_overall.csv"), read("metrics_per_class.csv"), \
        read("recall_by_deformation.csv")
    models = list(SERIES)

    # 1. mAP50 per test set
    sets = list(TEST_LABELS)
    vals = {m: [float(next(r["mAP50"] for r in overall if r["model"] == m and r["test_set"] == s)) for s in sets]
            for m in models}
    grouped_bars([TEST_LABELS[s] for s in sets], vals, "Detection accuracy (mAP50) by test set", "mAP50",
                 FIGURES / "results_map50.png")

    # 2. TACO per-class AP50
    def ap(m, c):
        r = [x for x in per_class if x["model"] == m and x["test_set"] == "taco" and x["class"] == c]
        return float(r[0]["AP50"]) if r else 0.0
    grouped_bars(names, {m: [ap(m, c) for c in names] for m in models},
                 "Real-world TACO photos: AP50 per class", "AP50", FIGURES / "results_taco_per_class.png")

    # 3. Recall by deformation level (synthetic test, all classes)
    groups = [("intact", "intact")] + [(t, lv) for t in ("crush", "bend") for lv in ("low", "medium", "high")]

    def rec(m, t, lv):
        r = [x for x in deform if x["model"] == m and x["class"] == "all" and x["deform_type"] == t
             and x["deform_level"] == lv]
        return float(r[0]["recall"]) if r and r[0]["recall"] != "" else 0.0
    grouped_bars(["intact" if t == "intact" else f"{t}\n{lv}" for t, lv in groups],
                 {m: [rec(m, t, lv) for t, lv in groups] for m in models},
                 "Recall on synthetic test objects by deformation (IoU 0.5, conf 0.25)", "recall",
                 FIGURES / "results_recall_deformation.png", ylim=1.05)

    # 4. Training curves
    training_curves(models, FIGURES / "results_training_curves.png")

    # Results markdown
    stats = dataset_stats(cfg)
    train_info = []
    for m in models:
        p = TRAIN / "runs" / m / "train_summary.json"
        if p.exists():
            train_info.append(json.loads(p.read_text()))
    md = ["# Results", "", "## Datasets", "", md_table(stats, ["dataset", "class", "images", "instances",
                                                               "deformed", "total_images"]),
          "", "## Training", ""]
    if train_info:
        md += [md_table(train_info, ["dataset", "model", "epochs", "imgsz", "batch", "device", "minutes"]), ""]
    md += ["## Overall metrics", "", md_table(overall, ["model", "test_set", "precision", "recall", "mAP50",
                                                         "mAP50_95"]),
           "", "## Per class", "", md_table(per_class, ["model", "test_set", "class", "precision", "recall",
                                                           "AP50", "AP50_95"]),
           "", "## Recall by deformation (synthetic test scenes, all classes)", "",
           md_table([r for r in deform if r["class"] == "all"],
                    ["model", "deform_type", "deform_level", "objects", "detected", "recall"]),
           "", "Figures: `outputs/figures/results_map50.png`, `results_taco_per_class.png`, "
           "`results_recall_deformation.png`, `predictions_taco.jpg`, `predictions_synthetic.jpg`."]
    (RESULTS / "results.md").write_text("\n".join(md) + "\n")
    with open(RESULTS / "dataset_stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stats[0]))
        w.writeheader()
        w.writerows(stats)
    print(f"wrote {RESULTS / 'results.md'} and 4 result figures")


if __name__ == "__main__":
    main()

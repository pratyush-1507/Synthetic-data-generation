# Results

## Datasets

| dataset | class | images | instances | deformed | total_images |
|---|---|---|---|---|---|
| B_deformed | bottle | 593 | 737 | 424 | 1200 |
| B_deformed | can | 529 | 669 | 392 | 1200 |
| B_deformed | cup | 556 | 714 | 432 | 1200 |
| B_deformed | carton | 564 | 688 | 427 | 1200 |
| A_intact | bottle | 592 | 723 | 0 | 1200 |
| A_intact | can | 529 | 651 | 0 | 1200 |
| A_intact | cup | 535 | 700 | 0 | 1200 |
| A_intact | carton | 566 | 689 | 0 | 1200 |
| taco (real) | bottle | 322 | 439 |  | 1500 |
| taco (real) | can | 169 | 263 |  | 1500 |
| taco (real) | cup | 158 | 186 |  | 1500 |
| taco (real) | carton | 115 | 138 |  | 1500 |

## Training

| dataset | model | epochs | imgsz | batch | device | minutes |
|---|---|---|---|---|---|---|
| A_intact | yolo11n.pt | 60 | 640 | 16 | mps | 31.3 |
| B_deformed | yolo11n.pt | 60 | 640 | 16 | mps | 45.1 |

## Overall metrics

| model | test_set | precision | recall | mAP50 | mAP50_95 |
|---|---|---|---|---|---|
| A_intact | synth_B | 0.909 | 0.81 | 0.8886 | 0.8676 |
| A_intact | synth_A | 0.9642 | 0.8811 | 0.9654 | 0.9368 |
| A_intact | taco | 0.1557 | 0.1116 | 0.0423 | 0.0334 |
| B_deformed | synth_B | 0.9096 | 0.871 | 0.9355 | 0.9135 |
| B_deformed | synth_A | 0.9328 | 0.8984 | 0.9533 | 0.9191 |
| B_deformed | taco | 0.1372 | 0.1369 | 0.0514 | 0.0374 |

## Per class

| model | test_set | class | precision | recall | AP50 | AP50_95 |
|---|---|---|---|---|---|---|
| A_intact | synth_B | bottle | 0.9508 | 0.8277 | 0.9131 | 0.9022 |
| A_intact | synth_B | can | 0.8533 | 0.821 | 0.8937 | 0.8785 |
| A_intact | synth_B | cup | 0.8463 | 0.6719 | 0.771 | 0.7371 |
| A_intact | synth_B | carton | 0.9857 | 0.9193 | 0.9765 | 0.9526 |
| A_intact | synth_A | bottle | 0.9884 | 0.9118 | 0.9802 | 0.9527 |
| A_intact | synth_A | can | 1.0 | 0.8113 | 0.9758 | 0.9567 |
| A_intact | synth_A | cup | 0.8848 | 0.8671 | 0.9201 | 0.8831 |
| A_intact | synth_A | carton | 0.9838 | 0.9342 | 0.9853 | 0.9546 |
| A_intact | taco | bottle | 0.0832 | 0.0405 | 0.0174 | 0.0118 |
| A_intact | taco | can | 0.1353 | 0.2129 | 0.0684 | 0.0544 |
| A_intact | taco | cup | 0.3144 | 0.0699 | 0.0569 | 0.0453 |
| A_intact | taco | carton | 0.0898 | 0.1232 | 0.0263 | 0.0222 |
| B_deformed | synth_B | bottle | 0.9656 | 0.9143 | 0.9708 | 0.9501 |
| B_deformed | synth_B | can | 0.857 | 0.8706 | 0.9194 | 0.9037 |
| B_deformed | synth_B | cup | 0.8336 | 0.7656 | 0.868 | 0.8357 |
| B_deformed | synth_B | carton | 0.9823 | 0.9333 | 0.9836 | 0.9644 |
| B_deformed | synth_A | bottle | 0.9927 | 0.9118 | 0.9748 | 0.9227 |
| B_deformed | synth_A | can | 0.8978 | 0.9092 | 0.9551 | 0.9437 |
| B_deformed | synth_A | cup | 0.8666 | 0.8385 | 0.9053 | 0.8688 |
| B_deformed | synth_A | carton | 0.9742 | 0.9342 | 0.9779 | 0.9413 |
| B_deformed | taco | bottle | 0.0631 | 0.0911 | 0.032 | 0.0215 |
| B_deformed | taco | can | 0.1213 | 0.2433 | 0.0719 | 0.0524 |
| B_deformed | taco | cup | 0.2564 | 0.0753 | 0.0679 | 0.0503 |
| B_deformed | taco | carton | 0.1081 | 0.1377 | 0.0341 | 0.0255 |

## Recall by deformation (synthetic test scenes, all classes)

| model | deform_type | deform_level | objects | detected | recall |
|---|---|---|---|---|---|
| A_intact | bend | all | 94 | 88 | 0.9362 |
| A_intact | bend | high | 43 | 40 | 0.9302 |
| A_intact | bend | low | 26 | 25 | 0.9615 |
| A_intact | bend | medium | 25 | 23 | 0.92 |
| A_intact | crush | all | 89 | 59 | 0.6629 |
| A_intact | crush | high | 31 | 14 | 0.4516 |
| A_intact | crush | low | 29 | 26 | 0.8966 |
| A_intact | crush | medium | 29 | 19 | 0.6552 |
| A_intact | intact | intact | 111 | 103 | 0.9279 |
| B_deformed | bend | all | 94 | 87 | 0.9255 |
| B_deformed | bend | high | 43 | 39 | 0.907 |
| B_deformed | bend | low | 26 | 26 | 1.0 |
| B_deformed | bend | medium | 25 | 22 | 0.88 |
| B_deformed | crush | all | 89 | 71 | 0.7978 |
| B_deformed | crush | high | 31 | 16 | 0.5161 |
| B_deformed | crush | low | 29 | 28 | 0.9655 |
| B_deformed | crush | medium | 29 | 27 | 0.931 |
| B_deformed | intact | intact | 111 | 104 | 0.9369 |

Figures: `outputs/figures/results_map50.png`, `results_taco_per_class.png`, `results_recall_deformation.png`, `predictions_taco.jpg`, `predictions_synthetic.jpg`.

# Road Sign Detection with YOLO 


## Task
Detect four kinds of road signs (stop, speed limit, crosswalk, traffic light) in photos: for each object the model predicts a bounding box, a class name and a confidence score. We fine-tune a small pretrained detector (yolo26n.pt, Ultralytics) on a public dataset.

## Dataset
* Source: Kaggle, [Road Sign Detection](https://www.kaggle.com/datasets/andrewmvd/road-sign-detection) (andrewmvd / Larxel).
* Classes: stop, speedlimit, crosswalk, trafficlight. Annotation format: Pascal VOC `.xml` (converted to YOLO `.txt` by the notebook).
* Split: images are shuffled with `random.Random(42)` and cut 70% / 15% / 15% into train / val / test. The same seed gives the same split on every run.
* Images and objects per split:

| split | images | objects | stop | speedlimit | crosswalk | trafficlight |
|---|---|---|---|---|---|---|
| train | 613 | 877 | 56 | 545 | 147 | 129 |
| val | 132 | 174 | 21 | 114 | 24 | 15 |
| test | 132 | 193 | 14 | 124 | 29 | 26 |

* Checks: 0 label problems found (missing files, empty labels, boxes outside the image, unknown classes). 2 groups of exact duplicate images exist, 2 of them entirely inside one split; 0 near-duplicate pairs (perceptual hash distance <= 4) cross different splits.
* Ground-truth examples: `results/gt_examples.png`.

## Setup and run
1. Open the notebook, choose `Runtime -> Change runtime type -> T4 GPU`.
2. `Runtime -> Run all`. The notebook installs the pinned versions from `requirements.txt`, downloads the dataset with `kagglehub` (no token needed), creates the split, trains both models, evaluates and shows predictions.
3. Set `SMOKE_TEST = True` in the setup cell for a 1-epoch dry run.

## Training (baseline, run 1)
| setting | value |
|---|---|
| framework | ultralytics 8.4.215 |
| torch | 2.14.1+cu130 |
| model / initial weights | yolo26n.pt |
| image size | 640 |
| batch size | 16 |
| epochs | 30 |
| optimizer | AdamW |
| learning rate (lr0) | 0.001 |
| seed | 42 |
| training time (baseline) | 5.4 min on Tesla T4 |

Outputs per run are in `results/<run>/`: `args.yaml`, `results.csv`, `results.png`, `run_info.json`, plots. Best checkpoints are selected by validation fitness.

## Experiment (run 2)
Hypothesis (stated before the results): Reducing the input resolution from 640 to 320 will make training faster but will lower recall and mAP@0.50, especially for small objects such as traffic lights, because they keep only a few pixels after downscaling.

Changed setting: imgsz=320. Everything else (split, initial weights, batch, epochs, optimizer, learning rate, seed) is identical.

Validation results (conf=0.001, NMS IoU=0.7):

| run | changed setting | precision | recall | mAP@0.50 | train time (min) |
|---|---|---|---|---|---|
| run1_baseline | none (baseline) | 0.981 | 0.877 | 0.925 | 5.4 |
| run2_experiment | imgsz=320 | 0.934 | 0.9 | 0.93 | 3.67 |

Learning curves: `results/learning_curves_comparison.png`.


## Final test results
Final run chosen by validation mAP@0.50: **run2_experiment**. Its best checkpoint was evaluated once on the test set (132 images, 193 objects), conf=0.001, NMS IoU=0.7, image size 320.

| run | precision | recall | mAP@0.50 | conf | IoU (NMS) | imgsz |
|---|---|---|---|---|---|---|
| run2_experiment | 0.905 | 0.845 | 0.862 | 0.001 | 0.7 | 320 |

Per class:

| class | precision | recall | mAP@0.50 |
|---|---|---|---|
| stop | 0.983 | 1.0 | 0.995 |
| speedlimit | 0.984 | 0.97 | 0.985 |
| crosswalk | 0.951 | 0.862 | 0.9 |
| trafficlight | 0.703 | 0.547 | 0.569 |

Metrics: precision = share of predicted boxes that are correct; recall = share of real objects that were found; mAP@0.50 = mean over classes of the area under the precision-recall curve, a prediction counting as correct when the class matches and IoU >= 0.50.

Predictions: `results/test_predictions.png` (confidence >= 0.25). Failure cases: `results/failure_cases.png`, `results/test_failures.csv`.


## Limitations
* The test set is small (132 images, 193 objects); class `stop` has only 56 training and 14 test objects, so its metrics are noisy.
* Classes are imbalanced (speed limit signs dominate).
* One training run per setting and one seed: differences between runs can be partly random.
* The split is random per image; the dataset has no scene or video ids, so very similar scenes could still sit in different splits (a perceptual-hash check found 0 such pairs).
* Small model and limited epochs because of the compute budget.


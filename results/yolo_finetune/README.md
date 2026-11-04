# CCT-20 YOLOv8n Fine-Tuning Results

This directory contains the compact, version-controlled artifacts from fine-tuning YOLOv8n as a single-class animal detector on the CCT-20 camera-trap dataset.

The complete Ultralytics run directory, generated YOLO dataset, caches, and model binaries are intentionally excluded from normal Git history.

## Experiment

The model starts from the COCO-pretrained YOLOv8n checkpoint and is fine-tuned for one detection class:

```text
0 = animal
```

All valid CCT-20 wildlife categories are collapsed into this class. Species classification is outside the scope of this detector-training experiment.

## Frozen Dataset Split

Training uses the committed location-disjoint CCT-20 split. Camera locations and capture sequences do not cross partition boundaries.

| Split | Camera locations | Images | Animal boxes | Empty images |
|---|---:|---:|---:|---:|
| Train | 14 | 32,029 | 30,419 | 2,731 |
| Validation | 3 | 7,441 | 7,404 | 585 |
| Test | 3 | 6,761 | 6,431 | 601 |

The held-out test partition is not used for training, early stopping, checkpoint selection, hyperparameter selection, or confidence-threshold selection.

## Dataset Preparation

The YOLO dataset is generated from the frozen split manifest with:

```bash
python scripts/prepare_yolo_dataset.py \
  --manifest data/cct20_splits/split_manifest.csv \
  --annotations data/cct20/caltech_bboxes_20200316.json \
  --output data/cct20_yolo
```

Dataset preparation:

- rescales boxes from CCT annotation dimensions to the actual downsized image dimensions;
- converts COCO-style boxes to normalized YOLO coordinates;
- collapses all wildlife categories to `animal`;
- creates explicit empty label files for known-empty frames;
- preserves the frozen train/validation/test assignment;
- validates image/label correspondence and normalized coordinate ranges.

The completed conversion produced:

```text
46,231 images
44,254 animal boxes
3,917 known-empty images
42,314 animal images requiring coordinate rescaling
0 box-count mismatches
```

Ultralytics subsequently scanned the training and validation sets with zero corrupt images.

## Training Configuration

The training job is defined by:

```text
scripts/sbatch-yolo-finetuning
```

Key parameters:

| Parameter | Value |
|---|---|
| Base model | YOLOv8n |
| Classes | 1 (`animal`) |
| Image size | 640 |
| Batch size | 64 |
| Optimizer | AdamW |
| Initial learning rate | 0.001 |
| Weight decay | 0.0005 |
| LR schedule | Cosine |
| Maximum epochs | 100 |
| Early-stopping patience | 20 |
| Seed | 42 |
| Deterministic training | enabled |

Training environment:

```text
Ultralytics 8.4.198
Python 3.11.18
PyTorch 2.14.1+cu126
GPU: Tesla V100S-PCIE-32GB
```

The complete Ultralytics training configuration is preserved in `yolov8n_cct_animal_640_args.yaml`.

## Training Outcome

Training stopped automatically after **61 epochs** because validation performance did not improve for 20 consecutive epochs.

The best checkpoint was selected at **epoch 41**.

Validation of the selected `best.pt` checkpoint produced:

| Metric | Value |
|---|---:|
| Precision | 0.922 |
| Recall | 0.786 |
| mAP50 | 0.858 |
| mAP50-95 | 0.557 |

These are **validation results**, not final held-out test results.

The per-epoch training trajectory is preserved in `yolov8n_cct_animal_640_results.csv`.

## Files in This Directory

### `yolov8n_cct_animal_640_args.yaml`

Ultralytics configuration captured from the completed fine-tuning run.

### `yolov8n_cct_animal_640_results.csv`

Per-epoch training and validation metrics, including:

- training box, classification, and DFL losses;
- validation losses;
- precision and recall;
- mAP50 and mAP50-95;
- learning rates.

### `README.md`

This experiment record and artifact documentation.

## Selected Checkpoint

- Model: YOLOv8n
- Best epoch: 41
- File: `best.pt`
- SHA-256: `80c94593b0a222ad7d1b67e3273be73c16eb7ba9d85d01bae755db0964660a7b`
- Original training-run path: `runs/detect/experiments/yolo_finetune/yolov8n_cct_animal_640/weights/best.pt`
- Validation precision: 0.922
- Validation recall: 0.786
- Validation mAP50: 0.858
- Validation mAP50-95: 0.557

The original training-run path is not a portable repository path. The checkpoint itself is intentionally not stored in normal Git history and should be shared through Git LFS, a GitHub Release asset, or project storage.

Anyone using the checkpoint for subsequent evaluation should verify its SHA-256 digest against the value above.

## Experimental Boundary

Model development follows this protocol:

```text
14 training camera locations
        |
        v
YOLO fine-tuning
        |
        v
3 validation camera locations
        |
        +-- early stopping
        +-- checkpoint selection
        +-- confidence-threshold selection
        |
        v
freeze model and settings
        |
        v
3 held-out test camera locations
        |
        v
final evaluation
```

The test set must remain untouched until the checkpoint and operating threshold have been frozen using validation data only.

## Next Step

The next step is to choose the operating confidence threshold on the validation partition, then evaluate the selected checkpoint on the three untouched test camera locations.

After the detector and threshold are frozen, the same model can be used for the project's final enhancement comparison:

```text
CCT-fine-tuned YOLOv8n
        |
        +-- raw
        +-- CLAHE
        +-- SNR-Aware
        +-- RetinexFormer
```

This will test whether the low-light enhancement conclusions observed with the generic COCO-pretrained detector persist after camera-trap domain adaptation.

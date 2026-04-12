# Notebook Structure Guideline

This document defines the standard structure for all experiment notebooks in this project. Every new notebook **must** follow this layout exactly to keep experiments comparable and easy to read.

---

## Cell Order

Each section consists of a **markdown header cell** (just the `##` title) followed immediately by the **code cell**. The only exception is Cell 1 — the experiment intro — which has no code and stands alone.

The full cell sequence is:

| # | Type | Content |
|---|---|---|
| 1 | Markdown | Experiment intro (Objective, Architecture table, Hypothesis) |
| 2 | Markdown | `## Import libraries, set seed, and choose device` |
| 3 | Code | Setup |
| 4 | Markdown | `## Load and split data` |
| 5 | Code | Data loading |
| 6 | Markdown | `## Model Definition` |
| 7 | Code | Model definition |
| 8 | Markdown | `## Training Loop` |
| 9 | Code | Training loop |
| 10 | Markdown | `## Plot Train and Validation Curves` |
| 11 | Code | Training curves |
| 12 | Markdown | `## Threshold Tuning (Best Val F2)` |
| 13 | Code | Threshold tuning |
| 14 | Markdown | `## Test Set Evaluation` |
| 15 | Code | Test evaluation |

Use these exact `##` titles — they should match across all notebooks for consistency.

---

### Cell 1 — Markdown: Experiment Header

The first cell is always a markdown cell with three sections:

**a. Objective** — one paragraph stating what this experiment is testing and why, relative to the previous iteration.

**b. Architecture Changes** — a table comparing this iteration against the previous one. Include every parameter that changed.

Example:

| Component | Previous iteration | This iteration |
|---|---|---|
| Unfrozen layers | `layer4[2]` only | `layer4` (all 3 blocks) |
| Trainable params | ~4.7M | ~15M |
| Weight decay | 1e-4 | 1e-3 |
| L1 lambda | 0 | 1e-3 |
| L2 lambda | 0 | 0 |
| Dropout | 0.5 | 0 |
| Epochs | 30 | 30 |

**c. Hypothesis** — one paragraph predicting what effect the changes will have and why.

---

### Cells 2–3 — Setup: Imports, Seed, Device

**Cell 2 (markdown):**
```
## Import libraries, set seed, and choose device
```

**Cell 3 (code)** — all three in a single cell, do not split:

```python
import sys
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

ROOT = next(p for p in [Path.cwd()] + list(Path.cwd().parents) if (p / 'src').exists())
sys.path.insert(0, str(ROOT))

from src.data.dataset import HAM10000Dataset
from src.data.dataloader import get_dataloaders
from src.data.transform import get_augmented_train_transforms
from src.models.<arch> import <model_factory>
from src.training.trainer import train_one_epoch, validate_one_epoch
from src.utils import plot_training_curves, find_best_threshold, evaluate_model

import pandas as pd

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f'Using device: {device}')
```

---

### Cells 4–5 — Data Loading

**Cell 4 (markdown):**
```
## Load and split data
```

**Cell 5 (code)** — load train, val, and test splits. Build the train loader manually (not via `get_dataloaders`) so that `num_workers > 0` and `persistent_workers=True` can be set — this prevents the augmentation pipeline from being the bottleneck.

```python
train_dataset = HAM10000Dataset(
    csv_path=str(ROOT / 'data_new/splits/train.csv'),
    image_dir=str(ROOT / 'data_new/images/train'),
    transform=get_augmented_train_transforms(image_size=224),
)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    persistent_workers=True,
)

_, val_loader, test_loader = get_dataloaders(
    train_csv=str(ROOT / 'data_new/splits/train.csv'),
    val_csv=str(ROOT / 'data_new/splits/val.csv'),
    test_csv=str(ROOT / 'data_new/splits/test.csv'),
    image_dir=str(ROOT / 'data_new/images/train'),
    test_image_dir=str(ROOT / 'data_new/images/test'),
    batch_size=32,
    image_size=224,
    num_workers=4,
)

train_df     = pd.read_csv(ROOT / 'data_new/splits/train.csv')
num_melanoma = (train_df['label'] == 1).sum()
num_nevus    = (train_df['label'] == 0).sum()
pos_weight   = torch.tensor([num_nevus / num_melanoma], dtype=torch.float32).to(device)
print('Positive weight:', pos_weight)
```

This cell is the same for every notebook. Do not change it unless the dataset changes.

---

### Cells 6–7 — Model Definition

**Cell 6 (markdown):**
```
## Model Definition
```

**Cell 7 (code)** — everything experiment-specific goes here: model construction, loss, optimiser, scheduler, regularisation lambdas, and trainable parameter count. Nothing shared — this is the cell that changes between iterations.

```python
model = get_resnet50(num_classes=1, freeze_backbone=True, dropout=0.0).to(device)

# Unfreeze the layers specific to this experiment
for param in model.layer4.parameters():
    param.requires_grad = True

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = optim.AdamW([
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(),     'lr': 1e-3},
], weight_decay=1e-3)

num_epochs = 30
scheduler  = CosineAnnealingLR(optimizer, T_max=num_epochs)

L1_LAMBDA = 1e-3
L2_LAMBDA = 0.0

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f'Trainable params: {trainable:,} / {total:,}')
print(f'L1 lambda: {L1_LAMBDA} | L2 lambda: {L2_LAMBDA} | Dropout: {dropout}')
```

---

### Cells 8–9 — Training Loop

**Cell 8 (markdown):**
```
## Training Loop
```

**Cell 9 (code)** — save the best model by **validation AUC-ROC** (not F2). AUC is threshold-independent and measures the model's raw discriminative ability — see `notebooks/resnet50/resnet50_summary.md` for the full rationale.

Print all five metrics per epoch in multi-line format:

```python
best_val_auc = 0.0
train_history, val_history = [], []

for epoch in range(num_epochs):
    train_metrics = train_one_epoch(
        model, train_loader, criterion, optimizer, device,
        l1_lambda=L1_LAMBDA, l2_lambda=L2_LAMBDA,
    )
    val_metrics = validate_one_epoch(model, val_loader, criterion, device)

    scheduler.step()

    train_history.append(train_metrics)
    val_history.append(val_metrics)

    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"  Train | Loss: {train_metrics['loss']:.4f}, Bal Acc: {train_metrics['balanced_accuracy']:.4f}, Recall: {train_metrics['recall']:.4f}, F2: {train_metrics['f2']:.4f}, AUC: {train_metrics['auc']:.4f}")
    print(f"  Val   | Loss: {val_metrics['loss']:.4f}, Bal Acc: {val_metrics['balanced_accuracy']:.4f}, Recall: {val_metrics['recall']:.4f}, F2: {val_metrics['f2']:.4f}, AUC: {val_metrics['auc']:.4f}")

    if val_metrics['auc'] > best_val_auc:
        best_val_auc = val_metrics['auc']
        torch.save(model.state_dict(), ROOT / 'models/<model_name>_best.pth')
        print(f'  -> Saved best model (val AUC: {best_val_auc:.4f})')
```

This cell is structurally the same for every notebook. Only the `l1_lambda`/`l2_lambda` arguments and the model checkpoint filename change.

---

### Cells 10–11 — Training Curves

**Cell 10 (markdown):**
```
## Plot Train and Validation Curves
```

**Cell 11 (code)** — single line:

```python
plot_training_curves(train_history, val_history)
```

Plots Loss, Balanced Accuracy, Recall, and F2 for train and val side by side. No customisation needed.

---

### Cells 12–13 — Threshold Tuning

**Cell 12 (markdown):**
```
## Threshold Tuning (Best Val F2)
```

**Cell 13 (code)** — load the best checkpoint, then sweep thresholds on the **validation set** to maximise F2:

```python
model.load_state_dict(torch.load(str(ROOT / 'models/<model_name>_best.pth'), map_location=device))
best_threshold, best_f2 = find_best_threshold(model, val_loader, device)
```

The function prints `Best threshold: X.XX | Val F2: X.XXXX` automatically.

Why F2 for threshold tuning: it weights recall 2× over precision, matching the clinical priority of melanoma screening (false negatives are more costly than false positives), while still penalising degenerate all-positive predictions. See `notebooks/resnet50/README.md` for the full rationale.

---

### Cells 14–15 — Test Set Evaluation

**Cell 14 (markdown):**
```
## Test Set Evaluation
```

**Cell 15 (code)** — apply the tuned threshold to the held-out test set:

```python
evaluate_model(model, test_loader, device, threshold=best_threshold)
```

Prints: threshold, AUC-ROC, balanced accuracy, F2, full classification report, and confusion matrix.

**Do not change the threshold after seeing the test results.** The threshold was fixed on the validation set.

---

## Architecture Folder Summary markdown

Every architecture folder under `notebooks/` must contain a `architecture_summary.md` that:

1. States the benchmark being targeted (Test F2, AUC, Recall against prior experiments and ISIC 2018 competition results)
2. Lists all experiments in a results table with columns: `#`, `Notebook`, configuration summary, `Best Val F2`, `Test F2`, `Test Recall`, `Test Precision`, `AUC-ROC`
3. Has a **Key Findings** section summarising what each iteration revealed
4. Has a **Progression** section showing the Test F2 and AUC trend across iterations

See `notebooks/resnet50/README.md` as the reference example.

---

## Summary of What Changes vs What Stays Fixed

| Cells | Content | Changes between experiments? |
|---|---|---|
| 1 | Markdown intro | Yes — Objective, Architecture table, Hypothesis |
| 2–3 | Setup | No — same imports, seed, device |
| 4–5 | Data loading | No — same dataset, splits, augmentation |
| 6–7 | Model definition | Yes — model, unfrozen layers, LRs, lambdas, dropout |
| 8–9 | Training loop | Minimal — checkpoint filename, lambda args |
| 10–11 | Training curves | No |
| 12–13 | Threshold tuning | Minimal — checkpoint filename |
| 14–15 | Test evaluation | No |

The markdown header cells (2, 4, 6, 8, 10, 12, 14) are always just the `##` title — no body text.

# ResNet-50 Experiments

Binary melanoma classification (HAM10000) using pretrained ResNet-50, progressively exploring backbone unfreezing and regularisation strategies.

---

## Benchmark to Beat

Best result from `notebooks/penalty_experiments/`:

| Model | Config | Best Val F2 | Test F2 | Test Recall | Test Precision | AUC-ROC | Notebook |
|---|---|---|---|---|---|---|---|
| EfficientNet-B0 | Aug + L1 (30ep, CosineAnnealingLR) | 0.6689 | **0.6879** | 0.8070 | 0.4326 | — | `efficientnet_b0_aug_l1_l2` |

**Target: Test F2 > 0.6879**

### ISIC 2018 Competition & State-of-the-Art Reference

| Source | Model | Melanoma Recall | AUC | Balanced Acc | F1 |
|---|---|---|---|---|---|
| ISIC 2018 Winner | — | 88.5% | 0.971 | — | — |
| SOTA | Xception + Deep Attention | — | — | 95.9% | — |
| SOTA | ResNet50 + EfficientNetB0 + Patient Metadata Fusion | 93% | >0.973 | — | 93% |

---

## Model Selection & Evaluation Strategy

### Why save by best val AUC (not F2)

From iteration 07 onward, models are saved based on **best validation AUC-ROC** instead of best val F2. Rationale:

1. **AUC is threshold-independent** — it measures the model's raw discriminative ability, which is the fundamental bottleneck (our best: 0.884 vs ISIC SOTA: 0.973). F2, recall, precision are all downstream of AUC and depend on the decision threshold.
2. **AUC is the ceiling** — a higher AUC model will always have a better precision-recall tradeoff available at some threshold. Improving AUC lifts all metrics.
3. **ISIC benchmarks confirm this** — the competition winners with AUC 0.97 achieved high scores on every metric simultaneously. At that AUC level, the threshold barely matters.

### Why tune threshold by F2 (not recall with a precision floor)

After saving the best-AUC model, the threshold is tuned to maximise **F2 on the validation set**. Alternatives considered:

- **Maximise recall with a precision floor (e.g., >= 0.20)**: rejected because it tanks F1 and balanced accuracy. With our current AUC (~0.88), there is a fundamental precision-recall tradeoff — pushing recall to 0.85+ forces precision down to ~0.25, giving F1 ≈ 0.39. The ISIC benchmarks achieve F1 = 93% because their AUC is high enough to sustain both.
- **Maximise F1**: gives equal weight to precision and recall. Reasonable, but for melanoma detection recall should be weighted higher (false negatives are more costly than false positives).
- **Maximise F2**: weights recall 2x over precision, matching the clinical priority of melanoma screening while still penalising degenerate all-positive predictions.

### Pipeline

1. **Model selection**: save checkpoint with best val AUC (strongest feature extractor)
2. **Threshold tuning**: sweep thresholds on val set, pick the one that maximises F2
3. **Reporting**: report all metrics (AUC, Recall, Precision, F1, F2, Balanced Accuracy) for honest comparison against ISIC benchmarks

---

All experiments use:
- **Loss**: BCEWithLogitsLoss with pos_weight (~8.1) for class imbalance
- **Augmentation**: HFlip + VFlip + Rotation(30°) + ColorJitter + RandomAffine
- **Optimiser**: AdamW with layer-wise LR (FC head 10x higher than backbone)
- **Scheduler**: CosineAnnealingLR
- **Dropout**: 0.5 before FC head
- **Evaluation**: threshold tuning on val set, reported on test set

---

## Results Summary

| # | Notebook | Unfrozen | Trainable Params | Regularisation | Best Val F2 | Test F2 | Test Recall | Test Precision | AUC-ROC |
|---|---|---|---|---|---|---|---|---|---|
| 01 | `01.resnet50_frozen_baseline` | None (FC only) | 2K | Dropout | 0.5807 | 0.5469 | 0.8187 | 0.2349 | 0.8290 |
| 02 | `02.resnet50_partial_unfreeze` | `layer4` (full) | 15M | Dropout + WD=1e-4 | 0.6575 | 0.5852 | 0.6667 | 0.3931 | 0.8804 |
| 03 | `03.resnet50_unfreeze_layer4_2` | `layer4[2]` only | 4.7M | Dropout + WD=1e-4 | 0.6067 | 0.5986 | 0.7953 | 0.3009 | 0.8678 |
| 04 | `04.resnet50_unfreeze_layer4_wd1e3` | `layer4` (full) | 15M | Dropout + WD=1e-3 | 0.6595 | 0.5859 | 0.7018 | 0.3529 | 0.8728 |
| 05 | `05.resnet50_layer4_l1_l2` | `layer4` (full) | 15M | Dropout + WD=1e-3 + L1=1e-3 + L2=1e-3 | 0.6459 | **0.6184** | 0.7485 | 0.3647 | **0.8841** |
| 06 | `06.resnet50_layer4_2_l1_l2` | `layer4[2]` only | 4.7M | Dropout + WD=1e-3 + L1=1e-3 + L2=1e-3 | 0.6003 | 0.5730 | 0.7485 | 0.2956 | 0.8460 |
| 07 | `07.resnet50_layer4_l1` | `layer4` (full) | 15M | WD=1e-3 + L1=1e-3 (no Dropout) | 0.6536 | 0.6074 | 0.8070 | 0.3053 | 0.8861 |
| 08 | `08.resnet50_layer4_l1_metadata` | `layer4` (full) | 15M | Dropout=0.5 + WD=1e-3 + L1=1e-3 + Metadata | 0.6575 | 0.6048 | 0.7018 | 0.3896 | 0.8848 |

---

## Key Findings

- **Frozen baseline (01)**: AUC-ROC 0.829 confirms ResNet-50 ImageNet features are transferable to dermoscopy, but linear probing alone is insufficient.
- **Full layer4 unfreeze without regularisation (02, 04)**: Consistent train/val F2 gap of ~0.23 regardless of weight decay strength — WD alone cannot control overfitting at 15M params.
- **Narrow unfreeze (03, 06)**: Reduces overfitting gap to ~0.01–0.09 but at the cost of model capacity; both underperform their full layer4 equivalents on test F2.
- **L1+L2 penalties (05)**: Best test F2 overall (0.6184), train/val gap reduced to 0.06 — explicit penalties are significantly more effective than weight decay alone on this dataset size.
- **Narrow unfreeze + heavy penalties (06)**: Over-regularised — penalties hit 4.7M params harder than 15M, suppressing learning entirely.
- **L1 only, no Dropout (07)**: Best val AUC (0.9024) and highest test recall (0.8070), but precision drops to 0.3053 — removing L2 and Dropout shifts the precision-recall tradeoff toward recall. Test AUC (0.8861) marginally improves over 05 (0.8841), confirming L1 alone is a strong regulariser. However, test F2 (0.6074) is slightly below 05 (0.6184), suggesting Dropout still contributes to generalisation on this dataset.
- **L1 + Dropout + Patient Metadata (08)**: Adding metadata fusion with dropout restored precision (0.3896) but reduced recall (0.7018) compared to 07, resulting in slightly lower test F2 (0.6048) and AUC (0.8848). The metadata encoder did not provide a clear lift — the val AUC ceiling (0.8998) is lower than iteration 07 (0.9024), suggesting the metadata signal may be too weak relative to image features at this dataset scale, or the fusion architecture needs more capacity.

---

## Progression

```
Test F2:  0.547 → 0.585 → 0.599 → 0.586 → 0.618 → 0.573 → 0.607 → 0.605
AUC-ROC: 0.829 → 0.880 → 0.868 → 0.873 → 0.884 → 0.846 → 0.886 → 0.885
              L1+L2+Dropout (05) remains best on F2; L1 only (07) best on AUC and recall
              Metadata fusion (08) did not improve over image-only models
```

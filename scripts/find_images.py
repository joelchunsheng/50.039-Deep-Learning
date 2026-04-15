"""
find_images.py
--------------
Runs single-pass inference on the MobileNetV3-Large test set and saves 4 example
images (with titles) to the scripts/ folder:
  - mel_correct.png    : melanoma predicted as melanoma (TP)
  - mel_wrong.png      : melanoma predicted as non-melanoma (FN)
  - no_mel_correct.png : non-melanoma predicted as non-melanoma (TN)
  - no_mel_wrong.png   : non-melanoma predicted as melanoma (FP)
"""

import sys
from pathlib import Path

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

ROOT = next(p for p in [Path(__file__).resolve().parent.parent] + list(Path(__file__).resolve().parents)
            if (p / 'src').exists())
sys.path.insert(0, str(ROOT))

from src.data.dataset import HAM10000DatasetWithMetadata
from src.models.mobilenet import MobileNetV3LargeWithMetadata
from src.data.transform import get_eval_transforms

# ── Config ────────────────────────────────────────────────────────────────────
THRESHOLD        = 0.54
MODEL_PATH       = ROOT / 'models/01.mobilenet_v3_metadata_best.pth'
TEST_CSV         = ROOT / 'data_new/splits/test.csv'
TEST_IMAGE_DIR   = ROOT / 'data_new/images/test'
TEST_METADATA    = ROOT / 'data_new/raw/ISIC2018_Task3_Test_GroundTruth.csv'
OUT_DIR          = Path(__file__).resolve().parent

TARGETS = {
    'mel_correct':    {'label': 1, 'pred': 1},
    'mel_wrong':      {'label': 1, 'pred': 0},
    'no_mel_correct': {'label': 0, 'pred': 0},
    'no_mel_wrong':   {'label': 0, 'pred': 1},
}

TITLES = {
    'mel_correct':    'Melanoma — Predicted: Melanoma ✓',
    'mel_wrong':      'Melanoma — Predicted: Non-Melanoma ✗',
    'no_mel_correct': 'Non-Melanoma — Predicted: Non-Melanoma ✓',
    'no_mel_wrong':   'Non-Melanoma — Predicted: Melanoma ✗',
}

# ── Device ────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f'Using device: {device}')

# ── Model ─────────────────────────────────────────────────────────────────────
model = MobileNetV3LargeWithMetadata(metadata_dim=17, num_classes=1, dropout=0.5).to(device)
model.load_state_dict(torch.load(str(MODEL_PATH), map_location=device))
model.eval()
print('Model loaded.')

# ── Dataset (returns PIL image when transform=None) ───────────────────────────
dataset = HAM10000DatasetWithMetadata(
    csv_path=str(TEST_CSV),
    image_dir=str(TEST_IMAGE_DIR),
    metadata_path=str(TEST_METADATA),
    transform=None,
)

eval_transform = get_eval_transforms(image_size=224)

found = {}

print(f'Scanning {len(dataset)} test images...')
with torch.no_grad():
    for idx in range(len(dataset)):
        if len(found) == 4:
            break

        pil_img, metadata, label = dataset[idx]
        tensor = eval_transform(pil_img).unsqueeze(0).to(device)
        meta_t = metadata.unsqueeze(0).to(device)

        prob = torch.sigmoid(model(tensor, meta_t)).item()
        pred = int(prob >= THRESHOLD)

        for key, cond in TARGETS.items():
            if key not in found and label == cond['label'] and pred == cond['pred']:
                found[key] = (pil_img, prob, label, pred)
                print(f'  [{key}] idx={idx}  label={label}  pred={pred}  prob={prob:.4f}')
                break

print(f'\nFound {len(found)}/4 cases. Saving images...')

for key, (pil_img, prob, label, pred) in found.items():
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(pil_img)
    ax.axis('off')
    ax.set_title(TITLES[key], fontsize=10, pad=8)
    fig.text(0.5, 0.01, f'Prob: {prob:.4f}', ha='center', fontsize=8, color='gray')
    out_path = OUT_DIR / f'{key}.png'
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'  Saved: {out_path}')

print('Done.')

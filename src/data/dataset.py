from pathlib import Path

import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class HAM10000Dataset(Dataset):

    def __init__(self, csv_path, image_dir, transform=None):
        self.csv_path = Path(csv_path)
        self.image_dir = Path(image_dir)
        self.transform = transform

        self.data = pd.read_csv(self.csv_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        image_name = row["image_id"] + ".jpg"
        label = int(row["label"])

        image_path = self.image_dir / image_name
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class HAM10000DatasetWithMetadata(Dataset):
    """HAM10000Dataset that also returns a patient metadata tensor.

    Metadata features (dim=17):
        - age         : float, normalised by /100; NaN filled with train-set mean
        - sex         : float, male=0.0 / female=1.0 / unknown=0.5
        - localization: one-hot across 15 fixed categories (dim=15)

    Returns: (image, metadata, label)
    """

    LOCALIZATION_CATS = [
        'back', 'lower extremity', 'trunk', 'upper extremity', 'abdomen',
        'face', 'chest', 'foot', 'unknown', 'neck', 'scalp', 'hand', 'ear',
        'genital', 'acral',
    ]
    SEX_MAP = {'male': 0.0, 'female': 1.0, 'unknown': 0.5}
    AGE_NORM = 100.0
    AGE_FILL = 0.52  # ~mean age / 100 across HAM10000

    def __init__(self, csv_path, image_dir, metadata_path, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform

        splits = pd.read_csv(csv_path)
        meta = pd.read_csv(metadata_path)

        self.data = splits.merge(
            meta[['image_id', 'age', 'sex', 'localization']],
            on='image_id',
            how='left',
        )

        self.data['age'] = (
            self.data['age'].fillna(self.AGE_FILL * self.AGE_NORM) / self.AGE_NORM
        )
        self.data['sex_enc'] = self.data['sex'].map(self.SEX_MAP).fillna(0.5)
        for cat in self.LOCALIZATION_CATS:
            self.data[f'loc_{cat}'] = (self.data['localization'] == cat).astype(float)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        image_path = self.image_dir / (row['image_id'] + '.jpg')
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        meta_feats = (
            [row['age'], row['sex_enc']]
            + [row[f'loc_{cat}'] for cat in self.LOCALIZATION_CATS]
        )
        metadata = torch.tensor(meta_feats, dtype=torch.float32)

        return image, metadata, int(row['label'])
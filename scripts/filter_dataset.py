import pandas as pd
from pathlib import Path

splits = [
    "data_new/splits/train.csv",
    "data_new/splits/val.csv",
    "data_new/splits/test.csv",
]

target_id = "ISIC_0035068"

for csv_path in splits:
    path = Path(csv_path)
    
    # Load CSV
    df = pd.read_csv(path)
    
    # Remove the row with the target image_id
    df = df[df["image_id"] != target_id]
    
    # Save back to the same file (overwrite)
    df.to_csv(path, index=False)
    
    print(f"Updated: {path}")
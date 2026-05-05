from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd

data_dir = Path("/work/u10767535/datasets/utah_forge_16b78_32/dataset_fmi16_v2_color")
out_csv = Path("reports/fmi_valid_region_diagnostics.csv")
thresholds = [5, 10, 15, 20, 25, 30]

rows = []
paths = sorted(data_dir.glob("*.png"))

for i, path in enumerate(paths):
    img = np.array(Image.open(path).convert("RGB"))
    row = {
        "file": path.name,
        "height": img.shape[0],
        "width": img.shape[1],
        "mean_rgb": float(img.mean()),
        "std_rgb": float(img.std()),
    }

    for thr in thresholds:
        black = (img[:, :, 0] <= thr) & (img[:, :, 1] <= thr) & (img[:, :, 2] <= thr)
        col_black = black.mean(axis=0)
        row[f"black_frac_thr{thr}"] = float(black.mean())
        row[f"valid_frac_thr{thr}"] = float((~black).mean())
        row[f"black_columns_90_thr{thr}"] = float((col_black >= 0.90).mean())
        row[f"black_columns_50_thr{thr}"] = float((col_black >= 0.50).mean())

    rows.append(row)

    if (i + 1) % 1000 == 0:
        print("processed", i + 1)

df = pd.DataFrame(rows)
out_csv.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_csv, index=False)

print("images:", len(df))
print("output:", out_csv)
print(df[[c for c in df.columns if c.startswith("black_frac_thr")]].describe().round(4))
print(df[[c for c in df.columns if c.startswith("black_columns_90")]].describe().round(4))

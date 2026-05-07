from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import sys

sys.path.insert(0, str(Path("repo_original").resolve()))

from data.dataset import FMIInpaintDataset

out_dir = Path("reports/mask_preview_vertical_v1")
out_dir.mkdir(parents=True, exist_ok=True)

ds = FMIInpaintDataset(
    data_root="/work/u10767535/datasets/utah_forge_16b78_32/dataset_fmi16_v2_color",
    data_len=12,
    image_size=[256, 256],
    mask_config={
        "mask_mode": "fmi_vertical",
        "wide_width_range": [12, 40],
        "thin_width_range": [1, 6],
        "num_wide_range": [1, 3],
        "num_thin_range": [2, 8],
        "full_height_probability": 0.85,
        "partial_height_range": [96, 256],
        "horizontal_dilation": 1,
        "max_area_ratio": 0.55
    },
    black_threshold=-0.95
)

def to_uint8_chw(x):
    arr = x.detach().cpu().numpy()
    arr = np.transpose(arr, (1, 2, 0))
    arr = ((arr + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return arr

for i in range(len(ds)):
    x = ds[i]
    gt = to_uint8_chw(x["gt_image"])
    cond = to_uint8_chw(x["cond_image"])
    mask = x["mask"].detach().cpu().numpy()[0]
    mask_rgb = np.zeros_like(gt)
    mask_rgb[:, :, 0] = (mask * 255).astype(np.uint8)
    overlay = gt.copy()
    overlay[mask > 0.5] = (0.65 * overlay[mask > 0.5] + 0.35 * np.array([255, 0, 0])).astype(np.uint8)

    panels = [
        Image.fromarray(gt),
        Image.fromarray(cond),
        Image.fromarray(mask_rgb),
        Image.fromarray(overlay),
    ]

    w, h = panels[0].size
    canvas = Image.new("RGB", (4 * w, h + 24), "white")
    labels = ["GT", "COND", "MASK", "OVERLAY"]
    draw = ImageDraw.Draw(canvas)
    for j, panel in enumerate(panels):
        canvas.paste(panel, (j * w, 24))
        draw.text((j * w + 4, 4), labels[j], fill=(0, 0, 0))

    out = out_dir / f"mask_preview_vertical_v1_{i:03d}_{x['path']}"
    canvas.save(out)

print("saved:", out_dir)

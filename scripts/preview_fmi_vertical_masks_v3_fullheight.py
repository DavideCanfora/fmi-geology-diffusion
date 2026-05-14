from pathlib import Path
import sys
import shutil

import torch
from PIL import Image, ImageDraw

ROOT = Path("/work/u10767535/repos/fmi-geology-diffusion")
REPO = ROOT / "repo_original"
sys.path.insert(0, str(REPO))

from data.dataset import FMIInpaintDataset


DATA_ROOT = "/work/u10767535/exports/fmi_quality_audit_v3/accepted_files.txt"
OUT_DIR = Path("/work/u10767535/exports/mask_preview_vertical_v3_fullheight")
OUT_DIR.mkdir(parents=True, exist_ok=True)

for p in OUT_DIR.glob("*.png"):
    p.unlink()

dataset = FMIInpaintDataset(
    data_root=DATA_ROOT,
    data_len=24,
    image_size=[256, 256],
    black_threshold=-0.95,
    min_distance_from_real_gap=4,
    min_valid_fraction_for_artificial_mask=0.995,
    max_mask_resample_tries=50,
    mask_config={
        "mask_mode": "fmi_vertical",
        "wide_width_range": [12, 40],
        "thin_width_range": [1, 6],
        "num_wide_range": [1, 3],
        "num_thin_range": [2, 8],
        "full_height_probability": 1.0,
        "partial_height_range": [96, 256],
        "horizontal_dilation": 1,
        "max_area_ratio": 0.55,
    },
)

def to_uint8_img(t):
    t = t.detach().cpu().float()
    if t.ndim == 3:
        t = t.permute(1, 2, 0)
    arr = ((t.clamp(-1, 1) + 1) * 127.5).byte().numpy()
    return Image.fromarray(arr)

def gray_mask_to_rgb(mask):
    m = mask.detach().cpu().float()
    if m.ndim == 3:
        m = m[0]
    arr = (m.clamp(0, 1) * 255).byte().numpy()
    return Image.fromarray(arr, mode="L").convert("RGB")

for i in range(len(dataset)):
    item = dataset[i]

    gt = to_uint8_img(item["gt_image"])
    mask = gray_mask_to_rgb(item["mask"])
    mask_img = to_uint8_img(item["mask_image"])
    valid = gray_mask_to_rgb(item["valid_region"])

    w, h = gt.size
    canvas = Image.new("RGB", (4 * w, h + 32), "white")
    canvas.paste(gt, (0, 32))
    canvas.paste(valid, (w, 32))
    canvas.paste(mask, (2 * w, 32))
    canvas.paste(mask_img, (3 * w, 32))

    draw = ImageDraw.Draw(canvas)
    labels = ["GT", "valid_region", "mask", "mask_image"]
    for j, label in enumerate(labels):
        draw.text((j * w + 8, 8), label, fill=(0, 0, 0))

    out = OUT_DIR / f"mask_preview_vertical_v3_fullheight_{i:03d}_{item['path']}"
    canvas.save(out)

print(OUT_DIR)
print(len(list(OUT_DIR.glob("*.png"))))

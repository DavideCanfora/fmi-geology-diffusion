from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import sys

sys.path.insert(0, str(Path("repo_original").resolve()))

from data.dataset import FMIRealGapDataset

out_dir = Path("reports/real_gap_preview_v2_thin")
out_dir.mkdir(parents=True, exist_ok=True)

ds = FMIRealGapDataset(
    data_root="/work/u10767535/datasets/utah_forge_16b78_32/dataset_fmi16_v2_color",
    data_len=12,
    image_size=[256, 256],
    mask_config={"mask_mode": "hybrid"},
    black_threshold=-0.95,
    column_black_ratio=0.90,
    thin_column_black_ratio=0.35,
    horizontal_dilation=1
)

def tensor_to_uint8_img(x):
    x = (x.detach().cpu().float().clamp(-1, 1) + 1.0) / 2.0
    x = (x * 255).byte().numpy()
    x = np.transpose(x, (1, 2, 0))
    return Image.fromarray(x)

def mask_to_uint8(mask):
    m = mask.detach().cpu().float().squeeze(0).numpy()
    m = (m * 255).astype(np.uint8)
    return Image.fromarray(m).convert("RGB")

for i in range(len(ds)):
    sample = ds[i]

    original = tensor_to_uint8_img(sample["gt_image"])
    mask = mask_to_uint8(sample["mask"])
    overlay = tensor_to_uint8_img(sample["mask_image"])
    cond = tensor_to_uint8_img(sample["cond_image"])

    w, h = original.size
    canvas = Image.new("RGB", (4 * w, h + 30), "white")
    canvas.paste(original, (0, 30))
    canvas.paste(mask, (w, 30))
    canvas.paste(overlay, (2 * w, 30))
    canvas.paste(cond, (3 * w, 30))

    draw = ImageDraw.Draw(canvas)
    labels = ["original", "real_gap_mask_v2", "mask_overlay", "cond_image"]
    for j, label in enumerate(labels):
        draw.text((j * w + 5, 8), label, fill=(0, 0, 0))

    out_path = out_dir / f"real_gap_v2_{i:03d}_{sample['path']}"
    canvas.save(out_path)

print("saved:", out_dir)
for p in sorted(out_dir.glob("*.png")):
    print(p)

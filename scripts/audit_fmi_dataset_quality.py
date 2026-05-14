from pathlib import Path
import argparse
import csv
import shutil

import numpy as np
from PIL import Image


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".ppm"}


def max_true_run_fraction(mask_1d):
    mask_1d = np.asarray(mask_1d, dtype=bool)
    if mask_1d.size == 0 or not mask_1d.any():
        return 0.0

    idx = np.where(mask_1d)[0]
    best = 1
    start = idx[0]
    prev = idx[0]

    for value in idx[1:]:
        if value == prev + 1:
            prev = value
        else:
            best = max(best, prev - start + 1)
            start = value
            prev = value

    best = max(best, prev - start + 1)
    return float(best / mask_1d.size)


def compact_black_score(arr):
    gray = arr.mean(axis=2)
    black = gray < 0.035
    very_dark = gray < 0.070

    black_frac = black.mean()
    very_dark_frac = very_dark.mean()
    row_black = black.mean(axis=1).max()
    col_black = black.mean(axis=0).max()

    return float(
        1.0 * black_frac
        + 1.5 * very_dark_frac
        + 2.0 * row_black
        + 2.0 * col_black
    )


def noise_score(arr):
    gray = arr.mean(axis=2)

    dx = np.abs(gray[:, 1:] - gray[:, :-1])
    dy = np.abs(gray[1:, :] - gray[:-1, :])

    hf = 0.5 * (dx.mean() + dy.mean())

    col_profile = gray.mean(axis=0)
    row_profile = gray.mean(axis=1)

    structure = np.std(col_profile) + np.std(row_profile) + 1e-6

    return float(hf / structure)


def bad_pale_green_band_score(arr):
    r = arr[..., 0]
    g = arr[..., 1]
    b = arr[..., 2]
    gray = arr.mean(axis=2)

    pale_green = (
        (gray > 0.58)
        & (r > 0.62)
        & (g > 0.62)
        & (b > 0.20)
        & (b < 0.62)
        & ((r - b) > 0.18)
        & ((g - b) > 0.16)
        & (np.abs(r - g) < 0.22)
    )

    col_frac = pale_green.mean(axis=0)
    strong_cols = col_frac > 0.55
    medium_cols = col_frac > 0.35

    max_strong_run = max_true_run_fraction(strong_cols)
    max_medium_run = max_true_run_fraction(medium_cols)

    if medium_cols.any():
        selected = arr[:, medium_cols, :]
        vertical_std = float(selected.mean(axis=2).std(axis=0).mean())
        uniformity = max(0.0, 1.0 - 4.0 * vertical_std)
    else:
        uniformity = 0.0

    return float(
        1.0 * pale_green.mean()
        + 3.0 * col_frac.max()
        + 2.0 * medium_cols.mean()
        + 3.0 * max_medium_run
        + 2.0 * max_strong_run
        + 1.0 * uniformity
    )


def uncertain_vertical_block_score(arr):
    r = arr[..., 0]
    g = arr[..., 1]
    b = arr[..., 2]
    gray = arr.mean(axis=2)

    pale = (
        (gray > 0.52)
        & (r > 0.45)
        & (g > 0.45)
        & (b < 0.70)
        & (np.abs(r - g) < 0.32)
    )

    col_frac = pale.mean(axis=0)
    row_frac = pale.mean(axis=1)

    suspicious_cols = col_frac > 0.35
    suspicious_rows = row_frac > 0.35

    max_col_run = max_true_run_fraction(suspicious_cols)
    max_row_run = max_true_run_fraction(suspicious_rows)

    return float(
        1.0 * pale.mean()
        + 2.0 * suspicious_cols.mean()
        + 1.0 * suspicious_rows.mean()
        + 2.0 * max_col_run
        + 1.0 * max_row_run
    )


def quality_scores(arr):
    metrics = {
        "compact_black_score": compact_black_score(arr),
        "noise_score": noise_score(arr),
        "bad_pale_green_band_score": bad_pale_green_band_score(arr),
        "uncertain_vertical_block_score": uncertain_vertical_block_score(arr),
    }

    metrics["total_score"] = (
        1.0 * metrics["bad_pale_green_band_score"]
        + 1.0 * metrics["compact_black_score"]
        + 0.25 * metrics["noise_score"]
    )

    return metrics


def classify_quality(metrics, greenband_threshold=2.00, black_threshold=6.10, noise_threshold=0.50):
    is_bad_greenband = metrics["bad_pale_green_band_score"] > greenband_threshold
    is_bad_black = metrics["compact_black_score"] > black_threshold
    is_bad_noise = metrics["noise_score"] > noise_threshold

    is_any_bad = is_bad_greenband or is_bad_black or is_bad_noise

    return {
        "bad_pale_green_bands": is_bad_greenband,
        "bad_black": is_bad_black,
        "bad_noise": is_bad_noise,
        "bad_any": is_any_bad,
        "accepted": not is_any_bad,
        "eligible_for_uncertain_vertical_blocks": not is_any_bad,
    }


def image_paths(root):
    root = Path(root)

    if root.is_file():
        for line in root.read_text().splitlines():
            line = line.strip()
            if line:
                yield Path(line)
        return

    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in IMG_EXTENSIONS:
            yield p


def load_rgb01(path):
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0


def copy_sample(path, out_dir, rank):
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, out_dir / f"{rank:04d}_{path.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sample-per-category", type=int, default=16)
    ap.add_argument("--greenband-threshold", type=float, default=2.00)
    ap.add_argument("--black-threshold", type=float, default=6.10)
    ap.add_argument("--noise-threshold", type=float, default=0.50)
    ap.add_argument("--uncertain-threshold", type=float, default=1.50)
    ap.add_argument("--uncertain-top-k", type=int, default=10)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    counters = {
        "accepted_sample": 0,
        "bad_black": 0,
        "bad_noise": 0,
        "bad_pale_green_bands": 0,
        "uncertain_vertical_blocks": 0,
        "uncertain_top_removed": 0,
    }

    paths = list(image_paths(args.data_root))
    print(f"Found images: {len(paths)}")

    for i, path in enumerate(paths):
        arr = load_rgb01(path)
        metrics = quality_scores(arr)
        labels = classify_quality(
            metrics,
            greenband_threshold=args.greenband_threshold,
            black_threshold=args.black_threshold,
            noise_threshold=args.noise_threshold,
        )

        uncertain_vertical_blocks = (
            labels["eligible_for_uncertain_vertical_blocks"]
            and metrics["uncertain_vertical_block_score"] > args.uncertain_threshold
        )

        row = {
            "path": str(path),
            **{k: f"{v:.8f}" for k, v in metrics.items()},
            **{k: int(v) for k, v in labels.items()},
            "uncertain_vertical_blocks": int(uncertain_vertical_blocks),
            "uncertain_top_removed": 0,
        }
        rows.append(row)

        if (i + 1) % 500 == 0:
            print(f"Processed {i + 1}/{len(paths)}")

    base_bad_paths = {r["path"] for r in rows if int(r["bad_any"]) == 1}

    uncertain_candidates = sorted(
        [r for r in rows if r["path"] not in base_bad_paths],
        key=lambda r: float(r["uncertain_vertical_block_score"]),
        reverse=True,
    )
    uncertain_top_paths = {
        r["path"] for r in uncertain_candidates[:max(0, args.uncertain_top_k)]
    }

    accepted = []
    rejected = []

    for row in rows:
        path = Path(row["path"])

        is_uncertain_top = row["path"] in uncertain_top_paths
        row["uncertain_top_removed"] = int(is_uncertain_top)

        if is_uncertain_top:
            row["bad_any"] = 1
            row["accepted"] = 0

        if int(row["accepted"]) == 1:
            accepted.append(row["path"])
        else:
            rejected.append(row["path"])

        if int(row["bad_black"]) and counters["bad_black"] < args.sample_per_category:
            counters["bad_black"] += 1
            copy_sample(path, out_dir / "bad_black", counters["bad_black"])

        if int(row["bad_noise"]) and counters["bad_noise"] < args.sample_per_category:
            counters["bad_noise"] += 1
            copy_sample(path, out_dir / "bad_noise", counters["bad_noise"])

        if int(row["bad_pale_green_bands"]) and counters["bad_pale_green_bands"] < args.sample_per_category:
            counters["bad_pale_green_bands"] += 1
            copy_sample(path, out_dir / "bad_pale_green_bands", counters["bad_pale_green_bands"])

        if int(row["uncertain_vertical_blocks"]) and counters["uncertain_vertical_blocks"] < args.sample_per_category:
            counters["uncertain_vertical_blocks"] += 1
            copy_sample(path, out_dir / "uncertain_vertical_blocks", counters["uncertain_vertical_blocks"])

        if is_uncertain_top and counters["uncertain_top_removed"] < args.sample_per_category:
            counters["uncertain_top_removed"] += 1
            copy_sample(path, out_dir / "uncertain_top_removed", counters["uncertain_top_removed"])

        if int(row["accepted"]) and counters["accepted_sample"] < args.sample_per_category:
            counters["accepted_sample"] += 1
            copy_sample(path, out_dir / "accepted_sample", counters["accepted_sample"])

    fieldnames = list(rows[0].keys()) if rows else []
    with (out_dir / "fmi_quality_scores.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    (out_dir / "accepted_files.txt").write_text("\n".join(accepted) + "\n")
    (out_dir / "rejected_files.txt").write_text("\n".join(rejected) + "\n")

    print("Summary")
    print(f"total: {len(paths)}")
    print(f"accepted: {len(accepted)}")
    print(f"rejected: {len(rejected)}")
    print(f"greenband_threshold: {args.greenband_threshold}")
    print(f"black_threshold: {args.black_threshold}")
    print(f"noise_threshold: {args.noise_threshold}")
    print(f"uncertain_top_k: {args.uncertain_top_k}")
    print(f"discard_greenband_count: {sum(int(r['bad_pale_green_bands']) for r in rows)}")
    print(f"discard_black_count: {sum(int(r['bad_black']) for r in rows)}")
    print(f"discard_noise_count: {sum(int(r['bad_noise']) for r in rows)}")
    print(f"discard_uncertain_top_count: {sum(int(r['uncertain_top_removed']) for r in rows)}")
    print(f"discard_any_count: {sum(int(r['bad_any']) for r in rows)}")
    print(f"kept_count: {sum(int(r['accepted']) for r in rows)}")
    for key, value in counters.items():
        print(f"sampled_{key}: {value}")
    print(out_dir)


if __name__ == "__main__":
    main()

# FMI diffusion project status

Last synced commit:

dba7003 Use audited clean FMI dataset for vertical diffusion training

## Repository state

Canonical Mac code repository:

~/Desktop/Tesi/codici/tesi_diffusion/repo_original

Server repository:

/work/u10767535/repos/fmi-geology-diffusion/repo_original

GitHub main is aligned with both Mac and server.

The server still has an untracked references/ folder. It is intentionally not committed.

## Local Mac workspace organization

Top-level Mac workspace:

~/Desktop/Tesi/codici/tesi_diffusion

Rules:

- repo_original/: code only, Git-controlled.
- outputs/: local organized results.
- outputs/audits/: dataset quality audits.
- outputs/runs/: model runs and result folders.
- outputs/debug/: smoke tests and intermediate debug.
- outputs/mask_previews/: artificial/real mask visual checks.
- outputs/dataset_previews/: FMI dataset previews.
- patches/: temporary transfer bundles or patches.
- notes/: project state and local summaries.

Ignored local folders:

outputs/
patches/
downloaded_server_results/

## Dataset audit

Original dataset:

/work/u10767535/datasets/utah_forge_16b78_32/dataset_fmi16_v2_color

Final accepted audit:

/work/u10767535/exports/fmi_quality_audit_v3

Final clean training dataset:

/work/u10767535/datasets/utah_forge_16b78_32/dataset_fmi16_v2_color_clean_v3

Clean dataset properties:

storage_mode: symlink
kept_images: 6138
source audit: fmi_quality_audit_v3

Audit thresholds:

bad_pale_green_band_score > 2.0
compact_black_score > 6.1
noise_score > 0.5
uncertain_top_k = 10

## Training config roots

These training configs now use the clean dataset:

repo_original/config/inpainting_fmi16_vertical_train_v2.json
repo_original/config/inpainting_fmi16_vertical_train_v3_losses_ft.json

Training split uses:

data_root = /work/u10767535/datasets/utah_forge_16b78_32/dataset_fmi16_v2_color_clean_v3
data_len = -1

Validation/test inside those training configs uses:

data_len = 64

Real-gap inference configs remain on the original dataset, because they must detect real missing black bands in original FMI images.

## Mask logic

Current diffusion mask logic:

- FMI artificial masks are vertical.
- For fmi_vertical, masks are kept as full vertical structures.
- The mask is not pixelwise-intersected with valid_region in the accepted path.
- Artificial masks avoid real missing columns using min_distance_from_real_gap = 4.
- Fallback exists if no clean candidate is found.

Purpose:

- avoid dirty/speckled artificial masks;
- avoid training artificial holes on naturally missing FMI gaps;
- keep the mask geometry closer to real vertical acquisition gaps.

## Losses currently implemented

Current training objective in models/network.py:

noise_loss + 0.5 * masked_l1_loss + 0.1 * masked_gradient_l1_loss

Losses:

- base Palette masked noise-prediction MSE;
- masked L1 reconstruction loss on predicted clean image;
- masked gradient L1 loss.

## Sampling / inference additions

Current code includes:

- standard vertical real-gap inference;
- RePaint-style sampling;
- clean-image data consistency during inpainting sampling.

## Current decision state

Do not launch new training blindly.

Next serious step:

1. wait for metric outputs from the parallel evaluation work;
2. compare old v2 training, v3 loss fine-tuning, data-consistency inference, and RePaint inference;
3. decide whether the next run should be full retrain from scratch, fine-tuning from v2 checkpoint, loss-weight adjustment, or sampling/inference change only.

Current preferred next experimental candidate, if metrics confirm that the pipeline is ready:

train_inpainting_fmi16_vertical_train_v2 on clean_v3 with structured full-height masks

Reason:

The previous v3 loss fine-tuning was trained before the final clean dataset/root correction and before final mask cleanup. A clean retrain is more interpretable than stacking more changes on top of an old checkpoint.

# Current status — Palette FMI diffusion

Repository:
- GitHub: DavideCanfora/fmi-geology-diffusion
- Latest commit: e9603f0 Document Palette FMI diffusion pipeline

Local project:
- Root: ~/Desktop/Tesi/codici/tesi_diffusion
- Palette source: repo_original/
- Debug config: repo_original/config/inpainting_fmi16_debug.json
- Stable config copy: configs/inpainting_fmi16_debug_working.json
- Local debug data: data/fmi16_debug_color/

Dataset:
- Main FMI dataset on server:
  - /work/u10767535/datasets/utah_forge_16b78_32/dataset_fmi16_v2_gray
  - /work/u10767535/datasets/utah_forge_16b78_32/dataset_fmi16_v2_color
- Number of patches: 6300 gray + 6300 color
- Patch size before Palette resize: 512 x 360
- Metadata:
  - /work/u10767535/datasets/utah_forge_16b78_32/dataset_fmi16_v2_metadata.csv

Implemented:
- FMIInpaintDataset in repo_original/data/dataset.py
- It detects non-black valid FMI regions.
- It restricts artificial Palette masks to valid FMI pixels.
- It does not yet implement FMI-specific vertical/gap masks.

Current debug pipeline:
- Uses Palette guided_diffusion UNet.
- Input channels: 6 = cond_image RGB + noisy/mixed RGB.
- Output channels: 3 = predicted RGB noise.
- Loss: MSE between true noise and predicted noise, restricted to mask.
- Metric: full-image MAE, only preliminary.
- Debug timesteps: 10.
- Debug image size: 256 x 256.
- Debug runs on Mac CPU.

Verified:
- Dataset loads correctly.
- DataLoader returns gt_image, cond_image, mask, valid_region.
- Training loop runs.
- Validation/restoration runs.
- Checkpoints and images are saved.
- GitHub is updated.

Important future constraints:
- Server may use AMD GPUs, likely requiring PyTorch ROCm.
- Before server training, verify:
  - torch.__version__
  - torch.cuda.is_available()
  - torch.version.cuda
  - torch.version.hip
  - torch.cuda.device_count()
  - torch.cuda.get_device_name(0)

Do not improvise scientific modifications.
Before changing:
- mask generation
- loss
- metric
- training strategy
- inference for real FMI gaps
- architecture

first inspect relevant papers and code, especially borehole completion papers such as:
- Filling borehole image gaps with a partial convolution neural network
- Completing Any Borehole Images / LogMAT
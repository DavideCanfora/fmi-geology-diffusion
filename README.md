# Tesi diffusion

Questo progetto contiene il ramo diffusion della tesi.

Paper di riferimento:
Palette: Image-to-Image Diffusion Models
https://arxiv.org/abs/2111.05826

Repository di partenza:
https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models

Obiettivo:
adattare Palette al problema di FMI borehole image inpainting.

Dataset principale:
Utah FORGE 16B(78)-32 FMI dataset v2

Percorso server:
 /work/u10767535/datasets/utah_forge_16b78_32/dataset_fmi16_v2_gray
 /work/u10767535/datasets/utah_forge_16b78_32/dataset_fmi16_v2_color

Strategia:
- partire da Palette come conditional diffusion model;
- addestrare da zero su immagini FMI;
- generare maschere artificiali FMI-like;
- usare input mascherato come condizione e immagine completa come target;
- iniziare da grayscale, poi valutare color afmhot.

## Stato Palette-FMI debug locale

È stato clonato il repository:
Janspiry/Palette-Image-to-Image-Diffusion-Models

È stata creata una configurazione debug:
repo_original/config/inpainting_fmi16_debug.json

È stato creato un mini dataset locale:
data/fmi16_debug_color/

È stata aggiunta una classe custom:
repo_original/data/dataset.py -> FMIInpaintDataset

La classe FMIInpaintDataset:
- carica immagini FMI colorate;
- identifica le bande nere già mancanti;
- genera maschere artificiali solo sulle regioni valide;
- evita di usare i gap reali come target artificiale;
- restituisce gt_image, cond_image, mask_image, mask, valid_region.

Sono state applicate patch per esecuzione locale CPU:
- core/util.py
- models/network.py

Sono state applicate patch compatibilità Pandas:
- core/logger.py

Run debug completata:
- training completato;
- validation completata;
- checkpoint salvati;
- immagini salvate in repo_original/experiments/debug_inpainting_fmi16_debug_*/results/.

Interpretazione:
questo non è ancora un esperimento scientifico, ma dimostra che Palette può essere adattato a FMI inpainting.

## Palette FMI debug configuration

The file `repo_original/config/inpainting_fmi16_debug.json` is a local smoke-test configuration, not a scientific training configuration.

It verifies that:
- `FMIInpaintDataset` loads FMI image patches correctly;
- artificial masks are restricted to valid FMI pixels;
- Palette runs locally on CPU with a very small UNet;
- training, validation, checkpointing and image export work end-to-end.

Serious training will require:
- full FMI dataset paths on the HPC server;
- GPU-enabled configuration;
- larger network;
- more diffusion timesteps;
- FMI-specific masks, losses and metrics selected from literature.

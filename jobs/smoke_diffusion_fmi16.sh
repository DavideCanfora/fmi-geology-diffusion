#!/bin/bash
#PBS -N smoke_diffusion_fmi16
#PBS -q gpu
#PBS -l select=1:ncpus=4:ngpus=1:mem=16gb
#PBS -l walltime=00:30:00
#PBS -o /work/u10767535/logs/fmi_geology_diffusion/smoke_diffusion_fmi16.out
#PBS -e /work/u10767535/logs/fmi_geology_diffusion/smoke_diffusion_fmi16.err

set -euo pipefail

echo "JOB_ID=$PBS_JOBID"
echo "HOSTNAME=$(hostname)"
echo "DATE=$(date)"
echo "PBS_NODEFILE=$PBS_NODEFILE"
cat "$PBS_NODEFILE"

cd /work/u10767535/repos/fmi-geology-diffusion/repo_original

PYTHON=/work/u10767535/venvs/virtual_core_torch/bin/python

echo "Python:"
$PYTHON --version

echo "Torch/GPU check:"
$PYTHON - << 'PYEOF'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
print("hip version:", torch.version.hip)
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
PYEOF

echo "Dataset count:"
find /work/u10767535/datasets/utah_forge_16b78_32/dataset_fmi16_v2_color -name "*.png" | wc -l

echo "Start Palette FMI smoke training"
$PYTHON run.py -p train -c config/inpainting_fmi16_server_smoke.json -d

echo "Latest experiment:"
ls -td experiments/debug_inpainting_fmi16_server_smoke_* | head -1

EXP=$(ls -td experiments/debug_inpainting_fmi16_server_smoke_* | head -1)
tail -40 "$EXP/train.log"
find "$EXP/results" -type f | head -40

echo "DONE $(date)"

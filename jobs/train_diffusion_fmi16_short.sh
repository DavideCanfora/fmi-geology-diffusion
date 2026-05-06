#!/bin/bash
#PBS -N train_diff_short
#PBS -q gpu
#PBS -l select=1:ncpus=8:ngpus=1:mem=32gb
#PBS -l walltime=02:00:00
#PBS -o /work/u10767535/logs/fmi_geology_diffusion/train_diffusion_fmi16_short.out
#PBS -e /work/u10767535/logs/fmi_geology_diffusion/train_diffusion_fmi16_short.err

set -euo pipefail

mkdir -p /work/u10767535/logs/fmi_geology_diffusion
exec > /work/u10767535/logs/fmi_geology_diffusion/train_diffusion_fmi16_short.manual.out 2> /work/u10767535/logs/fmi_geology_diffusion/train_diffusion_fmi16_short.manual.err

echo "JOB_ID=$PBS_JOBID"
echo "HOSTNAME=$(hostname)"
echo "DATE=$(date)"
echo "PBS_NODEFILE=$PBS_NODEFILE"
cat "$PBS_NODEFILE"

cd /work/u10767535/repos/fmi-geology-diffusion/repo_original

PYTHON=/work/u10767535/venvs/virtual_core_torch/bin/python

echo "Torch/GPU check:"
$PYTHON - << 'PYEOF'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
PYEOF

echo "Start Palette FMI short training"
$PYTHON run.py -p train -c config/inpainting_fmi16_server_short_train.json

echo "Latest experiment:"
ls -td experiments/train_inpainting_fmi16_server_short_train_* | head -1

EXP=$(ls -td experiments/train_inpainting_fmi16_server_short_train_* | head -1)
tail -120 "$EXP/train.log"
find "$EXP/checkpoint" -type f | sort | tail -20
find "$EXP/results" -type f | sort | head -40

echo "DONE $(date)"

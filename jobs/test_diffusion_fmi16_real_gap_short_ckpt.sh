#!/bin/bash
#PBS -N test_real_short
#PBS -q gpu
#PBS -l select=1:ncpus=4:ngpus=1:mem=16gb
#PBS -l walltime=00:30:00
#PBS -o /work/u10767535/logs/fmi_geology_diffusion/test_real_gap_short_ckpt.out
#PBS -e /work/u10767535/logs/fmi_geology_diffusion/test_real_gap_short_ckpt.err

set -euo pipefail

mkdir -p /work/u10767535/logs/fmi_geology_diffusion
exec > /work/u10767535/logs/fmi_geology_diffusion/test_real_gap_short_ckpt.manual.out 2> /work/u10767535/logs/fmi_geology_diffusion/test_real_gap_short_ckpt.manual.err

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

echo "Start real-gap Palette test with short checkpoint"
$PYTHON run.py -p test -c config/inpainting_fmi16_real_gap_test_short_ckpt.json

echo "Latest experiment:"
ls -td experiments/test_inpainting_fmi16_real_gap_test_short_ckpt_* | head -1

EXP=$(ls -td experiments/test_inpainting_fmi16_real_gap_test_short_ckpt_* | head -1)
tail -80 "$EXP/train.log" 2>/dev/null || true
find "$EXP/results" -type f | sort | head -80

echo "DONE $(date)"

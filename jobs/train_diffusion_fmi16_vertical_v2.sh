#!/bin/bash
#PBS -N train_fmi_vert_v2
#PBS -q gpu
#PBS -l select=1:ncpus=8:ngpus=1:mem=32gb
#PBS -l walltime=08:00:00
#PBS -o /work/u10767535/logs/fmi_geology_diffusion/train_fmi_vertical_v2.out
#PBS -e /work/u10767535/logs/fmi_geology_diffusion/train_fmi_vertical_v2.err

set -euo pipefail

mkdir -p /work/u10767535/logs/fmi_geology_diffusion
exec > /work/u10767535/logs/fmi_geology_diffusion/train_fmi_vertical_v2.manual.out 2> /work/u10767535/logs/fmi_geology_diffusion/train_fmi_vertical_v2.manual.err

echo "JOB_ID=$PBS_JOBID"
echo "HOSTNAME=$(hostname)"
echo "DATE=$(date)"
echo "PBS_NODEFILE=$PBS_NODEFILE"
cat "$PBS_NODEFILE"

cd /work/u10767535/repos/fmi-geology-diffusion/repo_original

echo "Git status:"
git status --short

echo "Torch/GPU check:"
/work/u10767535/venvs/virtual_core_torch/bin/python - << 'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
print("device count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY

echo "Start FMI vertical diffusion training v2"
export CUDA_VISIBLE_DEVICES=0

/work/u10767535/venvs/virtual_core_torch/bin/python run.py \
  -c config/inpainting_fmi16_vertical_train_v2.json \
  -p train

echo "Latest experiment:"
EXP=$(ls -td experiments/train_inpainting_fmi16_vertical_train_v2_* 2>/dev/null | head -1)
echo "$EXP"

if [ -n "$EXP" ]; then
  tail -120 "$EXP/train.log"
  find "$EXP/checkpoint" -type f | sort
  find "$EXP/results" -type f | sort | head -80
fi

echo "DONE $(date)"

#!/bin/bash
#PBS -N test_real_v2_rp
#PBS -q gpu
#PBS -l select=1:ncpus=4:ngpus=1:mem=16gb
#PBS -l walltime=00:45:00
#PBS -o /work/u10767535/logs/fmi_geology_diffusion/test_real_gap_vertical_v2_repaint.out
#PBS -e /work/u10767535/logs/fmi_geology_diffusion/test_real_gap_vertical_v2_repaint.err

set -euo pipefail

mkdir -p /work/u10767535/logs/fmi_geology_diffusion
exec > /work/u10767535/logs/fmi_geology_diffusion/test_real_gap_vertical_v2_repaint.manual.out 2> /work/u10767535/logs/fmi_geology_diffusion/test_real_gap_vertical_v2_repaint.manual.err

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

echo "Start real-gap vertical_v2 RePaint test"
export CUDA_VISIBLE_DEVICES=0

/work/u10767535/venvs/virtual_core_torch/bin/python run.py \
  -c config/inpainting_fmi16_real_gap_test_vertical_v2_repaint.json \
  -p test

echo "Latest experiment:"
EXP=$(ls -td experiments/test_inpainting_fmi16_real_gap_test_vertical_v2_repaint_* 2>/dev/null | head -1)
echo "$EXP"

if [ -n "$EXP" ]; then
  find "$EXP/results" -type f | sort | head -80
  tail -100 "$EXP/test.log" 2>/dev/null || true
fi

echo "DONE $(date)"

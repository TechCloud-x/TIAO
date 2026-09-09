#!/bin/bash
#SBATCH --job-name=TIAO-Qwen2.5-7B-CD-M80
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --exclude=paraai-n32-h-01-agent-92
#SBATCH --cpus-per-task=128
#SBATCH --qos=gpugpu
#SBATCH --output=slurm-mask80-%j.out

# Fixed CNN/DailyMail source-mask ablation: 80% masking for four epochs.
set -Eeo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec /bin/bash "${SCRIPT_DIR}/train_tiao_qwen2_5_7b_cnn_dailymail_mask_ablation_common.sh" \
  "0.8" \
  "mask80"

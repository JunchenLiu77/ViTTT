#!/bin/bash
#SBATCH --job-name=vittt_b_4gpu_bs1024_120epoch
#SBATCH --account=aip-fsanja
#SBATCH --output=output/vittt_b_4gpu_bs1024_120epoch/%x_%j.out
#SBATCH --error=output/vittt_b_4gpu_bs1024_120epoch/%x_%j.err
#SBATCH --time=00-16:00:00
#SBATCH --nodes=1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=h100:4
#SBATCH --ntasks-per-node=1
#SBATCH --exclude=kn122

# Mode: Training

echo "=============================================="
echo "ViT^3 Training"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: 4x h100"
echo "Start time: $(date)"
echo "Config: cfgs/vittt_b.yaml"
echo "=============================================="

# Load modules
module load python/3.12
module load cuda/12.6

# Environment variables
export OMP_NUM_THREADS=8
export IBV_FORK_SAFE=1

# Distributed training settings
export MASTER_ADDR=localhost
export MASTER_PORT=$(python3 - <<'PY'
import socket
s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(("", 0))
print(s.getsockname()[1])
s.close()
PY
)

# Suppress libibverbs warnings
exec 3>&2
exec 2> >(grep -v "libibverbs: Warning" >&3)

echo
echo "Starting training..."
echo

# Run the job
srun uv run python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=$MASTER_PORT \
    main.py \
    --cfg cfgs/vittt_b.yaml \
    --data-path ~/projects/aip-fsanja/shared/datasets/imagenet/ \
    --output output/vittt_b_4gpu_bs1024_120epoch \
    --batch-size 256 \
    --epochs 120 \
    --warmup-epochs 8 \
    --wandb-run-name vittt_b_4gpu_bs1024_120epoch \
    --amp

# Restore stderr
exec 2>&3
exec 3>&-

echo
echo "=============================================="
echo "Training completed at: $(date)"
echo "=============================================="

exit 0
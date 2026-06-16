#!/bin/bash
#SBATCH --job-name=inferssl_partial_charge
#SBATCH --account=choiseprojec
#SBATCH --nodes=4
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=4:00:00
#SBATCH --qos=high

# 2-feature comparison: SSL embedding extraction
# Kestrel: 4 nodes x 4 GPUs = 16 tasks (srun, one task per GPU)

export MPICH_GPU_SUPPORT_ENABLED=1

INPUT_DIR="./filtered_csv"
OUTPUT_DIR="./filtered_csv_embeddings_partial_charge"
WORLD_SIZE=16

srun --ntasks=${WORLD_SIZE} --cpu-bind=cores --gpus-per-task=1 bash -c '
  python inference_ssl.py \
    --filtered_csv "'"${INPUT_DIR}"'" \
    --output_dir "'"${OUTPUT_DIR}"'" \
    --rank ${SLURM_PROCID} \
    --world_size '"${WORLD_SIZE}"'
'

echo "Done."

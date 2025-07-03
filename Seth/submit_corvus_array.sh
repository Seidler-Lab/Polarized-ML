#!/bin/bash
#SBATCH --job-name=Mn_corvus_array
#SBATCH --array=0-1498
#SBATCH --account=stf
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=0:40:00
#SBATCH --mem=8GB
#SBATCH --output=logs/output_%A_%a.log
#SBATCH --error=logs/error_%A_%a.log

# source ~/.bashrc
# conda activate Corvus2

module load ompi
conda activate Corvus

# Get input file based on array index
INPUT_FILE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" corvus_good_file_list.txt)
INPUT_DIR=$(dirname "$INPUT_FILE")
INPUT_NAME=$(basename "$INPUT_FILE")

cd "$INPUT_DIR"
echo "Running: run-corvus -i $INPUT_NAME in $INPUT_DIR"
run-corvus -i "$INPUT_NAME"

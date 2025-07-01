#!/bin/bash
#PBS -N Co_corvus_array
#PBS -J 0-1451
#PBS -l select=1:ncpus=1:mem=2gb
#PBS -l walltime=02:00:00
#PBS -o logs/output_$PBS_ARRAY_INDEX.log
#PBS -e logs/error_$PBS_ARRAY_INDEX.log
#PBS -q workq
#PBS -V

# module purge

# for var in $(compgen -v | grep '^I_MPI_'); do unset "$var"; done
# unset LOADEDMODULES
# unset _LMFILES_

# export PATH="/home/sethshj/.conda/envs/Corvus2/bin:/opt/anaconda3/condabin:$HOME/.local/bin:$HOME/bin:$HOME/feff10/bin:/usr/bin:/bin:/usr/sbin:/usr/local/sbin"
# export LD_LIBRARY_PATH="/home/sethshj/.conda/envs/Corvus2/lib"

# eval "$(/opt/anaconda3/bin/conda shell.bash hook)"
source ~/.bashrc
conda activate Corvus2

# export PATH="/home/sethshj/.conda/envs/Corvus2/bin:/opt/anaconda3/condabin:$HOME/.local/bin:$HOME/bin:$HOME/feff10/bin:/usr/bin:/bin:/usr/sbin:/usr/local/sbin"
# export LD_LIBRARY_PATH="/home/sethshj/.conda/envs/Corvus2/lib"

# # === Print diagnostic info ===
# echo "===== FINAL ENVIRONMENT ====="
# echo "PATH = $PATH"
# echo "LD_LIBRARY_PATH = $LD_LIBRARY_PATH"
# which python
# which run-corvus
# env | grep -Ei 'mpi|corvus|feff'

cd $PBS_O_WORKDIR

#printenv | sort > env_batch.txt

# Get input file based on array index
INPUT_FILE=$(sed -n "$((PBS_ARRAY_INDEX + 1))p" corvus_good_file_list.txt)
INPUT_DIR=$(dirname "$INPUT_FILE")
INPUT_NAME=$(basename "$INPUT_FILE")

cd "$INPUT_DIR"
echo "Running: run-corvus -i $INPUT_NAME in $INPUT_DIR"
run-corvus -i "$INPUT_NAME"

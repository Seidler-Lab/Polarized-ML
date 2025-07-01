import subprocess
import pathlib as Path
import os
import time
import re


def create_list_of_corvus_input_files(calc_directory: Path) -> list:
    """
    Globs the parent directory for every corvus.in file that was created, and creates a list of all of them.
    Raises an error if none are found

    Args:
    - calc_directory: The root directory containing subdirectories with `.in` files.
    """

    # Recursively find all `.in` files in subdirectories
    corvus_in_files_list = sorted(calc_directory.rglob("*.in"))  # Use rglob for recursion

    if not corvus_in_files_list:
        ValueError("No .in files found in the directory or its subdirectories.")
        return
    
    print(f"This is the length of all the calculations that are coming from the creation of input files list {len(corvus_in_files_list)}")

    return corvus_in_files_list

def save_input_file_list(corvus_in_file_list, list_path="corvus_input_paths.txt"):
    with open(list_path, "w") as f:
        for path in corvus_in_file_list:
            f.write(str(path) + "\n")
        f.flush()
        os.fsync(f.fileno())
    return list_path

def write_corvus_array_script(job_list_file, script_path="submit_corvus_array.sh"):
    """
    Generate a PBS job array script that activates conda env and runs run-corvus -i in each input file's directory.

    Args:
        corvus_in_file_list (list of str): List of full input file paths.
        input_file_length (integer): Length of the corvus_in_file_list
        script_path (str): Output PBS script path.

    Returns:
        str: script path
    """
    with open(job_list_file) as f:
        num_jobs = len(f.readlines())

    # PBS job array script
    script = f"""#!/bin/bash
#PBS -N Co_corvus_array
#PBS -J 0-{num_jobs - 1}
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
INPUT_FILE=$(sed -n "$((PBS_ARRAY_INDEX + 1))p" {job_list_file})
INPUT_DIR=$(dirname "$INPUT_FILE")
INPUT_NAME=$(basename "$INPUT_FILE")

cd "$INPUT_DIR"
echo "Running: run-corvus -i $INPUT_NAME in $INPUT_DIR"
run-corvus -i "$INPUT_NAME"
"""

    with open(script_path, "w") as g:
        g.write(script)

    return script_path

def submit_corvus_job_array(job_list_file, script_path, poll_interval=300):
    """
    Submit the job array
    ARGS:
    -job_list_file: Path object of the .txt file that has all of the paths for the input files to run corvus on
    -script_path: Path object of the _qsub_array.script that we will call the qsub command on
    """        
    print(f"Submitting job array for {job_list_file}")
    with open(job_list_file) as h:
        num_jobs = len(h.readlines())
    
    qsub_array_command = [
        "qsub",
        "-J", f"0-{num_jobs - 1}",
        f"{script_path}"
    ]

    try:
            result = subprocess.run(qsub_array_command, check=True, text=True, capture_output=True)
            stdout = result.stdout.strip()
            print(f"Job submitted successfully: {stdout}")

            # Extract job ID from output
            match = re.search(r"(\d+)(?:\[\])?", stdout)
            if not match:
                raise ValueError("Could not parse job ID from qsub output.")
            
            job_id = match.group(1)
            print(f"Monitoring PBS job array with ID: {job_id}")

            # Poll until job disappears from qstat
            while True:
                qstat_result = subprocess.run(["qstat", f"{job_id}[]"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                if qstat_result.returncode != 0:
                    print("Job array is no longer in queue. Assuming it completed.")
                    break

                print(f"Job array {job_id} still running... sleeping for {poll_interval} seconds.")
                print(qstat_result)
                time.sleep(poll_interval)

            return True

    except subprocess.CalledProcessError as e:
        print(f"Error submitting job for {job_list_file}: {e.stderr}")
        return False


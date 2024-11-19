#!/usr/bin/env Corvus

import sys
import shutil 
import os
import json
#import logging 
#import corvus 
#import matplotlib
#import subprocess
#import corvus.controls
import re
import subprocess
from pathlib import Path
from pymatgen.io.cif import CifParser
from pymatgen.core import Structure
import time
import copy


"""
This is a full linear polarization XES script. It's goals:

1. Start in the CWD. Target a directory with a list of .cif files (created by pymatgen(eventually)) to this program.
It will cd into the target_directory and search the list of *.cifs if there is a directory of the same *.cif name.
If not, it will create a new directory with this name.

All of these operations will be written to a 'Calculations' directory. This way if the entire batch
is bad, you can just delete the 'Calculations' directory and you can restart with the *.cif batch 
still in target_directory ready to go again.

2. Creat a Corvus .in workflow to calculate the averaged spectrum between different linearly-polarized
spectrum. Creating subdirectories for the different spectra, "xy.*/", "yz.*/", "zx.*/".

-Upon Josh's new git branch 'polarization', the cfavg function will transcribe all three polarizations at once.
There is a large commented out section about changing the polarizations. Kept for historical sake 05AUG2024

3. (optional) Easy grapher function for the *.*xmu.dat files

"""
### Step 1: Global variables
target_directory = Path(sys.argv[1]).absolute()
CIF_PATHS = sorted(Path(target_directory).glob('*.cif'))
CIF_FILENAMES = [x.name for x in CIF_PATHS]

print(target_directory)

class Calculation:

    def __init__(self, cif_file):
        #These are Path objects for every instantiation.
        #Use this to read_csv in the graphing module
        self.cif_file = cif_file
        self.potential_files = None
        self.scf_files = None
        self.fms_files = None
        self.xmu_files = None
        self.cfg_avg_file = None

        #Use this as a dict to print off keys and values into an input file.
        #[0] is the location to print. All other key value pairs are meant to be written
        self.input_file = {'Output Path':None,
                            'cif_input':self.cif_file.name,
                            'absorbing_atom_type': None,
                            'feff.edge':'K',
                            'target_list':'cfavg',
                            'cfavg_target':'xes',
                            'feff.scf': '3.0 0 30 0.1 0',
                            'feff.fms': '4.0 0 0 0.0 0.0 40',
                            'feff.corehole':'None',
                            'Usehandlers':'Feff',
                            'feff.control':'1 1 1 1 1 1',
                            'feff.egrid':'e_grid -30 5 0.1',
                            'multiprocessing_ncpu': '2'}

        #Queryable info about the specific calculation instance.
        self.cif_information = None
        self.absorbing_atom = None
        self.calculation_time = time.strftime("%H:%M:%S", time.localtime())  # Current time in HH:MM:SS format
        self.calculation_date = time.strftime("%Y-%m-%d", time.localtime())

    def __str__(self):
        print(f"I'm the entire calculation object, I was created from {target_directory}")
        print(f"I am from the .cif {self.cif_file} in {Path.cwd()}")
        print(f"This is the input file: {self.input_file}")


    def __repr__(self):
        return f"[{self.cif_file}, {self.input_file}, {self.potential_files}, {self.scf_files}, {self.fms_files}]"
    
    def read_cif_file_custom_API(self, cif_file:Path)->dict: 
        """This is meant to iterate over an iterable. You loop this over all of the .cifs in a directory
        and you can draw out as much information from any API of your choice. I want to start using PMG 
        so I am using the 2024.07.08 pymatgen.io namespace https://pymatgen.org/pymatgen.io.html#pymatgen.io.cif.CifFile"""

        if cif_file.endswith('.cif'):

                #Try your own API here
                self.cif_information = []
                Calculation.cif_file()
                cif_stuff = CifParser(cif_file, occupancy_tolerance= 1.0, site_tolerance = 0.0001, frac_tolerance = 0.0001, check_cif = True, comp_tol = 0.01)
                cif_stuff.append(self.cif_information)
        
        else:
            print(f"We found {cif_file} in here that isn't a .cif.")
    
    def write_corvus_in_file(self, output_directory:Path):
        """Meant to be able to write the instance's constructed .in file and update the specific instance [0] key. 
        I want to get this hooked up with pyparsing and the corvus.config module
        for easy access."""

        in_file_path = output_directory / f"{output_directory.name}.in"

        with open(in_file_path, 'w') as f:  # Open the file in write mode
            for key, value in list(self.input_file.items())[1:]:  # Use items() to get key-value pairs
                f.write(f" {key} {{ {value} }}\n")
        
        self.input_file['Output Path'] = in_file_path

    def extract_elements(self) -> list:
        """
        Extract elements in string format (e.g., 'O', 'H', 'He') from a CIF file using Pymatgen.
        """
        # Load the structure from the CIF file
        structure = Structure.from_file(self.cif_file)
        
        # Extract unique elements from the structure
        elements = {str(site.specie) for site in structure}
        
        # Convert the set to a sorted list
        return sorted(elements)
    
    def update_input_file(self, key, value=None, start=None, increment=None, times=None):

        # Case 1: Directly updating with a provided value (string)
        if value is not None:
            self.input_file[key] = value
        
        # Case 2: Incremental counter values
        elif start is not None and increment is not None and times is not None:
            increments = [start + i * increment for i in range(times)]
            self.input_file[key] = increments
        else:
            raise ValueError("Provide either a 'value' or 'start', 'increment', and 'times' for incremental updates.")
        
        return self.input_file[key]
 
    def write_metadata_to_json(self, output_directory: Path):
        """Write instance variables to a hidden JSON metadata file."""
        metadata = {
            'cif_file': str(self.cif_file),
            #'potential_files': [str(p) for p in self.potential_files],
            #'scf_files': [str(s) for s in self.scf_files],
            #'fms_files': [str(f) for f in self.fms_files],
            #'xmu_files': [str(x) for x in self.xmu_files],
            #'cfg_avg_file': str(self.cfg_avg_file),
            'input_file': {str(key): str(value) for key, value in self.input_file.items()},
            'cif_information': self.cif_information,
            'absorbing_atom': self.absorbing_atom,
            'calculation_time': self.calculation_time,
            'calculation_date': self.calculation_date
        }

        # Create a hidden JSON file
        metadata_file_path = output_directory / '.metadata.json'
        
        with open(metadata_file_path, 'w') as json_file:
            json.dump(metadata, json_file, indent=4)

        print(f"Metadata written to hidden JSON file: {metadata_file_path}")

def make_calc_directory(target_directory)->dir:
    """This function will ask the user for individual parameters that they wish to increment over for every corvus.in file
    Ex: If you want to change the FMS radius from 2-10 au in increment steps of 2, or radius 3-4 au in increments of 0.2.
    This will ask for user input and turn into a whitespace separated list. All of this data now will be collected and 
    then sent through a wrapper function later down the line to write to the given .in file."""
    if not os.path.exists(target_directory):
        print(f"The target directory '{target_directory}' does not exist in the current working directory.")
        raise SystemExit
    
    # Asks for user input about what you want your calculation name to be, and detection of similar named directiories    
    global calculation_name
    calculation_name = input("What would you like to name your calculation? Letters and numbers are accepted: ")
    
    answer = None
    if os.path.exists(Path(target_directory) / calculation_name.replace('"','')):
        answer = input(f"The directory {calculation_name} already exists! Would you like to overwrite it? [y/n]. Case sensitive single letter answer please.")

        if not(answer in ['y', 'n', 'Y', 'N']):
            print("BAD ANSWER")
            quit()

        if answer == 'n' or 'N':
            print('Okay! Quiting ...')
            quit()
    
        else:
            shutil.rmtree(Path(target_directory) / calculation_name.replace('"',''))
    
    else:
        Path.mkdir(Path(target_directory) / calculation_name.replace('"',''))

    calc_directory_path = target_directory / calculation_name.replace('"','')
    
    return calc_directory_path

def copy_cifs_to_unique_directories(calc_directory:Path)->list:
    # Copy each .cif file to the newly created calculation directory
    all_calculation_instances = []
    for key, filename in enumerate(CIF_FILENAMES):
        
        compound_directory = calc_directory / filename.replace('.cif', '')
        os.mkdir(compound_directory)
        shutil.copy(CIF_PATHS[key], compound_directory)
        calculation_instance = Calculation(cif_file=Path(compound_directory / filename))
        all_calculation_instances.append(calculation_instance)
        print(f"File '{filename}' copied to '{compound_directory}'.")

    return all_calculation_instances



def make_dir_with_suffix(dir_path:Path, suffix:str)->dir:
    """First arg is the name you want to make the directory. Second arg is the path to another directory you want to make the new dir at.
    *args is meant to be a list or tuple of strings. You use this function to iterate over incremental values.
    List of strings is intentional for the write_corvus_in_file class method, as it is intended to print values of strings.
    These strings will also update the self.in_file."""
    if isinstance(suffix, str):
        dir_path = Path(dir_path).parent  # Get the parent directory path
        dir_name = Path(dir_path).name    # Get the filename with .cif extension

    # Remove the .cif suffix and add the new suffix to the directory name
        new_dir_name = dir_name.replace('.cif', '') + f'_{suffix}'

    # Construct the new directory path
        new_dir = dir_path / new_dir_name

    # Create the new directory
        new_dir.mkdir(parents=True, exist_ok=False)
        print(f"New directory created at: {new_dir}")

    else:
        print("The list contains non-string elements or is not a list.")
    return new_dir

def write_qsub_script(instance):
    """Write's the qsub.script to every individual calculation directory as we want.
    I want this separated as we can change this to match whatever system we want and keep
    it modular."""

    job_directory = instance.input_file['Output Path'].parent

    # Create the qsub.script file inside the job_directory
    script_path = job_directory / "qsub.script"  

    with open(script_path, "w") as script_file:
        script_file.write("#!/bin/bash -l\n") 
        script_file.write("\n")
        script_file.write("# Find all the `.cif` files in the current directory\n")
        script_file.write("cif_files=(*.cif)\n")
        script_file.write("\n")
        script_file.write("# Check if there are any `.cif` files found\n")
        script_file.write("echo \"Found CIF files: ${cif_files[@]}\"\n")
        script_file.write("\n")
        script_file.write("echo $cif_files\n")
        script_file.write("\n")
        script_file.write("# Use the first .cif file to set the job name, as PBS only allows one job name\n")
        script_file.write("cif_basename=$(basename \"${cif_files[0]}\" .cif)\n")
        script_file.write("# Set the job name dynamically\n")
        script_file.write("#PBS -N \"$cif_basename\"\n")
        script_file.write("#PBS -l nodes=1:ppn=1\n")
        script_file.write(f"#PBS -o {job_directory}/job_${{cif_basename}}.sout\n")
        script_file.write(f"#PBS -e {job_directory}/job_${{cif_basename}}.serr\n")
        script_file.write("#PBS -V\n")
        script_file.write("\n")
        script_file.write("# Use the first argument ($1) as the job directory\n")
        script_file.write("job_directory=\"$job_directory\"\n")
        script_file.write("\n")
        script_file.write("# Change to the job directory\n")
        script_file.write("cd \"$job_directory\" || exit 1  # Exit if the directory change fails\n")
        script_file.write("\n")
        script_file.write("# Activate conda environment\n") 
        script_file.write("conda activate Corvus2\n")
        script_file.write("if [ $? -ne 0 ]; then\n")
        script_file.write("    echo \"Failed to activate conda environment\"\n")
        script_file.write("    exit 1\n")
        script_file.write("fi\n")
        script_file.write("\n")
        script_file.write("# Function to search for all `.in` files and run the `run-corvus` command\n")
        script_file.write("search_and_run() {\n")
        script_file.write("    local files=$(find . -type f -name \"*.in\")\n")
        script_file.write("    for file in $files; do\n")
        script_file.write("        # Start time for the job\n")
        script_file.write("        start_time=$(date +%s)\n")
        script_file.write("\n")
        script_file.write("        # Clean up old files (if any)\n")
        script_file.write("        rm -rf Corvus1_PYMATGEN/ Corvus2_helper/ Corvus.cfavg.out Corvus.nest\n")
        script_file.write("\n")
        script_file.write("        # Run the Corvus command and redirect output\n")
        script_file.write("        run-corvus -i *.in &> testing.out\n")
        script_file.write("    done\n")
        script_file.write("}\n")
        script_file.write("\n")
        script_file.write("search_and_run\n")  

def submit_job_with_directory(instance):
    """Submits a job using a central qsub script but specifies different working directories for each job."""
    
    job_directory = instance.input_file['Output Path'].parent
    script_path = job_directory / "qsub.script"
    
    # Use the qsub `-d` option to specify the directory in which the job should run
    process = subprocess.Popen(
        ['qsub', '-v', f'job_directory={job_directory}', str(script_path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        raise RuntimeError(f"Error submitting job: {stderr.decode().strip()}")

    job_id = stdout.decode().strip().split('.')[0]
    return job_id

def wait_for_job(job_id):

    while True:
        time.sleep(30)  # Wait for a while before checking again
        process = subprocess.Popen(['qstat'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            raise RuntimeError(f"Error checking job status: {stderr.decode().strip()}")

        # Check if job ID is still in the output of qstat
        if job_id not in stdout.decode():
            print(f"Job {job_id} has finished.")
            break
        else:
            print(f"Job {job_id} is still running...")

def print_job_output(job_id):
   
    output_file = f"{job_id}.o{job_id.split('.')[0]}"  # Adjust as necessary based on your job output naming convention

    try:
        with open(output_file, 'r') as f:
            print(f"Contents of job {job_id} output:")
            print(f.read())
    except FileNotFoundError:
        print(f"Output file for job {job_id} not found.") 

# def search_and_run_qsub_script(directory):
#     """Meant to hit the customized qsub.script in a directory. Will only hit one time. Meant to be used to walk along
#     and entire floor of a file tree"""
#     for root, dirs, files in os.walk(directory):
#         if 'qsub.script' in files:
#             qsub_script_path = os.path.join(root, 'qsub.script')
#             try:
#                 subprocess.run(['qsub', qsub_script_path], cwd=root, check=True)
#                 print(f"Successfully ran 'qsub qsub.script' in {root}")
#             except subprocess.CalledProcessError as e:
#                 print(f"Error running 'qsub qsub.script' in {root}: {e}")
#                 # Handle the error as needed

#This is how the program is called 'python Full_Polarization.py [NAME OF TARGET DIRECTORY OF CIFS]'
if __name__ == "__main__":
    calc_directory = make_calc_directory(target_directory)
    all_calculation_instances = copy_cifs_to_unique_directories(calc_directory)

    print(all_calculation_instances)
    
    new_instances = []
    
    for instance in all_calculation_instances:

        #instance.clean_cif(instance.cif_file)
        ox_states = instance.extract_elements()

        for ox_state in ox_states:
            copied_instance = copy.deepcopy(instance)
            copied_instance.input_file['absorbing_atom_type'] = ox_state
            print('I am reaching here')
            new_instances.append(copied_instance)
            print(new_instances)
            new_dir = make_dir_with_suffix(copied_instance.cif_file, ox_state)
            shutil.copy(copied_instance.cif_file, new_dir)
            copied_instance.write_corvus_in_file(new_dir)
            copied_instance.write_metadata_to_json(new_dir)
            write_qsub_script(copied_instance)
            job_id = submit_job_with_directory(copied_instance)
            print_job_output(job_id)

if len(sys.argv) != 2:
    print("Usage: python Full_Polarization.py <target_directory>")
    sys.exit(1)


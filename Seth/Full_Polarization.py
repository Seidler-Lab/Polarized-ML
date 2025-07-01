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
TARGET_DIRECTORY = Path(sys.argv[1]).absolute()
CIF_PATHS = sorted(Path(TARGET_DIRECTORY).glob('*.cif'))
CIF_FILENAMES = [x.name for x in CIF_PATHS]

print(TARGET_DIRECTORY)

class Calculation:

    def __init__(self, cif_file):
        #These are Path objects for every instantiation.
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
                            'cfavg.target':'xes',
                            'feff.scf': '5.0 0 30 0.1 0',
                            'feff.fms': '5.0 0 0 0.0 0.0 40',
                            'feff.corehole':'None',
                            'Usehandlers':'Feff',
                            'feff.control':'1 1 1 1 1 1',
                            'feff.egrid':'e_grid -30 5 0.1',
                            'multiprocessing.ncpu': '2'}

        #Queryable info about the specific calculation instance.
        self.cif_information = None
        self.absorbing_atom = None
        self.calculation_time = time.strftime("%H:%M:%S", time.localtime())  # Current time in HH:MM:SS format
        self.calculation_date = time.strftime("%Y-%m-%d", time.localtime())

    def __str__(self):
        return (f"I'm the entire calculation object, I was created from {TARGET_DIRECTORY}\n"
                f"I am from the .cif {self.cif_file} in {Path.cwd()}\n"
                f"This is the input file: {self.input_file}")


    def __repr__(self):
        return f"[{self.cif_file}, {self.input_file}, {self.potential_files}, {self.scf_files}, {self.fms_files}]"
    
    def read_cif_file_custom_API(self, cif_file:Path)->dict: 
        """This is meant to iterate over an iterable. You loop this over all of the .cifs in a directory
        and you can draw out as much information from any API of your choice. I want to start using PMG 
        so I am using the 2024.07.08 pymatgen.io namespace https://pymatgen.org/pymatgen.io.html#pymatgen.io.cif.CifFile"""

        if cif_file.suffix == '.cif':
            
            try:
                parser = CifParser(cif_file)
                self.cif_information = parser.parse_structures()

            except Exception as e:
                print(f"Error parsin CIF file {cif_file}: {e}") 
        
        else:
            TypeError("This file is not a .cif file!", type(cif_file))

    def write_corvus_in_file(self, output_directory:Path):
        """Meant to be able to write the instance's constructed .in file and update the specific instance [0] key. 
        I want to get this hooked up with pyparsing and the corvus.config module
        for easy access."""

        in_file_path = output_directory / f"{output_directory.name}.in"

        with open(in_file_path, 'w') as f:  # Open the file in write mode
            for key, value in list(self.input_file.items())[1:]:  # Use items() to get key-value pairs
                f.write(f" {key} {{ {value} }}\n")
        
        self.input_file['Output Path'] = in_file_path

    def extract_elements(self) -> set:
        """
        Extract elements in string format (e.g., 'O', 'H', 'He') from a CIF file using Pymatgen, and places them in a set.
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
            #'cif_information': self.cif_information,
            'absorbing_atom': self.absorbing_atom,
            'calculation_time': self.calculation_time,
            'calculation_date': self.calculation_date
        }

        # Create a hidden JSON file
        metadata_file_path = output_directory / '.metadata.json'
        
        with open(metadata_file_path, 'w') as json_file:
            json.dump(metadata, json_file, indent=4)

        print(f"Metadata written to hidden JSON file: {metadata_file_path}")

def make_calc_directory(target_directory)->Path:
    """
    This function creates the calculation directory inside of the target directory.
    First it checks whether the target directory even exists. After that it prompts the user to 
    create a name for the calculation directory.
    """

    """
    INPUT PARAMETERS: 
    target_directory: system argument from the command line by calling the module 'python Full_Polarization.py {target_directory}

    OUTPUT:
    creates the calculation direcotry inside of the target directory
    """

    """
    RETURN:
    path of the calculation directory
    """

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
        calc_directory_path = target_directory / calculation_name.replace('"','')
        Path.mkdir(calc_directory_path)
    
    print("This is the path where all of the calculations will go to:", calc_directory_path)
    return calc_directory_path

def copy_cifs_to_calc_directory(target_directory:Path, calc_directory:Path):
    """
    This function should be called when all .cif files have been downloaded into the target_directory.
    Goes through every single file and ensures it ends in the extension .cif and moves it to the newly
    created calc_directory (the output from make_calc_directory()). All other files are printed as 
    errors.
    """

    """
    INPUT PARAMETERS:
    target_directory: a directory Path object with .cif files (or any other file type) inside
    
    calc_directory: a directory Path object that all of the .cif files will be moved to
    
    OUTPUT: all .cif files will now be inside calc_directory
    """

    if not isinstance(target_directory, Path):
        TypeError("The first parameter must be a Path object")
    
    if not isinstance(calc_directory, Path):
        TypeError("The second parameter must be a Path object")

    if not target_directory.is_dir():
        ValueError("The first parameter must be a directory: ", target_directory)

    if not calc_directory.is_dir():
        ValueError("The second parameter must be a directory: ", calc_directory)

    for file in target_directory.iterdir():
        if file.is_file() and file.suffix == '.cif':
            shutil.copy(file, calc_directory)
            print(f"Copied {file} to {calc_directory}")

        elif not file.is_file():
            TypeError("This is not a file. Skipping: ", file)
        
        elif file.suffix != '.cif':
            ValueError("This file does not end in the suffix .cif. Skipping: ", file)
    
    print("All .cif files copied to the Calculation Directory!")


def make_dir_with_suffix(dir_path:Path, endpiece:str)->dir:
    """
    Creates a new directory with a new suffix. Intended to increment over a specific parameter multiple times
    """

    """
    INPUT:
    dir_path: directory path object that you want to nest new directories inside
    
    suffix: accepts a string, and adds it as an underscore at the end of the directory path     
    
    OUTPUT:
    new directory inside of dir_path
    """

    """
    Return:
    new directory nested inside of dir_path
    """

    if not isinstance(dir_path, Path):
        raise TypeError("The first parameter must be a Path object: ", dir_path)
    
    # if not dir_path.is_dir():
    #     raise ValueError("The first parameter must be a directory: ", dir_path)
    
    if not isinstance(endpiece, str):
        raise TypeError("The second parameter must be a string: ", endpiece)
    
    dir_path = Path(dir_path)
    
    new_dir = dir_path / f'{dir_path.stem}_{endpiece}'

    new_dir.mkdir(parents=True, exist_ok=True)

    print(f"New directory created at: {new_dir}")

    return new_dir

def write_qsub_script(instance):
    """Write's the qsub.script to every individual calculation directory as we want.
    I want this separated as we can change this to match whatever system we want and keep
    it modular."""

    """
    INPUT:
    A Calculation object
    """

    """
    OUTPUT:
    A qsub.script file is written to the directory that is located at the Calculation
    object's Output Path value using it's key
    """

    job_directory = Path(instance.input_file['Output Path']).parent
    print("This is the job directory", job_directory)
    # Create the qsub.script file inside the job_directory
    script_path = job_directory / "qsub.script"  

    with open(script_path, "w") as script_file:
        script_file.write("#!/bin/bash -l\n")
        script_file.write("set -x \n") 
        script_file.write("\n")
        script_file.write("echo \"JOB_DIRECTORY=${JOB_DIRECTORY}\" >> debug.log \n")
        script_file.write("# PBS Directives \n")
        script_file.write(f"#PBS -N {job_directory.name}\n")
        script_file.write("#PBS -l nodes=1:ppn=1\n")
        script_file.write(f"#PBS -o {job_directory}/job_${job_directory.name}.sout\n")
        script_file.write(f"#PBS -e {job_directory}/job_${job_directory.name}.serr\n")
        script_file.write("#PBS -V\n")
        script_file.write("\n")
        script_file.write("# Use the argument from the subprocess.run as the job directory\n")
        script_file.write("job_directory=\"${JOB_DIRECTORY}\" \n")
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
        script_file.write("    files=$(find . -type f -name \"*.in\")\n")
        script_file.write("    for file in $files; do\n")
        script_file.write("        # Start time for the job\n")
        script_file.write("        start_time=$(date +%s)\n")
        script_file.write("\n")
        script_file.write("        # Clean up old files (if any)\n")
        script_file.write("        rm -rf Corvus1_PYMATGEN/ Corvus2_helper/ Corvus.cfavg.out Corvus.nest\n")
        script_file.write("\n")
        script_file.write("        # Run the Corvus command and redirect output\n")
        script_file.write("        echo \"Running Corvus on ${file}\" >> testing.out \n")
        script_file.write("        run-corvus -i \"${file}\" >> testing.out 2>&1 || echo \"Failed to run-corvus on ${file}\" >> testing.out \n")
        script_file.write("        echo \"Completed Corvus on ${file}\" >> testing.out \n")
        script_file.write("    done\n")
        script_file.write("}\n")
        script_file.write("\n")
        script_file.write("search_and_run\n")  

def submit_job_with_directory(instance):
    """Submits a job using a central qsub script but specifies different working directories for each job."""

    """
    INPUT:
    A Calculation object
    """

    """
    OUTPUT:
    Initiation of the run-corvus command by usage of the qsub.scipt file, and submission
    to the given job system
    """

    
    job_directory = instance.input_file['Output Path'].parent

    script_path = job_directory / "qsub.script"
    print("This is the script path that is being initiated:", script_path)

    job_name = job_directory.name

    if not job_directory.exists():
        raise FileNotFoundError(f"Job directory does not existL {job_directory}")
    
    if not script_path.exists():
        raise FileNotFoundError(f"Script file does not exist: {script_path}")
    
    try:

    # Use the qsub `-v` option to specify the directory in which the job should run
        process = subprocess.run(
            ['qsub', '-N', job_name, '-v', f'JOB_DIRECTORY={job_directory}', str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # vstdout="/dev/null",
            # vstderr="/dev/null",
            check=True,
            text=True

        )

    except subprocess.CalledProcessError as e:
        print(f"Error submitting job: \nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}")
        raise RuntimeError(f"Error submitting jobL {e.stderr.strip()}") from e

    job_id = process.stdout.strip().split('.')[0]
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
        print(f"output file for job {job_id} not found.") 

def reference_cif(mpapi: str, calc_directory: Path)-> list:
    """
    Downloads all of the reference CIFs for linear background.
    """

    """
    INPUT:
    mpapi: personal mp api key
    

    calc_directory: the calc_directory already created. should already be made up of entirely .cif files
    """

    """
    OUTPUT:
    download one or more of these unique transition metal .cifs into the calc_directory
    """
    from pymatgen.ext.matproj import MPRester
    
    #these are all non-symmetrized until further notice 25nov
    references = [
        "mp-644481", #Sc
        "mp-1215", #Ti
        "mp-18937", #V
        "mp-19177", #Cr
        "mp-510408", #Mn
        "mp-19770", #Fe
        "mp-22408", #Co
        "mp-19009", #Ni
        "mp-704645", #Cu
        "mp-2133", #Zn
    ]

    downloaded_files = []

    with MPRester(mpapi) as mpr:
        for reference in references:
            try:
                # get the structure using the material id
                structure = mpr.get_structure_by_material_id(reference)

                # generate the path where the .cif file will be saved
                file_path = calc_directory / f"{reference}.cif"
                            
                # save the structure as a .cif file
                structure.to(filename=file_path, fmt="cif")
                downloaded_files.append(file_path)

                print(f"downloaded and saved {reference} reference at {file_path}")
                        
            except Exception as e:
                print(f"error downloading {reference}: {e}")
            
            else:
                print(f"no reference mp-id found for linear background subtraction {reference}. skipping.")

    return downloaded_files

#this is how the program is called 'python full_polarization.py [name of target directory of cifs]'
if __name__ == "__main__":
    calc_directory = make_calc_directory(TARGET_DIRECTORY)
    copy_cifs_to_calc_directory(TARGET_DIRECTORY, calc_directory)   

   #initial run for control 1 1 1 1 1 1 
    all_calculation_instances = [] 

    # reference_cifs = reference_cif(mpapi= 'Vw5EOA3uyseD8Hi81bsRXYA1XIX2lXiY', calc_directory= calc_directory)
    # for reference in reference_cifs:
    #     reference_instance = Calculation(reference)
    #     all_calculation_instances.append(reference_instance)
    #     Calculation.read_cif_file_custom_API(reference_instance, reference_instance.cif_file)
    
    for cif_file in CIF_PATHS:
        calc_instance = Calculation(cif_file)
        all_calculation_instances.append(calc_instance)
        print(all_calculation_instances)
        Calculation.read_cif_file_custom_API(calc_instance, calc_instance.cif_file)
        
    #downloads the reference .cifs for linear background deletion (specifically for xes)
    #reference_cifs = reference_cif(mpapi= 'Vw5EOA3uyseD8Hi81bsRXYA1XIX2lXiY', calc_directory= calc_directory)
    #for reference in reference_cifs:
    #    reference_instance = Calculation(reference)
    #    all_calculation_instances.append(reference_instance)

    print(f"These are all of the calculation instances: {all_calculation_instances}")
    for instance in all_calculation_instances:

        elements = instance.extract_elements()
        print(f"These are the elements: {elements} from this calculation instance: {instance}")

        if 'Cr' in elements:
            copied_instance = copy.deepcopy(instance)
            copied_instance.input_file['absorbing_atom_type'] = 'Cr' 
            cif_file_path = calc_directory / copied_instance.cif_file
            new_dir = make_dir_with_suffix(cif_file_path.parent / cif_file_path.stem, 'Cr')
            shutil.copy(copied_instance.cif_file, new_dir)
            copied_instance.write_corvus_in_file(new_dir)
            copied_instance.write_metadata_to_json(new_dir)
            write_qsub_script(copied_instance)
            job_id = submit_job_with_directory(copied_instance)
   

if len(sys.argv) != 2:
    print("Usage: python Full_Polarization.py <target_directory>")
    sys.exit(1)


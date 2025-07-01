#!/usr/bin/env Corvus

import qsub_job_create_n_submission as qsub

import sys
import shutil 
import os
import json
#import logging 
#import corvus 
#import matplotlib
#import subprocess
#import corvus.controls
#import re
import numpy as np
import subprocess
from pathlib import Path
from pymatgen.io.cif import CifParser
from pymatgen.core import Structure, Composition
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.ext.matproj import MPRester

import time


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
ABSORBING_ATOM = sys.argv[2]
CIF_PATHS = sorted(Path(TARGET_DIRECTORY).glob('*.cif'))
CIF_FILENAMES = [x.name for x in CIF_PATHS]

print(TARGET_DIRECTORY)

class Calculation:

    def __init__(self, cif_file):
        #These are Path objects for every instantiation.
        self.cif_file = cif_file
        self.reduced_formula = None
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
                            'target_list':'cluster_array',
                            'cfavg.target':'xes',
                            'feff.scf': '3.0 0 30 0.1 0',
                            'feff.fms': '3.0 0 0 0.0 0.0 40',
                            'feff.corehole':'None',
                            'Usehandlers':'Feff',
                            'feff.control':'1 1 1 1 1 1',
                            'feff.egrid':'e_grid -30 5 0.1',
                            'feff.MPI.CMD' : 'mpirun',
                            'feff.MPI.ARGS' : '-n 1',
                            #'multiprocessing.ncpu': '1'}
        }
        #Queryable info about the specific calculation instance.
        self.cif_information = None
        self.absorbing_atom = None
        self.calculation_time = time.strftime("%H:%M:%S", time.localtime())  # Current time in HH:MM:SS format
        self.calculation_date = time.strftime("%Y-%m-%d", time.localtime())
        self.quadropole_tensor = None
        self.diagonalized_quadropole_tensor = None

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
                self.reduced_formula = Structure.reduced_formula(cif_file)

            except Exception as e:
                print(f"Error parsin CIF file {cif_file}: {e}") 
        
        else:
            TypeError("This file is not a .cif file!", type(cif_file))

    def write_corvus_in_file(self, output_directory:Path):
        """Meant to be able to write the instance's constructed .in file and update the specific instance [0] key. 
        I want to get this hooked up with pyparsing and the corvus.config module
        for easy access."""

        in_file_path = output_directory / f"{output_directory.name}.in"

        # print("This is where I am writing the input file to: ", in_file_path)

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
            'calculation_date': self.calculation_date,
            'quadrupole tensor': self.quadropole_tensor,
            'diagonalized quadrupole tensor': self.diagonalized_quadropole_tensor
        }

        # Create a hidden JSON file
        metadata_file_path = output_directory / '.metadata.json'
        
        with open(metadata_file_path, 'w') as json_file:
            json.dump(metadata, json_file, indent=4)

        # print(f"Metadata written to hidden JSON file: {metadata_file_path}")

    # def get_reduced_formula_name_from_cif_file(self):
    #     """Get's reduced formula name to be passed to quadropole function for matrix"""

    #     reduced_formula_for_cif = Structure.composition.reduced_formula(self.cif_file())
    #     print("This is in get_reduced_formula_name_from_cif_file the reduced formula for the cif", reduced_formula_for_cif)

    #     return reduced_formula_for_cif


def get_oxidation_state_formula(cif_file:Path) -> dict:
    """
    Determines the oxidation states of each element in a given formula.
    1. First tries to make a Composition object that usese the oxi_state_guesses
        to try to parse the oxidation states
    2. Secondly attempts to use a Structure object to use the Bond Valence Analyzer
        to guess the oxidation states
    3. Thirdly, uses the MPRester summary search to see if the online Materials project
        has possible species
    
    Args:
        Path-like object: Path to the cif file
    
    Returns:
        dict: A dictionary mapping each element to its oxidation state(s).
        ex: {'V':2.0, 'O':-2.0}
    """

    from collections import Counter

    try:
        cif_parser = CifParser(cif_file)

        structure = cif_parser.parse_structures()[0]
        composition = structure.composition
        reduced_formula = composition.reduced_formula
        # Use pymatgen's Composition to parse the formula
        composition = Composition(reduced_formula)
        
        # Try oxidation state guessing
        oxidation_states = composition.oxi_state_guesses()

        if oxidation_states:
            #print(f"[USING COMPOSITION GUESS] This is the oxidation state of {reduced_formula}: {oxidation_states}")
            # Return the first guess (most probable based on pymatgen's algorithm)
            return oxidation_states[0]
        
        #print("[BOND VALENCE ANALYZER GUESS] Trying BVA")
        bv = BVAnalyzer()

        oxidized_structure = bv.get_oxi_state_decorated_structure(structure)
        #print(oxidized_structure)

        oxi_dict = {}
        
        for site in oxidized_structure:
            #print("This is a site in the oxidized Structure object: ", site)
            specie = site.specie
            #print(f"Specie: {specie}, type: {type(specie)}")
            elem = specie.element.symbol
            oxi = specie.oxi_state
            if elem not in oxi_dict:
                oxi_dict[elem] = []
            
            oxi_dict[elem].append(oxi)

        oxi_summary = {el: Counter(oxis).most_common(1)[0][0] for el, oxis in oxi_dict.items()}
        #print("This is the oxidation dictionary after finding the most common oxidation states: ", oxi_summary)

        return oxi_summary

        # else:

        #     with MPRester(MPAPIKEY) as mpr:
        #         results = mpr.summary.search(formula = reduced_formula)
        #         print("This is the full list of results: ", results)

        #         material = results[0]

        #         possible_species = material.possible_species
        #         print("This is the possible species of the material: ", possible_species)
        #         return "Oxidation states could not be determined."

    except Exception as e:
        return f"An error occurred: {e}"

    
def get_Nx4_arrays_from_cluster_array_json(oxidation_dict, cluster_array_json_path):
    """
    Reads in cluster_array.json and for every cluster, creates an Nx4 
    NumPy array of the atomic positions. This output can be used in the
    quadrupole function.

    Parameters:
        cluster_array_json_path (str): Path to the cluster JSON file (default: 'cluster_array.json')

    Returns:
        list of np.ndarray: A list of Nx3 arrays, one for each cluster
    """

    assert isinstance(oxidation_dict, dict)

    #print(f'This is the path of the cluster array {cluster_array_json_path}')

    with open(cluster_array_json_path, 'r') as f:
        clusters = json.load(f)

    cluster_arrays = []

    # The data should be in the format: [id1, value1, atoms1, id2, value2, atoms2, ...]
    for cluster_entry in clusters:
        #print('This is the cluster entry ', cluster_entry)
        atom_list = cluster_entry[2]
        cluster = []
        for atom in atom_list:
            #print("This is in the Nx4 arrays. This is atom", atom)
            element = atom[0]
            x,y,z = atom[1], atom[2], atom[3]
            charge = oxidation_dict.get(element, None)
            #print('This is the result of the charge from the oxidation_dict ', charge)

            if charge is None:
                #print("This charge cannot be determined, skippin")
                continue
            

            dist = np.sqrt((x**2 + y**2 + z**2))
            if dist > 0: # Avoid atom at 0, 0, 0
                cluster.append([charge, x, y, z])

        cluster_arrays.append(np.array(cluster))

    #print("This is in Nx4 function. This is the full cluster array of all the clusters: ", cluster_arrays)

    return cluster_arrays

def quadrupole_moment(cluster_arrays):
    """
    Calculate the non traceless form of the quadrupole moment tensor for a system of point charges. 

    Args:
    - cluster_arrays: list of M, Nx4 arrays. M is the number of clusters that are built out where N is the number of particles in each cluster. Each N row is the (charge, x, y, z) charge and coordinates of a particle.

    Returns:
    - Q: A list of 3x3 numpy arrays representing the quadrupole moment tensor for each cluster from the cluster arrays.
    """
    assert(isinstance(cluster_arrays, list))
    
    list_quadrupole_moments = []

    for cluster in cluster_arrays:
        #print("This is in quadrupole_moment. This is the cluster we are looking at")
        #print(cluster)
        Q = np.zeros(
            # Initialize the quadrupole moment tensor as a 3x3 zero matrix.
            (3, 3))
        for row in cluster:
            charge, r_x, r_y, r_z = row
            # print("This is the x ", r_x)
            # print(r_y)
            # print(r_z)

            dist = (r_x**2 + r_y**2 + r_z**2)**(1/2)
            # print(charge)
            # print(dist)
            norm_factor = charge / dist**7

            # Update the Q matrix using the formula.
            Q[0, 0] += norm_factor * (r_x * r_x)
            Q[0, 1] += norm_factor * (r_x * r_y)
            Q[0, 2] += norm_factor * (r_x * r_z)

            Q[1, 0] += norm_factor * (r_y * r_x)
            Q[1, 1] += norm_factor * (r_y * r_y)
            Q[1, 2] += norm_factor * (r_y * r_z)

            Q[2, 0] += norm_factor * (r_z * r_x)
            Q[2, 1] += norm_factor * (r_z * r_y)
            Q[2, 2] += norm_factor * (r_z * r_z)

        list_quadrupole_moments.append(Q)
        #print("This is in qudrupole_moment. This is the quadrupole matrix of this cluster")
        #print(Q)
    
    # print("These are all of the quadrupole moment tensors: ", list_quadrupole_moments)
    return list_quadrupole_moments

def average_over_all_quadrupole_moments(list_of_quadrupole_matrices):
    """Sum all quadrupole matrices and divide by N qudropole matrices in list"""

    assert(list(list_of_quadrupole_matrices))

    sum_of_quadrupole_matrices = np.sum(list_of_quadrupole_matrices, axis=0)
    # print(sum_of_quadrupole_matrices)

    # Compute average
    average_quadrupole_matrix = sum_of_quadrupole_matrices / len(list_of_quadrupole_matrices)

    # print("This is the average ", average_quadrupole_matrix)

    #print("This is the average quadrupole matrix for this material:\n", average_quadrupole_matrix) 
    return average_quadrupole_matrix

def diagonalize_matrix(matrix):
    """
    Diagonalizes a given square matrix using eigen-decomposition.
    
    Parameters:
        matrix (ndarray): A square NumPy array (e.g., 3x3)
        
    Returns:
        eigenvalues (ndarray): 1D array of eigenvalues
        eigenvectors (ndarray): 2D array whose columns are the normalized eigenvectors
    """
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    try:
        R_inv = np.linalg.inv(eigenvectors)
        D = R_inv @ matrix @ eigenvectors
        D_clean = np.round(D, decimals=10)

    except np.linalg.LinAlgError:
        print("Eigenvector matrix is not invertible")
        return eigenvalues, eigenvectors, None
    
    # print("These are the eigenvalues: ", eigenvalues)
    # print("These are the eigenvectors: ", eigenvectors)
    # print("This is the diagaonal matrix D = R^-1 @ A @ R:\n:", D)
    #print("This is the clean diagonalized matrix that completely removes the off diagonal: ", D_clean) 
    return eigenvalues, eigenvectors, D_clean

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
    
    else:
        calc_directory_path = f"{target_directory.name}_{ABSORBING_ATOM}_{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())}"

        Path.mkdir(target_directory / calc_directory_path)
    
    print("This is the path where all of the calculations will go to:", calc_directory_path)
    return Path(target_directory / calc_directory_path)

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

    Path.mkdir(new_dir, parents=True, exist_ok=True)

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
    """Submits a job using a central qsub script but specifies different working directories for each job.

    INPUT:
    A Calculation object

    OUTPUT:
    Initiation of the run-corvus command by usage of the qsub.scipt file, and submission
    to the given job system
    """

    
    job_directory = instance.input_file['Output Path'].parent

    script_path = job_directory / "qsub.script"
    # print("This is the script path that is being initiated:", script_path)

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

def wait_for_file(file_path, timeout=1200, check_interval=5):
    """
    Waits for a file to appear on disk.

    Parameters:
    - file_path: Path object or str to the target file.
    - timeout: Max time to wait in seconds.
    - check_interval: How often to check in seconds.
    
    Raises:
    - FileNotFoundError: if the file does not appear in time.
    """
    file_path = Path(file_path)
    start_time = time.time()
    
    while not file_path.exists():
        if time.time() - start_time > timeout:
            raise FileNotFoundError(f"File {file_path} did not appear within {timeout} seconds.")
        
        print("The file was not found, going back to sleep")
        time.sleep(check_interval)

    # print(f"File {file_path} is now available.")

def format_polarization_block(matrix):
    """
    Converts a 3x3 matrix into a formatted string for polarization input.
    
    Example output:
    polarization {
    22.609 0 0
    0 1376.769 0
    0 0 1376.769
    }
    """
    lines = []
    for row in matrix:
        formatted_row = ' '.join(f"{val:.3f}" for val in row)
        lines.append(formatted_row)

    #print("These are the lines in format_polarization_block ", lines)
    #print('\n'.join(lines))

    return '\n'.join(lines)

def main(TARGET_DIRECTORY, ABSORBING_ATOM):
    """Is the main function"""

    calc_directory = make_calc_directory(TARGET_DIRECTORY)
    copy_cifs_to_calc_directory(TARGET_DIRECTORY, calc_directory)   

   #initial run for control 1 1 1 1 1 1 
    all_calculation_instances = [] 

    # reference_cifs = reference_cif(mpapi= MPAPIKEY, calc_directory= calc_directory)
    # for reference in reference_cifs:
    #     reference_instance = Calculation(reference)
    #     all_calculation_instances.append(reference_instance)
    #     Calculation.read_cif_file_custom_API(reference_instance, reference_instance.cif_file)
    
    for cif_file in CIF_PATHS:
        calc_instance = Calculation(cif_file)
        shutil.copy(src= cif_file, dst=calc_directory)
        all_calculation_instances.append(calc_instance)
        #Calculation.read_cif_file_custom_API(calc_instance, calc_instance.cif_file)
        
    # print(f"These are all of the calculation instances: {all_calculation_instances}")
    for instance in all_calculation_instances:

        elements = ABSORBING_ATOM

        if ABSORBING_ATOM in elements:
            instance.input_file['absorbing_atom_type'] = ABSORBING_ATOM 
            cif_file_path = calc_directory / instance.cif_file
            #print('This is the cif_file_path', cif_file_path)
            new_dir = make_dir_with_suffix(calc_directory / cif_file_path.stem, ABSORBING_ATOM)
            shutil.copy(instance.cif_file, new_dir)
            print("This is the new_dir", new_dir)
            instance.write_corvus_in_file(new_dir)
            
    corvus_in_file_list = qsub.create_list_of_corvus_input_files(calc_directory)
    input_list_file = qsub.save_input_file_list(corvus_in_file_list)
    script_path = qsub.write_corvus_array_script(input_list_file)
    qsub.submit_corvus_job_array(input_list_file, script_path)
    #All of the files should be finished
    print("It is now finished!")

    good_calculation_list = []

    for instance in all_calculation_instances:
        individual_path = Path(calc_directory / instance.cif_file.stem / f"{instance.cif_file.stem}_{ABSORBING_ATOM}")
        #print("This is the path to this calculation:", individual_path)

        oxidation_dict = get_oxidation_state_formula(instance.cif_file)
        if isinstance(oxidation_dict, str):
            print("[SKIPPING] oxidation state cannot be determined: ", oxidation_dict)
            print(f"I am breaking the for loop because I am {oxidation_dict} this is for {individual_path}")
            continue

        #print('This is the oxidation dict: ', oxidation_dict)
        #print(f"This is supposed to be a dictionary, {type(oxidation_dict)}")

        wait_for_file(file_path=new_dir/"cluster_array.json")
        list_of_cluster_arrays = get_Nx4_arrays_from_cluster_array_json(oxidation_dict, cluster_array_json_path=f'{individual_path}/cluster_array.json')
        #print("This is the list of cluster arrays: ", list_of_cluster_arrays)

        list_of_quadrupole_matrices = quadrupole_moment(list_of_cluster_arrays)
        #print("This is the list of quadrupole matricies: ", list_of_quadrupole_matrices)

        avg_quad_matrix = average_over_all_quadrupole_moments(list_of_quadrupole_matrices)
        #print(f"This is the average quad matrix {avg_quad_matrix} for path {individual_path}")
        #print(type(avg_quad_matrix))
        if not np.any(avg_quad_matrix):

            print(f"[SKIPPING] completely zero matrix, no need to run. This is the job {individual_path}")
            continue

        eigvalues, eigvecs, D_clean = diagonalize_matrix(avg_quad_matrix.copy())

        #print("These are the:")
        #print("The eigvecs: ", eigvecs)
        eigvecs = eigvecs.T # For writing to file simplicty
        instance.quadropole_tensor = avg_quad_matrix.tolist()
        instance.diagonalized_quadropole_tensor = D_clean.tolist()

        instance.write_metadata_to_json(new_dir)
        instance.update_input_file('target_list', value='cfavg')
        matrix_turned_into_string = format_polarization_block(eigvecs)
        #print("This is what is being turned into a string to be put into the input file ", matrix_turned_into_string)
        instance.update_input_file('polarization', value=matrix_turned_into_string)
        instance.write_corvus_in_file(individual_path)

        json_dict = {
            'Determined Charges': {Element(el).Z: oxi for el, oxi in oxidation_dict.items()},
            'Avg Quadrupole Matrix': instance.quadropole_tensor,
            'Avg Diagonalized Quadrupole matrix': instance.diagonalized_quadropole_tensor
            }            
        

        with open(Path(f"{individual_path}/{individual_path.name}.json"), 'w') as f:
            #print("This is me trying to see where this individual loop is writing the file to ", Path(f"{individual_path}/{individual_path.name}.json"))
            json.dump(json_dict, f, indent=4)

        good_calculation_list.append(f"{individual_path}/{individual_path.name}.in")

    print("I SHOULD NOW BE DONE WRITING ALL OF THE JSON FILES.")
    good_input_list_file = qsub.save_input_file_list(good_calculation_list, "corvus_good_file_list.txt")
    script_path = qsub.write_corvus_array_script(good_input_list_file)
    qsub.submit_corvus_job_array(good_input_list_file, script_path)

#    for instance in all_calculation_instances:
        #initialize JSON dict here
        #individual_path = Path(calc_directory / instance.cif_file.stem / f"{instance.cif_file.stem}_{ABSORBING_ATOM}")

#this is how the program is called 'python full_polarization.py [name of target directory of cifs]'
if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: python Full_Polarization.py <target_directory> <absorbing_atom>")
        sys.exit(1)
    
    main(TARGET_DIRECTORY, ABSORBING_ATOM)



import helpers
import json
import pprint
import os
import sys
import numpy as np
from datetime import datetime 
from pymatgen.core import Structure

import importlib
importlib.reload(helpers) #Reload helpers if necessary

def factor_dictionary(structure, mp_id, central_atom, crystal_NN=False, cluster_radius=0, cif_name = 'cif_name'):
    """
    Generate a dictionary containing various properties and factors for a material cluster extracted 
    from a CIF file. This function computes various descriptors such as Steinhart parameters, quadrupole 
    moments, and material properties, and returns them in a dictionary format.
    """
   
    # Get cluster name from the CIF file
    cluster_name = structure.composition.reduced_formula
    print(f"Extracted cluster name: {cluster_name}")
    

    # Get Materials Project ID (MP-ID)
    try:
        print(f"MP-ID for {cluster_name}: {mp_id}")
    except KeyError:
        print(f"MP-ID for {cluster_name} not found.")
        return None

    # Add space group analysis
    print(f"Analyzing space group for structure with MP-ID {mp_id}")
    try:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        sga = SpacegroupAnalyzer(structure)
        space_group = sga.get_space_group_symbol()
        space_group_number = sga.get_space_group_number()
        print(f"Space group: {space_group} (Number: {space_group_number})")
    except Exception as e:
        space_group = "Unknown"
        space_group_number = -1
        print(f"Error analyzing space group: {e}")

    # Choose how to extract the cluster
    if not crystal_NN:
        print(f"Extracting cluster data from structure within a radius of {cluster_radius} Angstroms")
        try:
            cluster_data = helpers.extract_cluster_structure(structure, central_atom, cluster_radius)
            coords, atomic_symbols, atomic_numbers = cluster_data
            neighbors = []
        except Exception as e:
            print(f"Error extracting cluster data: {e} using distance from central atom.")
            return None
    else:
        print(f"Extracting cluster data from structure with Crystal NN")
        try:
            cluster_data = helpers.crystalnn_extract_cluster_structure(structure, central_atom)
            coords, atomic_symbols, atomic_numbers, neighbors = cluster_data
            center_site = structure[0]
        except Exception as e:
            print(f"Error extracting cluster data: {e} using crystal_NN.")
            return None

    # Log the central atom and initial atomic data 
    print(f"Central atom: {central_atom}")
    print(f"Number of atoms in the cluster: {len(coords)}")

    print("\nSymbol   Atomic Number   x-coord   y-coord   z-coord")
    print("=" * 50)

    for symbol, atomic_number, coord in zip(atomic_symbols, atomic_numbers, coords):
        x, y, z = coord
        print(f"{symbol:<8}{atomic_number:<15}{x:<10.4f}{y:<10.4f}{z:<10.4f}")

    # Remove the central atom from the cluster
    coords = coords[1:]
    atomic_symbols = atomic_symbols[1:]
    atomic_numbers = atomic_numbers[1:]

    
    print("\nRemoving cetral atom from cluster:")
    

    factor_dict = {}
    factor_dict['MP-ID'] = mp_id
    factor_dict['CIF Name'] = cif_name
    factor_dict['Material'] = cluster_name
    factor_dict['Space Group'] = space_group
    factor_dict['Space Group Number'] = space_group_number
    
    # Retrieve material properties and add them to the dictionary
    print(f"\nRetrieving material properties for MP-ID {mp_id}")
    try:
        properties = helpers.get_cluster_properties(mp_id)
        factor_dict.update(properties)
        print("Material properties retrieved successfully.")
    except Exception as e:
        print(f"Error retrieving material properties: {e}")
        return None
    
    if neighbors != []:
        print(f"Computing number of ligands")
        try:
            number_of_unique_ligands = helpers.compute_number_of_unique_ligands(neighbors)
        except Exception as e:
            print(f"Error computing number of ligands: {e}")
            return None

        print(f"Computing average bond length")
        try:
            average_bond_length = helpers.compute_average_bond_distance(neighbors, center_site)
        except Exception as e:
            print(f"Error computing average bond length: {e}")
            return None
        
        print(f"Computing bond length std")
        try:
            bond_length_std = helpers.compute_bond_length_std(neighbors, center_site)
        except Exception as e:
            print(f"Error computing std of bond length: {e}")
            return None
        
        print(f'Computing average and standard deviation of electronegativity')
        try:
            avg_en, std_en = helpers.compute_electronegativity_stats(neighbors)
        except Exception as e:
            print(f"Error computing average and std of electronegativity: {e}")
            return None    

        factor_dict["Number of Unique Ligands"] = number_of_unique_ligands
        factor_dict['Average Bond Length'] = average_bond_length
        factor_dict['Bond Length Std'] = bond_length_std
        factor_dict["Average Electronegativity"] = avg_en
        factor_dict["Electronegativity Std"]= std_en
    
    else:
        factor_dict["Number of Unique Ligands"] = "Could not compute"
        factor_dict['Average Bond Length'] = "Could not compute"
        factor_dict['Bond Length Std'] = "Could not compute"
        factor_dict["Average Electronegativity"] = "Could not compute"
        factor_dict["Electronegativity Std"]= "Could not compute"


    # Get charges based on possible species
    oxidation_states = helpers.get_oxidation_state_formula(cluster_name)
    factor_dict["oxidation_states"] = oxidation_states

    if oxidation_states == "Oxidation states could not be determined.":  # Check if the oxidation states could be found
        print(f"Could not get oxidation states for the cluster. Setting normalized dipole and quadrupole moment and corresponding calculations to 'Could not compute'.")
        factor_dict["dipole moment normalized"] = "Could not compute"
        factor_dict["normalized dipole anisotropy matrix sum"] = "Could not compute"
        factor_dict["quadrupole moment normalized"] = "Could not compute"
        factor_dict["normalized quadrupole anisotropy matrix sum"] = "Could not compute"
    
    else:
        print(f'Oxidation states: {oxidation_states}')

        print(oxidation_states)

        charges = helpers.get_charges(atomic_symbols, oxidation_states)

        # Compute the normalized dipole moment
        print("Computing normalized dipole moment")
        try:
            dipole_moment_normalized = helpers.dipole_moment_normalized(coords, charges)
            factor_dict["dipole moment normalized"] = dipole_moment_normalized
            print("Normalized dipole moment calculated successfully.")
        except Exception as e:
            print(f"Error computing normalized dipole moment: {e}")
            return None
        
        # Compute the normalized dipole moment anisotropy matrix
        print("Computing normalized dipole anisotropy matrix")
        try:

            normalized_dipole_anisotropy_matrix = helpers.dipole_anisotropy_matrix(dipole_moment_normalized)

            factor_dict["normalized dipole anisotropy matrix"] = normalized_dipole_anisotropy_matrix
            print("Normalized dipole anisotropy matrix calculated successfully.")
        except Exception as e:
            print(f"Error computing normalized dipole anisotropy matrix: {e}")
            return None
        
        # Compute the normalized dipole moment anisotropy matrix sum
        print("Computing normalized dipole anisotropy matrix sum")
        try:
            normalized_dipole_anisotropy_matrix_sum = helpers.p_anisotropy_matrix_sum(normalized_dipole_anisotropy_matrix)

            factor_dict["normalized dipole anisotropy matrix sum"] = normalized_dipole_anisotropy_matrix_sum
            print("Computing normalized dipole anisotropy matrix sum calculated successfully.")
        except Exception as e:
            print(f"Error computing normalized dipole anisotropy matrix sum: {e}")
            return None

        # Compute the normalized quadrupole moment
        print("Computing normalized quadrupole moment")
        try:
            quad_moment_normalized = helpers.quadrupole_moment_normalized(coords, charges)
            factor_dict["quadrupole moment normalized"] = quad_moment_normalized
            print("Normalized quadrupole moment calculated successfully.")
        except Exception as e:
            print(f"Error computing normalized quadrupole moment: {e}")
            return None
        
        # Compute the normalized quadrupole moment anisotropy matrix
        print("Computing normalized quadrupole anisotropy matrix")
        try:
            #Get diagonal components of the normalized quadrupole moment matrx ie Q00,Q11,Q22
            qxx = quad_moment_normalized[0,0]
            qyy = quad_moment_normalized[1,1]
            qzz = quad_moment_normalized[2,2]

            normalized_quadrupole_anisotropy_matrix = helpers.quadrupole_anisotropy_matrix(qxx,qyy,qzz)

            factor_dict["normalized quadrupole anisotropy matrix"] = normalized_quadrupole_anisotropy_matrix
            print("Normalized quadrupole anisotropy matrix calculated successfully.")
        except Exception as e:
            print(f"Error computing normalized quadrupole anisotropy matrix: {e}")
            return None
        
        # Compute the normalized quadrupole moment anisotropy matrix sum
        print("Computing normalized quadrupole anisotropy matrix sum")
        try:
            normalized_quadrupole_anisotropy_matrix_sum = helpers.q_anisotropy_matrix_sum(normalized_quadrupole_anisotropy_matrix)

            factor_dict["normalized quadrupole anisotropy matrix sum"] = normalized_quadrupole_anisotropy_matrix_sum
            print("Computing normalized quadrupole anisotropy matrix sum calculated successfully.")
        except Exception as e:
            print(f"Error computing normalized quadrupole anisotropy matrix sum: {e}")
            return None


    sys.stdout.flush()  # Flush to ensure all output is printed
    return factor_dict


def write_factor_dictionary_to_file(factor_dict, filename):

    """
    Writes the factor dictionary to a JSON file.

    Args:
    - factor_dict: Dictionary containing data (including numpy arrays).
    - filename: The name of the file to write the dictionary to.
    """
    print(f"Started writing dictionary to {filename}")
    # Convert the dictionary to a JSON-serializable format
    serializable_dict = helpers.convert_to_json_serializable(factor_dict)

    with open(filename, "w") as fp:
        json.dump(serializable_dict, fp, indent=4)  # Use JSON to serialize the dictionary
        fp.flush()  # Ensure the buffer is flushed
        os.fsync(fp.fileno())  # Ensure file is written to disk
    print(f"Done writing dict to {filename}")


#Add conditional statement to check if cof file is found if not mark a failure

def process_folder_of_cifs(
    cif_folder,
    output_folder,
    crystal_NN=False,
    cluster_radius=0,
    central_atom="",
    mp_id="",
    log_message=None,
):
    """
    Processes all CIF files in (and below) the given folder structure, detects
    the transition metal, calculates the factor dictionary for each, and writes
    the results to separate files in the output folder.

    Args:
        cif_folder (str): Path to the folder (possibly containing nested subfolders) with CIFs.
        output_folder (str): Path to the folder where the output dictionaries will go.
        crystal_NN (bool): Whether to use CrystalNN to extract neighbors.
        cluster_radius (float): The radius (in angstroms) to consider around the central atom.
        central_atom (str): If not empty, forces the same central atom for all CIFs.
        mp_id (str): If not empty, forces the same mp_id for all CIFs.
        log_message (str): A custom message to include at the beginning of the process log.

    Returns:
        None
    """
    # Ensure a unique output folder exists
    output_folder = helpers.get_unique_output_folder(output_folder)

    # Create a subfolder for logs inside the output folder
    logs_folder = os.path.join(output_folder, "logs")
    os.makedirs(logs_folder, exist_ok=True)  # Create the logs folder if it doesn't exist

    # Get time
    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Set up the log file
    log_path = os.path.join(logs_folder, "process_log.txt")
    
    with open(log_path, "w") as log_file:
        # Add the custom log message if provided
        if log_message:
            log_file.write(f"{log_message}\n\n")
        # Write the start of the log
        log_file.write(f"Beginning Factor Dictionary Calculations at {time_str}\n")
        log_file.write("=" * 50 + "\n")

    # Redirect stdout to both log file and terminal
    with open(log_path, "a") as log_file:
        sys.stdout = helpers.DualWriter(log_file)

        try:
            print("Starting to Compute Factor Dictionary\n")
            successful_runs = 0
            incomplete_runs = 0
            failed_runs = 0
            index = 1
            incomplete_index = []
            failed_index = []

            # -----------------------------------------------------------
            # Recursively gather all .cif files from cif_folder
            # -----------------------------------------------------------
            all_cif_paths = []
            for root, dirs, files in os.walk(cif_folder):
                for file in files:
                    if file.endswith(".cif"):
                        all_cif_paths.append(os.path.join(root, file))

            if not all_cif_paths:
                print(f"No CIF files found under {cif_folder}.\n")

            # -----------------------------------------------------------
            # Now process each .cif we found
            # -----------------------------------------------------------
            for cif_path in all_cif_paths:
                cif_file = os.path.basename(cif_path)
                
                cif_name = cif_name = os.path.splitext(cif_file)[0] #Grab just the name of the cif file

            # Only extract MP-ID from the filename if it's not provided in the function input
                if mp_id == "":
                    mp_id = os.path.splitext(cif_file)[0]
                    mp_reset = True #Bool var used to distinguish if mp id is given or not in function input

                print("\n\n----------------------------------------------------------")
                print(f"Index: {index}\n")
                print(f"Processing CIF file: {cif_path}")

                print(f"mp id {mp_id}")

                # Load the structure and detect the transition metal
                try:
                    structure = Structure.from_file(cif_path)
                    cluster_name = structure.composition.reduced_formula

                    # If central_atom is still blank, detect from structure
                    if central_atom == "":
                        central_atom = helpers.detect_3d_transition_metal(structure)

                    if not central_atom:
                        print(f"No transition metal found in {cif_file}. Skipping...")
                        continue

                    print(f"Detected transition metal {central_atom} in {cif_file}")

                except Exception as e:
                    print(f"Error detecting transition metal in {cif_file}: {e}")
                    failed_runs += 1
                    continue

                # Call your factor dictionary function
                factor_dict = factor_dictionary(
                    structure,
                    mp_id,
                    central_atom,
                    crystal_NN,
                    cluster_radius,
                    cif_name
                )

                print("\nFactor Dictionary")
                pprint.pprint(factor_dict)

                if factor_dict is not None:
                    # Write out the factor dictionary
                    output_filename = os.path.splitext(cif_file)[0] + "_factor_dict.json"
                    output_path = os.path.join(output_folder, output_filename)
                    write_factor_dictionary_to_file(factor_dict, output_path)

                    # Check for incomplete conditions
                    if factor_dict["oxidation_states"] == "Oxidation states could not be determined.":
                        print(f"Completed processing CIF file: {cif_file}, but oxidation state could not be found")
                        incomplete_runs += 1
                        incomplete_index.append(("No oxidation states", "Index:", index, mp_id, cluster_name))
                    else:
                        successful_runs += 1
                else:
                    print(f"Failed to process {cif_file}")
                    failed_runs += 1
                    failed_index.append(("None", "Index:", index, mp_id, cluster_name))

                index += 1

                #Reset mp-id after run
                if mp_reset == True:
                    mp_id = ""

            # Summary of runs
            print(f"\nSummary:")
            print(f"Success: {successful_runs}")
            print(f"Incomplete: {incomplete_runs}")
            print(f"Failures: {failed_runs}")

            if incomplete_index:
                print(f"Incomplete {incomplete_index}")
            if failed_index:
                print(f"Failure {failed_index}")

        finally:
            sys.stdout = sys.__stdout__  # Restore original stdout

Cr_log_message = """Date: 2/21/2025, Extraction: CNN, Central Atom Cr. First calculation with bond length num of ligands and electronegativity."""
Cu_log_message = """Date: 2/10/2025, Extraction: CNN, Central Atom Cu. First calculation with Space group numbers."""
Fe_log_message = """Date: 2/15/2025, Extraction: Radial 10 A, Central Atom Fe."""


process_folder_of_cifs("Cr_Data_dir", "Cr_data/Cr_fd_2_21_2025", central_atom ="Cr", crystal_NN=True, log_message = Cr_log_message)
#process_folder_of_cifs("Cu_Data_dir", "Cu_data/Cu_fd_2_10_2025", central_atom ="Cu", crystal_NN=True, log_message = Cu_log_message)
#process_folder_of_cifs("Fe_Data_dir", "Fe_data/Fe_fd_2_10_2025", central_atom ="Fe", crystal_NN=True, log_message = Fe_log_message)

#process_folder_of_cifs("Cr_Data_dir", "Cr_data/Cr_fd_2_9_2025", central_atom ="Cr", crystal_NN=True, log_message = Cr_log_message)
#process_folder_of_cifs("Cu_Data_dir", "Cu_data/Cu_fd_2_10_2025", central_atom ="Cu", crystal_NN=True, log_message = Cu_log_message)
#process_folder_of_cifs("Fe_Data_dir", "Fe_data/Fe_fd_2_15_2025", central_atom ="Fe", crystal_NN=False, cluster_radius=10, log_message = Fe_log_message)

#process_folder_of_cifs("Practice_Cif", "Practice_Cif/Practice_Cif_fd", crystal_NN=False, cluster_radius=10)
#process_folder_of_cifs("Practice_Cif", "Practice_Cif/Practice_Cif_fd", crystal_NN=True)
#process_folder_of_cifs(cif_folder = "Practice_fd_runs/NiO_stretched_structures", output_folder = "Practice_fd_runs/NiO_stretched_structures_fd_2182025", crystal_NN=True, central_atom = "Ni", mp_id = "mp-19009")
#process_folder_of_cifs(cif_folder = "Practice_fd_runs/Cr2O3_stretched_structures", output_folder = "Practice_fd_runs/Cr203_stretched_structures_fd_2212025", crystal_NN=True, central_atom = "Cr", mp_id = "mp-19399")

#Set up process folder so that I can add a message to the top of the process log describing the run
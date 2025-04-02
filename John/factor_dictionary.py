from datetime import datetime 

load_up_start = datetime.now()

import helpers
import json
import pprint
import os
import sys
import numpy as np
from pymatgen.core import Structure

import importlib
importlib.reload(helpers) #Reload helpers if necessary

#Define Global Variables

DEGREE_L = 10 #Defined for steinhart parameters

#Specify the range of exponents for the distance norm in the ST QM and DM
NORM_EXP_LOW = 0
NORM_EXP_HIGH = 14

CHARGES = "uniform" #Set to "uniform" to make all charges equal to 1

load_up_end = datetime.now()

total_load_time = load_up_end - load_up_start #Gets the time needed to load all the dependencies 

def factor_dictionary(structure, mp_id, central_atom, crystal_NN=False, cluster_radius=0, cif_name = 'cif_name'):
    """
    Generate a dictionary of physical descriptors and chemical properties for a local cluster 
    around a central atom in a crystal structure.

    This function performs cluster extraction (using either distance-based or CrystalNN methods),
    computes oxidation states, space group information, local bonding environment features,
    and a wide range of descriptors including Steinhart vectors, dipole moments, and quadrupole moments
    over a customizable range of 1/r^n normalizations. The output dictionary is designed for machine
    learning or analysis pipelines involving material property prediction.

    Parameters
    ----------
    structure : pymatgen.core.structure.Structure
        The input crystal structure from which the cluster will be extracted.

    mp_id : str
        Materials Project ID corresponding to the crystal structure.

    central_atom : str
        The chemical symbol of the central atom to serve as the cluster center.

    crystal_NN : bool, optional
        If True, use CrystalNN to determine the bonded cluster. If False, extract
        a spherical cluster based on radial distance. Default is False.

    cluster_radius : float, optional
        If `crystal_NN=False`, this defines the radial cutoff in angstroms to select
        surrounding atoms around the central atom. Default is 0.

    cif_name : str, optional
        The name of the input CIF file for labeling purposes. Default is 'cif_name'.

    Returns
    -------
    factor_dict : dict or None
        A dictionary containing computed descriptors and material metadata. Returns None if
        oxidation states are not determined or if any descriptor calculation fails.

    Notes
    -----
    - If oxidation states cannot be determined for the cluster, the function exits early with None.
    - The cluster is pruned to exclude the central atom before descriptor computation.
    - Structural and chemical descriptors include:
        - Space group number and symbol
        - Bond lengths, bond angles, and ligand counts
        - Electronegativity statistics
    - Steinhart vectors and sums are computed for `norm_exp` ranging from NORM_EXP_LOW to NORM_EXP_HIGH.
    - Dipole and quadrupole moments and their anisotropy matrices are computed using 1/r^n decay factors.
    - All data is flushed to stdout after processing for real-time logging.
    """
    factor_dict = {}

    
    # Get cluster name from the CIF file
    cluster_name = structure.composition.reduced_formula
    print(f"Extracted cluster name: {cluster_name}")


    # Get charges based on possible species
    oxidation_states = helpers.get_oxidation_state_formula(cluster_name)
    factor_dict["oxidation states"] = oxidation_states

    # ------------------------------------------------------------------------
    #
    # Skip the cluster if the oxidation state cannot be found for that cluster
    #
    # ------------------------------------------------------------------------

    if oxidation_states == "Oxidation states could not be determined.":  # Check if the oxidation states could be found
        
        print(f"Could not get oxidation states for the cluster. Skipping cluster.... ")#Skip the cluster if there isnt an oxidation state
        return None  
    
    else:
        print(f'Oxidation states: {oxidation_states}')

        print(oxidation_states)

        #Get Coords and compute the whole factor dict
        # Get Materials Project ID (MP-ID)
    try:
        print(f"MP-ID for {cluster_name}: {mp_id}")
    except KeyError:
        print(f"MP-ID for {cluster_name} not found.")
        return None

    print("Adding mp-id, cif name and chem formula to factor dictionary")

    factor_dict['mp-id'] = mp_id
    factor_dict['cif name'] = cif_name
    factor_dict['chem formula'] = cluster_name

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

    print("Adding space group and space group number to factor dictionary")

    factor_dict['space group'] = space_group
    factor_dict['space group number'] = space_group_number

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
        
    print("\nRemoving cetral atom from cluster:")

    # Remove the central atom from the cluster
    coords = coords[1:]
    atomic_symbols = atomic_symbols[1:]
    atomic_numbers = atomic_numbers[1:]


    print(f"Adding atomic coordinates to the factor dictionary")
    # Combine coordinates and atomic symbols in a structured format
    factor_dict['atoms'] = [
        {'symbol': symbol, 'coords': coord}
        for symbol, coord in zip(atomic_symbols, coords)
    ]
    
    # Retrieve material properties and add them to the dictionary
    print(f"\nRetrieving material properties for MP-ID {mp_id}")
    try:
        print(mp_id)
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
        
        print(f'Computing average and standard deviation of bond angle')
        try:
            avg_bond_angle, bond_angle_std = helpers.bond_angle_statistics(center_site, neighbors)
        except Exception as e:
            print(f"Error computing average and std bond angle: {e}")
            return None
        
        print(f'Computing average and standard deviation of electronegativity')
        try:
            avg_en, std_en = helpers.compute_electronegativity_stats(neighbors)
        except Exception as e:
            print(f"Error computing average and std of electronegativity: {e}")
            return None

        factor_dict["number of unique ligands"] = number_of_unique_ligands
        factor_dict['average bond length'] = average_bond_length
        factor_dict['bond length std'] = bond_length_std
        factor_dict['average bond angle'] = avg_bond_angle
        factor_dict['bond angle std'] = bond_angle_std
        factor_dict["average electronegativity"] = avg_en
        factor_dict["electronegativity std"]= std_en
    
    else:

        print(f"Extraction method cluster of radius {cluster_radius} could not compute number of ligands setting to None")
        try:
            number_of_unique_ligands = None
        except Exception as e:
            print(f"Error computing number of ligands: {e}")
            return None

        print(f'Extraction method cluster of radius {cluster_radius} could not compute avg and std of bond length setting to None')
        try:
            average_bond_length, std_bond_length = None, None
        except Exception as e:
            print(f"Error computing bond length stats: {e}")
            return None
        
        print(f'Extraction method cluster of radius {cluster_radius} could not compute avg and std of bond angle setting to None')
        try:
            avg_bond_angle = None
            bond_angle_std = None
        except Exception as e:
            print(f"Error computing average and std bond angle: {e}")
            return None
        
        print(f'Computing average and standard deviation of electronegativity for cluster')
        try:
            avg_en, std_en = helpers.compute_electronegativity_stats_cluster(atomic_symbols)
        except Exception as e:
            print(f"Error computing average and std of electronegativity: {e}")
            return None

        factor_dict["number of unique ligands"] = number_of_unique_ligands
        factor_dict['average bond length'] = average_bond_length
        factor_dict['bond length std'] = std_bond_length
        factor_dict['average bond angle'] = avg_bond_angle
        factor_dict['bond angle std'] = bond_angle_std
        factor_dict["average electronegativity"] = avg_en
        factor_dict["electronegativity std"]= std_en

    # Convert Cartesian coordinates to spherical coordinates
    print("Converting Cartesian coordinates to spherical coordinates to calculate steinhart parameters")
    spherical_coords = helpers.cartesian_to_spherical(coords)

    for st_exp in range(NORM_EXP_LOW,NORM_EXP_HIGH):

        # Compute the Steinhart vector
        print(f"Computing Steinhart vector (l={DEGREE_L}) for the cluster with normalization 1/r^{st_exp}")
        try:
            steinhart_vector = helpers.compute_steinhart_vector(spherical_coords, atomic_numbers, DEGREE_L, cluster_name, norm_exp=st_exp)
            factor_dict[f"steinhart vector 1/r^{st_exp}"] = steinhart_vector
            print(f"Steinhart vector 1/r^{st_exp} normalization  computed successfully.")
        except Exception as e:
            print(f"Error computing Steinhart vector: {e}")
            return None

        # Compute the Steinhart parameter sum
        print(f"Computing Steinhart parameter sum normalization 1/r^{st_exp}")
        try:
            steinhart_param_sum = helpers.calculate_steinhart_sum(spherical_coords, atomic_numbers, DEGREE_L, st_exp)
            factor_dict[f"steinhart parameter sum 1/r^{st_exp}"] = steinhart_param_sum
            print(f"Steinhart parameter sum norm 1/r^{st_exp}: {steinhart_param_sum}")
        except Exception as e:
            print(f"Error computing Steinhart parameter sum: {e}")
            return None

        if CHARGES == "uniform":
            print("----------------- Using uniform charges -----------------")
            charges = np.ones(len(coords))
        else:
            charges = helpers.get_charges(atomic_symbols, oxidation_states)
            print("Using charges from oxidation state")
        
        #Range of exponents to
        for qm_exp in range(NORM_EXP_LOW, NORM_EXP_HIGH):
            for dm_exp in range(qm_exp - 2, qm_exp + 1):

                # Compute the normalized dipole moment
                print(f"Computing normalized dipole moment with 1/r^{dm_exp}")
                try:
                    dipole_moment_normalized = helpers.dipole_moment_normalized(coords, charges, dm_exp)
                    factor_dict[f"dipole moment normalized 1/r^{dm_exp}"] = dipole_moment_normalized
                    print(f"Normalized dipole moment 1/r^{dm_exp} calculated successfully.")
                except Exception as e:
                    print(f"Error computing normalized dipole moment 1/r^{dm_exp}: {e}")
                    return None
                
                # Compute the normalized dipole moment anisotropy matrix
                print(f"Computing normalized dipole anisotropy matrix 1/r^{dm_exp}")
                try:
                    normalized_dipole_anisotropy_matrix = helpers.dipole_anisotropy_matrix(dipole_moment_normalized)

                    factor_dict[f"normalized dipole anisotropy matrix 1/r^{dm_exp}"] = normalized_dipole_anisotropy_matrix
                    print(f"Normalized dipole anisotropy matrix 1/r^{dm_exp} calculated successfully.")
                except Exception as e:
                    print(f"Error computing normalized dipole anisotropy matrix 1/r^{dm_exp}: {e}")
                    return None
                
                # Compute the normalized dipole moment anisotropy matrix sum
                print(f"Computing normalized dipole anisotropy matrix sum 1/r^{dm_exp}")
                try:
                    normalized_dipole_anisotropy_matrix_sum = helpers.d_anisotropy_matrix_sum(normalized_dipole_anisotropy_matrix)

                    factor_dict[f"normalized dipole anisotropy matrix sum 1/r^{dm_exp}"] = normalized_dipole_anisotropy_matrix_sum
                    print(f"Computing normalized dipole anisotropy matrix sum 1/r^{dm_exp} calculated successfully.")
                except Exception as e:
                    print(f"Error computing normalized dipole anisotropy matrix sum 1/r^{dm_exp}: {e}")
                    return None

                # Compute the normalized quadrupole moment
                print(f"Computing normalized quadrupole moment with 1/r^{qm_exp}")
                try:
                    quad_moment_normalized = helpers.quadrupole_moment_normalized(coords, charges, qm_exp)
                    factor_dict[f"quadrupole moment normalized 1/r^{qm_exp}"] = quad_moment_normalized
                    print("Normalized quadrupole moment 1/r^{qm_exp} calculated successfully.")
                except Exception as e:
                    print(f"Error computing normalized quadrupole moment 1/r^{qm_exp}: {e}")
                    return None
                
                # Compute the normalized quadrupole moment anisotropy matrix
                print(f"Computing normalized quadrupole anisotropy matrix 1/r^{qm_exp}")
                try:
                    #Get diagonal components of the normalized quadrupole moment matrx ie Q00,Q11,Q22
                    qxx = quad_moment_normalized[0,0]
                    qyy = quad_moment_normalized[1,1]
                    qzz = quad_moment_normalized[2,2]

                    normalized_quadrupole_anisotropy_matrix = helpers.quadrupole_anisotropy_matrix(qxx,qyy,qzz)

                    factor_dict[f"normalized quadrupole anisotropy matrix 1/r^{qm_exp}"] = normalized_quadrupole_anisotropy_matrix
                    print(f"Normalized quadrupole anisotropy matrix 1/r^{qm_exp} calculated successfully.")
                except Exception as e:
                    print(f"Error computing normalized quadrupole anisotropy matrix 1/r^{qm_exp}: {e}")
                    return None
                
                # Compute the normalized quadrupole moment anisotropy matrix sum
                print(f"Computing normalized quadrupole anisotropy matrix sum 1/r^{qm_exp}")
                try:
                    normalized_quadrupole_anisotropy_matrix_sum = helpers.q_anisotropy_matrix_sum(normalized_quadrupole_anisotropy_matrix)

                    factor_dict[f"normalized quadrupole anisotropy matrix sum 1/r^{qm_exp}"] = normalized_quadrupole_anisotropy_matrix_sum
                    print(f"Computing normalized quadrupole anisotropy matrix sum 1/r^{qm_exp} calculated successfully.")
                except Exception as e:
                    print(f"Error computing normalized quadrupole anisotropy matrix sum 1/r^{qm_exp}: {e}")
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

                # Call factor dictionary function
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
                    if factor_dict["oxidation states"] == "Oxidation states could not be determined.":
                        print(f"Completed processing CIF file: {cif_file}, but oxidation state could not be found")
                        incomplete_runs += 1
                        incomplete_index.append(("No oxidation states", "Index:", index, mp_id, cluster_name))
                    elif np.isnan(factor_dict[f'steinhart parameter sum 1/r^{6}']): #Arbitrary norm if one is nan they are all nan
                        print(f"Completed processing CIF file: {cif_file}, but steinhart_parameter_sum is NaN")
                        incomplete_runs += 1
                        incomplete_index.append(("Steinhart is NaN", f"Index: {index}", mp_id, cluster_name))
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

            # Log the total computation time
            end_time = datetime.now()
            total_time = end_time - datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S') + total_load_time

            # Format the total time nicely
            hours, remainder = divmod(total_time.total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)

            print(f"Total computation time: {hours}h {minutes}m {seconds:.2f}s")

        finally:
            sys.stdout = sys.__stdout__  # Restore original stdout



Cr_log_message = """Date: 2/24/2025, Extraction: Cluster 5 A, Central Atom Cr. Trying to extract larger cluster because NN is not getting all atoms for every cluster QM normalized by 1/r^3 and DM by 1/r^2"""
Cr_oxides_log_message = """Date: 3/29/2025, Extraction: Cluster 6 A, Central Atom Cr. Running the materials charles sent after he ran his own calculations on Cr Oxide spectra"""
Cu_log_message = """Date: 2/24/2025, Extraction: CNN, Central Atom Cr. Re added steinhart parameters added chemical info and refactored the nameing convention for the dictionarys"""
Fe_log_message = """Date: 2/24/2025, Extraction: CNN, Central Atom Cr. Re added steinhart parameters added chemical info and refactored the nameing convention for the dictionarys"""


#process_folder_of_cifs("Cr_Data_dir", "Cr_data/Cr_fd_2_24_2025", central_atom ="Cr", crystal_NN=True, log_message = Cr_log_message)
#process_folder_of_cifs("Cr_oxide_Data_dir", "Cr_oxide_data/Cr_oxide_fd_3_08_2025", central_atom ="Cr", crystal_NN=True, log_message = Cr_oxides_message)
#process_folder_of_cifs("Cu_Data_dir", "Cu_data/Cu_fd_2_24_2025", central_atom ="Cu", crystal_NN=True, log_message = Cu_log_message)
#process_folder_of_cifs("Fe_Data_dir", "Fe_data/Fe_fd_2_24_2025", central_atom ="Fe", crystal_NN=True, log_message = Fe_log_message)

#process_folder_of_cifs("Cr_Data_dir", "Cr_data/Cr_fd_2_9_2025", central_atom ="Cr", crystal_NN=True, log_message = Cr_log_message)
#process_folder_of_cifs("Cu_Data_dir", "Cu_data/Cu_fd_2_10_2025", central_atom ="Cu", crystal_NN=True, log_message = Cu_log_message)
#process_folder_of_cifs("Fe_Data_dir", "Fe_data/Fe_fd_2_15_2025", central_atom ="Fe", crystal_NN=False, cluster_radius=10, log_message = Fe_log_message)

#process_folder_of_cifs("Practice_Cif", "Practice_Cif/Practice_Cif_fd", crystal_NN=False, cluster_radius=6, qm_exponent=14 )
process_folder_of_cifs("Practice_Cif", "Practice_Cif/Practice_Cif_fd", crystal_NN=True)
#process_folder_of_cifs(cif_folder = "Practice_fd_runs/NiO_stretched_structures", output_folder = "Practice_fd_runs/NiO_stretched_structures_fd_2182025", crystal_NN=True, central_atom = "Ni", mp_id = "mp-19009")
#process_folder_of_cifs(cif_folder = "Practice_fd_runs/Cr2O3_stretched_structures", output_folder = "Practice_fd_runs/Cr203_stretched_structures_fd_2212025", crystal_NN=True, central_atom = "Cr", mp_id = "mp-19399")


#process_folder_of_cifs("Cr_oxide_Data_dir_Charles",f"Cr_oxide_data/Cr_oxide_fd_3_29_2025", central_atom ="Cr", crystal_NN=False, cluster_radius=6 , log_message = Cr_oxides_log_message)
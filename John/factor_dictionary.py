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

def factor_dictionary(structure, mp_id, degree_l, central_atom, crystal_NN=False, cluster_radius=0):
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

    # Choose how to extract the cluster
    if not crystal_NN:
        print(f"Extracting cluster data from structure within a radius of {cluster_radius} Angstroms")
        try:
            cluster_data = helpers.extract_cluster_structure(structure, central_atom, cluster_radius)
            coords, atomic_symbols, atomic_numbers = cluster_data
        except Exception as e:
            print(f"Error extracting cluster data: {e} using distance from central atom.")
            return None
    else:
        print(f"Extracting cluster data from structure with Crystal NN")
        try:
            cluster_data = helpers.crystalnn_extract_cluster_structure(structure, central_atom)
            coords, atomic_symbols, atomic_numbers = cluster_data
        except Exception as e:
            print(f"Error extracting cluster data: {e} using crystal_NN.")
            return None

    # Log the central atom and initial atomic data
    center_atom = atomic_symbols[0]
    print(f"Central atom: {center_atom}")
    print(f"Number of atoms in the cluster: {len(coords)}")


    # Remove the central atom from the cluster
    coords = coords[1:]
    atomic_symbols = atomic_symbols[1:]
    atomic_numbers = atomic_numbers[1:]
    
    print("Cluster after removing central atom:")
    print(f"Atomic symbols: {atomic_symbols}")
    print(f"Atomic numbers: {atomic_numbers}")
    
    # Convert Cartesian coordinates to spherical coordinates
    print("Converting Cartesian coordinates to spherical coordinates")
    spherical_coords = helpers.cartesian_to_spherical(coords)
    
    factor_dict = {}
    factor_dict['Material'] = cluster_name

    # Compute the Steinhart vector
    print(f"Computing Steinhart vector (l={degree_l}) for the cluster")
    try:
        steinhart_vector = helpers.compute_steinhart_vector(spherical_coords, atomic_numbers, degree_l, cluster_name)
        factor_dict["steinhart_vector"] = steinhart_vector
        print("Steinhart vector computed successfully.")
    except Exception as e:
        print(f"Error computing Steinhart vector: {e}")
        return None

    # Compute the Steinhart parameter sum
    print("Computing Steinhart parameter sum")
    try:
        steinhart_param_sum = helpers.calculate_steinhart_sum(spherical_coords, atomic_numbers, degree_l)
        factor_dict["steinhart_parameter_sum"] = steinhart_param_sum
        print(f"Steinhart parameter sum: {steinhart_param_sum}")
    except Exception as e:
        print(f"Error computing Steinhart parameter sum: {e}")
        return None

    # Retrieve material properties and add them to the dictionary
    print(f"Retrieving material properties for MP-ID {mp_id}")
    try:
        properties = helpers.get_cluster_properties(mp_id, center_atom)
        factor_dict.update(properties)
        print("Material properties retrieved successfully.")
    except Exception as e:
        print(f"Error retrieving material properties: {e}")
        return None

    # Compute charges based on possible species
    possible_species = factor_dict.get('possible_species', [])

    if not possible_species:  # Check if the possible_species list is empty
        print(f"No possible species found. Setting quadrupole moment to 'Could not compute'.")
        factor_dict["quadrupole moment"] = "Could not compute"
        factor_dict["quadrupole moment normalized"] = "Could not compute"
    else:
        print(f"Computing charges for possible species: {possible_species}")
        try:
            charges = helpers.get_charges(possible_species, atomic_symbols)
            print(f"Charges computed: {charges}")
        except Exception as e:
            print(f"Error computing charges: {e}")
            return None

        # Compute the quadrupole moment
        print("Computing quadrupole moment for the cluster")
        try:
            quad_moment = helpers.quadrupole_moment(coords, charges)
            factor_dict["quadrupole moment"] = quad_moment
            print("Quadrupole moment calculated successfully.")
        except Exception as e:
            print(f"Error computing quadrupole moment: {e}")
            return None

        # Compute the normalized quadrupole moment
        print("Computing normalized quadrupole moment")
        try:
            quad_moment_normalized = helpers.quadrupole_moment_normalized(coords, charges, atomic_numbers)
            factor_dict["quadrupole moment normalized"] = quad_moment_normalized
            print("Normalized quadrupole moment calculated successfully.")
        except Exception as e:
            print(f"Error computing normalized quadrupole moment: {e}")
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

def process_folder_of_cifs(cif_folder, output_folder, degree_l, crystal_NN=False, cluster_radius=0):
    """
    Processes all CIF files in the given folder structure, detects the transition metal, calculates the factor dictionary for each,
    and writes the resulting dictionary to a separate file in the output folder.

    Args:
        cif_folder (str): Path to the folder containing subfolders with CIF files.
        output_folder (str): Path to the folder where the output dictionaries will be written.
        degree_l (int): Degree of the spherical harmonics used in the Steinhart calculation.
        crystal_NN (bool): Whether to use CrystalNN to extract neighbors instead of a fixed radius.
        cluster_radius (float): The radius (in angstroms) to consider around the central atom.

    Returns:
        None
    """
    # Ensure a unique output folder exists
    output_folder = helpers.get_unique_output_folder(output_folder)

    #Get time
    time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Set up the log file
    log_path = os.path.join(output_folder, "process_log.txt")
    with open(log_path, "w") as log_file:
        sys.stdout = helpers.DualWriter(log_file)  # Redirect stdout to the log file and terminal uses custom class in helpers

        try:
            # Get all subfolders in the directory
            subfolders = [os.path.join(cif_folder, d) for d in os.listdir(cif_folder) if os.path.isdir(os.path.join(cif_folder, d))]

            print(f"Beginning Factor Dictionary Calculations at {time}")
            successful_runs = 0
            incomplete_runs = 0
            failed_runs = 0
            index = 1
            incomplete_index = []
            failed_index = []
            
            # Loop over each subfolder
            for subfolder in subfolders:
                # Find a CIF file in the subfolder
                cif_files = [f for f in os.listdir(subfolder) if f.endswith(".cif")]
                
                if not cif_files:
                    print(f"No CIF file found in {subfolder}. Skipping...")
                    incomplete_runs += 1
                    incomplete_index.append(("No CIF found in folder", f"Index: {index}", os.path.basename(subfolder)))
                    index += 1
                    continue
                
                # Assume each subfolder contains only one CIF file
                cif_file = cif_files[0]
                cif_path = os.path.join(subfolder, cif_file)

                # Extract mp_id from the filename (e.g., "mp-2515" from "mp-2515.cif")
                mp_id = os.path.splitext(cif_file)[0]
                
                print("\n\n----------------------------------------------------------")
                print(f"Index: {index}\n")
                print(f"Processing CIF file: {cif_path}")

                # Load the structure and detect the transition metal
                try:
                    structure = Structure.from_file(cif_path)
                    cluster_name = structure.composition.reduced_formula
                    central_atom = helpers.detect_3d_transition_metal(structure)
                    if not central_atom:
                        print(f"No transition metal found in {cif_file}. Skipping...")
                        continue
                    print(f"Detected transition metal {central_atom} in {cif_file}")
                except Exception as e:
                    print(f"Error detecting transition metal in {cif_file}: {e}")
                    failed_runs += 1
                    continue
                
                # Call the factor dictionary function
                factor_dict = factor_dictionary(structure, mp_id, degree_l, central_atom, crystal_NN, cluster_radius)

                print("Factor Dictionary")
                pprint.pprint(factor_dict)

                if factor_dict is not None:
                    # Write the factor dictionary to file
                    output_filename = os.path.splitext(cif_file)[0] + "_factor_dict.json"
                    output_path = os.path.join(output_folder, output_filename)
                    write_factor_dictionary_to_file(factor_dict, output_path)
                    
                    if factor_dict['possible_species'] == []:
                        print(f"Completed processing CIF file: {cif_file}, but possible species could not be found")
                        incomplete_runs += 1
                        incomplete_index.append(("No possible species", "Index: ", index, mp_id, cluster_name))
                    elif np.isnan(factor_dict['steinhart_parameter_sum']):
                        print(f"Completed processing CIF file: {cif_file}, but possible species could not be found")
                        incomplete_runs += 1
                        incomplete_index.append(("Steinhart is NaN", f"Index: {index} ", mp_id, cluster_name))
                    else:
                        successful_runs += 1
                else:
                    print(f"Failed to process {cif_file}")
                    failed_runs += 1
                    failed_index.append(("None", "Index: ", index, mp_id, cluster_name))

                index += 1

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
    
#process_folder_of_cifs("Practice_Cif", "Practice_Cif/Factor_Dictionary", 10, crystal_NN=True)
process_folder_of_cifs("Cr_Data_dir", "Factor_Dictionary/Cr_Factor_Dictionary", 10, crystal_NN=True)
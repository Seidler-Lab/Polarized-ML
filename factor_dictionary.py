import helpers
import pickle
import pprint
import os
import sys
import numpy as np
from pymatgen.core import Structure

# Define globally so dict is only created once
mp_ids = helpers.read_mp_id_file("Cif/Materials2/mp_id.txt")


def factor_dictionary(structure, degree_l, central_atom, crystal_NN=False, cluster_radius=0):
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
        mp_id = mp_ids[cluster_name]
        print(f"MP-ID for {cluster_name}: {mp_id}")
    except KeyError:
        print(f"MP-ID for {cluster_name} not found.")
        return None

    # Choose how to extract the cluster
    if not crystal_NN:
        print(f"Extracting cluster data from structure within a radius of {cluster_radius} Angstroms")
        try:
            cluster_data = helpers.extract_cluster(structure, central_atom, cluster_radius)
            coords, atomic_symbols, atomic_numbers = cluster_data
        except Exception as e:
            print(f"Error extracting cluster data: {e} using distance from central atom.")
            return None
    else:
        print(f"Extracting cluster data from structure with Crystal NN")
        try:
            cluster_data = helpers.crystalnn_extract_cluster(structure, central_atom)
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
    Writes the factor dictionary to a file using pickle for serialization.

    Args:
    - factor_dict: Dictionary containing data (including numpy arrays).
    - file_path: The name of the file to write the dictionary to.
    """
    print(f"Started writing dictionary to {filename}")
    with open(filename, "wb") as fp:
        pickle.dump(factor_dict, fp)  # Use pickle to serialize the dictionary
        fp.flush()  # Ensure the buffer is flushed
        os.fsync(fp.fileno())  # Ensure file is written to disk
    print(f"Done writing dict to {filename}")


def process_folder_of_cifs(cif_folder, output_folder, degree_l, crystal_NN=False, cluster_radius=0):
    """
    Processes all CIF files in the given folder, detects the transition metal, calculates the factor dictionary for each,
    and writes the resulting dictionary to a separate file in the output folder.

    Args:
        cif_folder (str): Path to the folder containing the CIF files.
        output_folder (str): Path to the folder where the output dictionaries will be written.
        degree_l (int): Degree of the spherical harmonics used in the Steinhart calculation.
        crystal_NN (bool): Whether to use CrystalNN to extract neighbors instead of a fixed radius.
        cluster_radius (float): The radius (in angstroms) to consider around the central atom.

    Returns:
        None
    """
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get all CIF files from the directory
    cif_files = [f for f in os.listdir(cif_folder) if f.endswith(".cif")]

    print("Beginning Factor Dictionary Calculations")
    sys.stdout.flush()

    successful_runs = 0
    incomplete_runs = 0
    failed_runs = 0
    index = 1
    incomplete_index = []
    failed_index = []

    # Loop over each CIF file
    for cif_file in cif_files:
        cif_path = os.path.join(cif_folder, cif_file)

        print("\n\n----------------------------------------------------------")

        print(f"Index: {index}\n")
        sys.stdout.flush()  # Ensure buffer flushes immediately

        print(f"Processing CIF file: {cif_path}")

        # Load the structure and detect the transition metal
        try:
            structure = Structure.from_file(cif_path)
            central_atom = helpers.detect_transition_metal(structure)
            if not central_atom:
                print(f"No transition metal found in {cif_file}. Skipping...")
                continue
            print(f"Detected transition metal {central_atom} in {cif_file}")
        except Exception as e:
            print(f"Error detecting transition metal in {cif_file}: {e}")
            failed_runs += 1
            continue

        # Call the factor dictionary function
        factor_dict = factor_dictionary(structure, degree_l, central_atom, crystal_NN, cluster_radius)

        print("Factor Dictionary")
        pprint.pprint(factor_dict)
        sys.stdout.flush()  # Ensure pprint is flushed immediately

        if factor_dict is not None:
            # Write the factor dictionary to file
            output_filename = os.path.splitext(cif_file)[0] + "_factor_dict.pkl"
            output_path = os.path.join(output_folder, output_filename)
            write_factor_dictionary_to_file(factor_dict, output_path)
            if factor_dict['possible_species'] == []:
                print(f"Completed processing CIF file: {cif_file}, but possible species could not be found")
                sys.stdout.flush()  # Flush at the end of each loop
                incomplete_runs += 1
                incomplete_index.append(("No possible species" , index))
            elif np.isnan(factor_dict['steinhart_parameter_sum']) :
                print(f"Completed processing CIF file: {cif_file}, but possible species could not be found")
                sys.stdout.flush()  # Flush at the end of each loop
                incomplete_runs += 1
                incomplete_index.append(("Steinhart is NaN", index))
            else:
                successful_runs += 1
        else:
            print(f"Failed to process {cif_file}")
            failed_runs += 1
            failed_index.append(("None", index))

        index += 1

    # Summary of runs
    print(f"\nSummary:")
    print(f"Success: {successful_runs}")
    print(f"Incomplete: {incomplete_runs}")
    print(f"Failures: {failed_runs}")

    if incomplete_index != []:
        print(f"Incomplete at indicies {incomplete_index}")
    if failed_index != []:
        print(f"Failure at indicies {failed_index}")
    sys.stdout.flush()  # Final flush to ensure all output is printed
    
process_folder_of_cifs("Cif/Materials2", "Cif/Materials2/Factor_Dictionary", 10, crystal_NN=True)
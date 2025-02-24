from pymatgen.core import Structure
import numpy as np
import numpy.linalg as linalg
from scipy.special import sph_harm
import os
import sys
from mp_api.client import MPRester
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core import Element
from pymatgen.core.periodic_table import Element
import pyscal.core as pc
import json
from pymatgen.core import Composition
from pathlib import Path
import pandas as pd

parity = True  # Include parity or not for the plot
degree_l = 40

api_key = "M244rOwcXhVorQLQwwH6s2GXVO88BCIJ"

# List of 3d transition metals based on their atomic numbers
transition_metals_3d = [Element(sym).symbol for sym in [
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"
]]

def read_xyz_file(file_path):
    """
    Read atomic coordinates from an XYZ file.

    Parameters:
    file_path (str): The path to the XYZ file to be read.

    Returns:
    np.ndarray: A NumPy array of atomic coordinates with shape (n, 3), where n is the number of atoms in the molecule. Each row corresponds to the x, y, and z coordinates of an atom.
    """

    with open(file_path, 'r') as xyz_file:
        # Skip the first two lines (metadata)
        lines = xyz_file.readlines()[2:]

    # Extract atomic symbols (not used here but might be useful for other purposes)
    atomic_symbols = np.array([line.split()[0] for line in lines])

    # Extract and store atomic coordinates
    atomic_coordinates = np.array([line.split()[1:4]
                                  for line in lines], dtype=float)

    return atomic_coordinates, atomic_symbols


def extract_cluster_cif_index(cif_file, index_number, cluster_radius=3):
    """
    Extracts a cluster of atoms around a specified index position in a crystal structure from a CIF file.
    The central atom (specified by index_number) will be the first in the returned arrays.

    Parameters:
    - cif_file (str): Path to the CIF file containing the crystal structure.
    - index_number (int): Index of the atom to be used as the center of the cluster.
    - cluster_radius (float): Radius of the cluster in angstroms within which atoms will be selected.

    Returns:
    - coords (np.ndarray): A 2D array of shape (n, 3) where 'n' is the number of atoms in the cluster.
      Each row contains the x, y, z coordinates of the corresponding atom.
    - atomic_symbols (np.ndarray): A 1D array containing the atomic symbols of the atoms in the cluster,
      with the central atom's symbol first.
    - atomic_numbers (np.ndarray): A 1D array containing the atomic numbers (Z) of the atoms in the cluster,
      with the central atom's atomic number first.
    """

    # Load the crystal structure from the provided CIF file
    structure = Structure.from_file(cif_file)

    # Find all sites (atoms) within the specified radius around the atom at 'index_number'
    sites = structure.get_sites_in_sphere(structure[index_number].coords, cluster_radius)

    # Create a new structure consisting of only the atoms within the cluster
    cluster_structure = Structure.from_sites(sites)

    # Initialize lists to hold the coordinates, atomic symbols, and atomic numbers
    coords = []
    atomic_symbols = []
    atomic_numbers = []

    # Reverse the structure so that the central atom is the first one in the final list
    for site in reversed(cluster_structure):
        coords.append(site.coords)
        atomic_symbols.append(site.specie.symbol)
        atomic_numbers.append(site.specie.Z)

    # Convert the lists to numpy arrays and center the central atom at (0,0,0)
    coords = np.array(translate_coords(coords))
    atomic_symbols = np.array(atomic_symbols)
    atomic_numbers = np.array(atomic_numbers)

    return coords, atomic_symbols, atomic_numbers


def extract_cluster_cif_symbol(cif_file, atomic_symbol, cluster_radius=3):
    """
    Extracts a cluster of atoms around the first occurrence of the specified atomic symbol in a crystal structure from a CIF file.
    The central atom (specified by atomic_symbol) will be the first in the returned arrays.
    """

    structure = Structure.from_file(cif_file)

    # Find the first atom matching the atomic symbol
    for i, site in enumerate(structure):
        if site.specie.symbol == atomic_symbol:
            chosen_atom_index = i
            break
    else:
        raise ValueError(f"No atoms with symbol '{atomic_symbol}' found in structure.")

    # Find all sites within the specified radius around the chosen atom
    sites = structure.get_sites_in_sphere(structure[chosen_atom_index].coords, cluster_radius)

    # Create a new structure consisting of only the atoms within the cluster
    cluster_structure = Structure.from_sites(sites)

    coords = []
    atomic_symbols = []
    atomic_numbers = []

    for site in reversed(cluster_structure):
        coords.append(site.coords)
        atomic_symbols.append(site.specie.symbol)
        atomic_numbers.append(site.specie.Z)

    coords = np.array(translate_coords(coords))
    atomic_symbols = np.array(atomic_symbols)
    atomic_numbers = np.array(atomic_numbers)

    return coords, atomic_symbols, atomic_numbers


def extract_cluster_structure(structure, atomic_symbol, cluster_radius=3):
    """
    Extracts a cluster of atoms around the first occurrence of the specified atomic symbol in a crystal structure.
    Ensures the specified atomic symbol is treated as the central atom and appears first in the returned arrays.
    """
    # Find the first atom matching the atomic symbol
    central_atom_index = None
    for i, site in enumerate(structure):
        if site.specie.symbol == atomic_symbol:
            central_atom_index = i
            break
    if central_atom_index is None:
        raise ValueError(f"No atoms with symbol '{atomic_symbol}' found in structure.")

    # Get the coordinates of the central atom
    central_coords = structure[central_atom_index].coords

    # Find all sites within the specified radius around the central atom
    sites = structure.get_sites_in_sphere(central_coords, cluster_radius)

    # Explicitly ensure the central atom is added first
    central_site = structure[central_atom_index]
    sorted_sites = [central_site] + [
        site[0] for site in sites if not np.allclose(site[0].coords, central_coords)
    ]

    # Create the cluster structure from sorted sites
    cluster_structure = Structure.from_sites(sorted_sites)

    coords = []
    atomic_symbols = []
    atomic_numbers = []

    for site in cluster_structure:
        coords.append(site.coords)
        atomic_symbols.append(site.specie.symbol)
        atomic_numbers.append(site.specie.Z)

    # Translate coordinates to center the cluster
    coords = np.array(translate_coords(coords))
    atomic_symbols = np.array(atomic_symbols)
    atomic_numbers = np.array(atomic_numbers)

    return coords, atomic_symbols, atomic_numbers


def crystalnn_extract_cluster_cif(cif_file, atomic_symbol):
    """
    Extracts a cluster of atoms around the first occurrence of the specified atomic symbol in a crystal structure from a CIF file.
    The central atom (specified by atomic_symbol) will be the first in the returned arrays. The cluster is based on CrystalNN nearest neighbors.

    Parameters:
    - cif_file (str): Path to the CIF file containing the crystal structure.
    - atomic_symbol (str): Atomic symbol (e.g., "Si", "O") of the atom to center the cluster around.

    Returns:
    - coords (np.ndarray): A 2D array of shape (n, 3) where 'n' is the number of atoms in the cluster.
      Each row contains the x, y, z coordinates of the corresponding atom.
    - atomic_symbols (np.ndarray): A 1D array containing the atomic symbols (e.g., "H", "O") of the atoms in the cluster,
      with the central atom's symbol first.
    - atomic_numbers (np.ndarray): A 1D array containing the atomic numbers (Z) of the atoms in the cluster,
      with the central atom's atomic number first.
    """

    # Load the crystal structure from the provided CIF file
    structure = Structure.from_file(cif_file)

    # Initialize a SpacegroupAnalyzer to symmetrize the structure
    structure_analyzer = SpacegroupAnalyzer(structure)

    # Get the symmetrized structure
    symmetrized_structure = structure_analyzer.get_symmetrized_structure()

    # Find the first atom matching the atomic symbol
    for i, site in enumerate(symmetrized_structure):
        if site.specie.symbol == atomic_symbol:
            chosen_atom_index = i
            break
    else:
        raise ValueError(f"No atoms with symbol '{atomic_symbol}' found in structure.")

    # Initialize CrystalNN nearest neighbors calculator
    crystal_nn = CrystalNN()

    # Get the CrystalNN neighbors of the selected atom
    neighbors = crystal_nn.get_nn_info(symmetrized_structure, chosen_atom_index)

    # Include the central atom in the cluster
    cluster_sites = [structure[chosen_atom_index]]

    # Collect the neighbor sites
    for neighbor in neighbors:
        cluster_sites.append(neighbor['site_index'])

    # Initialize lists to hold the coordinates, atomic symbols, and atomic numbers
    coords = []
    atomic_symbols = []
    atomic_numbers = []

    # Append the central atom first, followed by its neighbors
    for site in cluster_sites:
        coords.append(site.coords)  # Append the coordinates of each atom
        atomic_symbols.append(site.specie.symbol)  # Append the atomic symbol
        atomic_numbers.append(site.specie.Z)  # Append the atomic number

    # Convert the lists to numpy arrays for easier manipulation and center the central atom at (0,0,0)
    coords = np.array(translate_coords(coords))
    atomic_symbols = np.array(atomic_symbols)
    atomic_numbers = np.array(atomic_numbers)

    # Return the coordinates, atomic symbols, and atomic numbers
    return coords, atomic_symbols, atomic_numbers, neighbors


def crystalnn_extract_cluster_structure(structure, atomic_symbol):
    """
    Extracts a cluster of atoms around the first occurrence of the specified atomic symbol in a crystal structure.
    The central atom (specified by atomic_symbol) will be the first in the returned arrays. The cluster is based on CrystalNN nearest neighbors.

    Parameters:
    - structure (Structure): Pymatgen Structure object of the crystal structure.
    - atomic_symbol (str): Atomic symbol (e.g., "Si", "O") of the atom to center the cluster around.

    Returns:
    - coords (np.ndarray): A 2D array of shape (n, 3) where 'n' is the number of atoms in the cluster.
      Each row contains the x, y, z coordinates of the corresponding atom.
    - atomic_symbols (np.ndarray): A 1D array containing the atomic symbols (e.g., "H", "O") of the atoms in the cluster,
      with the central atom's symbol first.
    - atomic_numbers (np.ndarray): A 1D array containing the atomic numbers (Z) of the atoms in the cluster,
      with the central atom's atomic number first.
    """

    # Find the first atom matching the atomic symbol in the original structure
    chosen_atom_index = None
    for i, site in enumerate(structure):
        if site.specie.symbol == atomic_symbol:
            chosen_atom_index = i
            break
    if chosen_atom_index is None:
        raise ValueError(f"No atoms with symbol '{atomic_symbol}' found in structure.")

    # Initialize CrystalNN nearest neighbors calculator
    crystal_nn = CrystalNN()

    # Get the CrystalNN neighbors of the selected atom
    neighbors = crystal_nn.get_nn_info(structure, chosen_atom_index)

    # Include the central atom in the cluster
    cluster_sites = [structure[chosen_atom_index]]

    # Collect the neighbor sites
    for neighbor in neighbors:
        cluster_sites.append(neighbor['site'])

    # Initialize lists to hold the coordinates, atomic symbols, and atomic numbers
    coords = [site.coords for site in cluster_sites]  # Collect coordinates
    atomic_symbols = [site.specie.symbol for site in cluster_sites]  # Collect atomic symbols
    atomic_numbers = [site.specie.Z for site in cluster_sites]  # Collect atomic numbers

    # Convert lists to numpy arrays for easier manipulation and center the central atom
    coords = np.array(translate_coords(coords))
    atomic_symbols = np.array(atomic_symbols)
    atomic_numbers = np.array(atomic_numbers)

    # Return the coordinates, atomic symbols, and atomic numbers
    return coords, atomic_symbols, atomic_numbers, neighbors


def voronoi_extract_cluster_cif(cif_file, atomic_symbol):
    """
    Extracts a cluster of atoms around the first occurrence of the specified atomic symbol in a crystal structure from a CIF file.
    The central atom (specified by atomic_symbol) will be the first in the returned arrays. The cluster is based on Voronoi neighbors.

    Parameters:
    - cif_file (str): Path to the CIF file containing the crystal structure.
    - atomic_symbol (str): Atomic symbol (e.g., "Si", "O") of the atom to center the cluster around.

    Returns:
    - coords (np.ndarray): A 2D array of shape (n, 3) where 'n' is the number of atoms in the cluster.
      Each row contains the x, y, z coordinates of the corresponding atom.
    - atomic_symbols (np.ndarray): A 1D array containing the atomic symbols of the atoms in the cluster,
      with the central atom's symbol first.
    - atomic_numbers (np.ndarray): A 1D array containing the atomic numbers (Z) of the atoms in the cluster,
      with the central atom's atomic number first.
    """
    
    # Load the crystal structure from the provided CIF file
    structure = Structure.from_file(cif_file)

    # Find the first atom matching the atomic symbol
    for i, site in enumerate(structure):
        if site.specie.symbol == atomic_symbol:
            chosen_atom_index = i
            break
    else:
        raise ValueError(f"No atoms with symbol '{atomic_symbol}' found in structure.")

    # Initialize Voronoi nearest neighbors calculator
    voronoi_nn = VoronoiNN()

    # Get the Voronoi neighbors of the selected atom
    neighbors = voronoi_nn.get_nn_info(structure, chosen_atom_index)

    # Include the central atom in the cluster
    cluster_sites = [structure[chosen_atom_index]]

    # Collect the neighbor sites
    for neighbor in neighbors:
        cluster_sites.append(structure[neighbor['site_index']])

    # Initialize lists to hold the coordinates, atomic symbols, and atomic numbers
    coords = []
    atomic_symbols = []
    atomic_numbers = []

    # Append the central atom first, followed by its neighbors
    for site in cluster_sites:
        coords.append(site.coords)
        atomic_symbols.append(site.specie.symbol)
        atomic_numbers.append(site.specie.Z)

    # Convert the lists to numpy arrays for easier manipulation and center the central atom at (0,0,0)
    coords = np.array(translate_coords(coords))
    atomic_symbols = np.array(atomic_symbols)
    atomic_numbers = np.array(atomic_numbers)

    return coords, atomic_symbols, atomic_numbers


def voronoi_extract_cluster_structure(structure, atomic_symbol):
    """
    Extracts a cluster of atoms around the first occurrence of the specified atomic symbol in a crystal structure.
    The central atom (specified by atomic_symbol) will be the first in the returned arrays. The cluster is based on Voronoi neighbors.

    Parameters:
    - structure (Structure): Pymatgen Structure object of the crystal.
    - atomic_symbol (str): Atomic symbol (e.g., "Si", "O") of the atom to center the cluster around.

    Returns:
    - coords (np.ndarray): A 2D array of shape (n, 3) where 'n' is the number of atoms in the cluster.
      Each row contains the x, y, z coordinates of the corresponding atom.
    - atomic_symbols (np.ndarray): A 1D array containing the atomic symbols of the atoms in the cluster,
      with the central atom's symbol first.
    - atomic_numbers (np.ndarray): A 1D array containing the atomic numbers (Z) of the atoms in the cluster,
      with the central atom's atomic number first.
    """

    # Find the first atom matching the atomic symbol
    for i, site in enumerate(structure):
        if site.specie.symbol == atomic_symbol:
            chosen_atom_index = i
            break
    else:
        raise ValueError(f"No atoms with symbol '{atomic_symbol}' found in structure.")

    # Initialize Voronoi nearest neighbors calculator
    voronoi_nn = VoronoiNN()

    # Get the Voronoi neighbors of the selected atom
    neighbors = voronoi_nn.get_nn_info(structure, chosen_atom_index)

    # Include the central atom in the cluster
    cluster_sites = [structure[chosen_atom_index]]

    # Collect the neighbor sites
    for neighbor in neighbors:
        cluster_sites.append(structure[neighbor['site_index']])

    coords = []
    atomic_symbols = []
    atomic_numbers = []

    for site in cluster_sites:
        coords.append(site.coords)
        atomic_symbols.append(site.specie.symbol)
        atomic_numbers.append(site.specie.Z)

    coords = np.array(translate_coords(coords))
    atomic_symbols = np.array(atomic_symbols)
    atomic_numbers = np.array(atomic_numbers)

    return coords, atomic_symbols, atomic_numbers


def translate_coords(coords):
    """
    Translates a list of 3D coordinates so that the first entry is at (0, 0, 0),
    and applies the same translation to all other coordinates.

    Args:
        coords (list of lists or numpy array): A list of coordinates, where each coordinate is a list or array of [x, y, z].

    Returns:
        translated_coords (numpy array): The translated coordinates with the first entry centered at (0, 0, 0).
    """
    # Convert coords to numpy array if not already
    coords = np.array(coords)

    # Take the first entry as the translation vector
    translation_vector = coords[0]

    # Print message for checking cluster
    print("Checking cluster...")

    # Check if the first entry is already at (0, 0, 0)
    if np.allclose(translation_vector, [0, 0, 0]):
        print("Cluster already centered.")
        return coords
    else:
        print("Centering cluster...")

    # Apply translation: subtract the translation vector from all coords
    translated_coords = coords - translation_vector

    return translated_coords


def detect_3d_transition_metal(structure):
    """
    Detect the first occurrence of a transition metal in the given structure.
    
    Args:
        structure (Structure): A pymatgen Structure object representing the crystal.
    
    Returns:
        str: The symbol of the transition metal, or None if no transition metal is found.
    """
    for site in structure:
        if site.specie.symbol in transition_metals_3d:
            return site.specie.symbol
    return None


def extract_cluster_test(cif_folder, mp_id_file):
    """
    Automatically extracts clusters around transition metals for all CIF files in a folder using CrystalNN.

    Args:
        cif_folder (str): Path to the folder containing CIF files.
        mp_id_file (str): Path to the text file containing compound names and their MP-IDs.

    Returns:
        None: Prints the results for each CIF file and its corresponding MP-ID.
    """
    
    # Call the read_mp_id_file function to load the MP-IDs
    mp_id_mapping = read_mp_id_file(mp_id_file)

    # Get a list of all .cif files in the specified directory
    cif_files = [f for f in os.listdir(cif_folder) if f.endswith(".cif")]

    # Loop through the CIF files and extract clusters using CrystalNN
    for cif_file in cif_files:
        compound_name = os.path.splitext(cif_file)[0]  # Extract compound name from file name
        file_path = os.path.join(cif_folder, cif_file)  # Construct full path to the CIF file
        
        mp_id = mp_id_mapping.get(compound_name, "Unknown MP-ID")  # Get MP-ID for the compound
        
        # Load the crystal structure from the CIF file
        structure = Structure.from_file(file_path)
        
        # Detect the transition metal in the structure
        transition_metal = detect_3d_transition_metal(structure)
        
        if transition_metal is None:
            print(f"No transition metal found in {cif_file}. Skipping...")
            continue

        print(f"Processing {cif_file} for transition metal {transition_metal} (MP-ID: {mp_id})...")
        
        # Call the extract_crystalnn_cluster function
        coords, symbols, numbers = crystalnn_extract_cluster_structure(file_path, transition_metal)
        
        # Print the extracted data
        print(f"Extracted cluster for {transition_metal} in {cif_file} (MP-ID: {mp_id}):")
        print("Atomic symbols:", symbols)
        print("Atomic numbers:", numbers)
        print("Coordinates:", coords)
        print("\n")


def cartesian_to_spherical(coords):
    """
    Convert Cartesian coordinates to spherical coordinates.

    Parameters:
    coords (np.ndarray): A NumPy array of shape (n, 3) containing n points in
                         Cartesian coordinates, where each row represents [x, y, z].

    Returns:
    np.ndarray: A NumPy array of shape (n, 3) containing n points in spherical
                coordinates, where each row represents [r, theta, phi].
                Theta and phi are in radians. Theta is polar angle phi is the azimuthal angle.
    """

    # Extract x, y, and z coordinates from the input array
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    
    # Compute the radial distance for each point
    r = np.sqrt(x**2 + y**2 + z**2)

    # Calculate theta(polar anlge), defaulting to 0 where r is 0 (which also covers x=y=0)
    theta = np.where(r > 0.0, np.arctan2(y, x), 0)

    # Calculate phi(azimuthal angle), handling division by zero by setting theta to 0 where z is 0
    phi = np.where(r > 0.0, np.arccos(z/r), 0)

    # Stack the computed spherical coordinates into a single array
    spherical_coords = np.vstack((r, theta, phi)).T

    return spherical_coords


def calculate_real_sph_harm(m, l, theta, phi):
    # Calculate the spherical harmonic Y_l^m
    Ylm = sph_harm(m, l, theta, phi)

    # Take the complex conjugate of Ylm
    Ylm_conjugate = np.conj(Ylm)

    # Compute the real part of the spherical harmonic according to the given piecewise function
    if m > 0:
        Ylm_real = ((-1)**m / np.sqrt(2.0)) * (Ylm + Ylm_conjugate)
    elif m == 0:
        Ylm_real = Ylm  # Ylm is real if m is 0
    else:  # m < 0
        Ylm_real = ((-1)**m / (1j*np.sqrt(2.0))) * (Ylm - Ylm_conjugate)

    return Ylm_real


def calculate_lbop_r(sph_coords, atomic_numbers, degree_l, order_m, parity=True):
    """
    Calculate the local bond order paramater for a set of spherical coordinates.

    This function computes the local bond order paramater, which is a measure used
    in the analysis of local atomic environments. It involves
    summing up spherical harmonics for a set of points described by spherical
    coordinates, relative to a central atom assumed to be at the origin.

    Parameters:
    - sph_coords (np.ndarray): An array of spherical coordinates for the neighbors,
      where each row represents a point with [r, theta, phi] format.
    - degree_l (int): The degree 'l' of the spherical harmonic, a non-negative integer.
    - order_m (int): The order 'm' of the spherical harmonic, where m is an integer
      such that -l <= m <= l.

    Returns:
    - float: The local bond order paramater calculated for the given spherical coordinates
      and spherical harmonic parameters. Weighted by a factor of 1/r^4
    """
    # Get number of nearest neighbors to central atom at (0,0,0)
    
    n_neighbors = sph_coords.shape[0]

    # Extract r from the spherical coordinates
    r = sph_coords[:, 0]

    # Extract theta(asimuthal angle) and phi(polar angle) from the spherical coordinates
    theta = sph_coords[:, 1]
    phi = sph_coords[:, 2]

    # Compute the spherical harmonics for each [r, theta, phi] pair and sum them up
    # Note: calculate_real_sph_harm expects phi first, then theta

    # Compute considering parity
    if parity == True:
        # Sum over all neighbors
        Ylm_sum = np.sum(sph_harm(order_m, degree_l, theta, phi)*(1/r**6)*1/atomic_numbers)

    # Compute without considering parity
    else:
        # Sum over all neighbors
        Ylm_sum = np.sum(
            np.abs(calculate_real_sph_harm(order_m, degree_l, theta, phi))*(1/r**6)*1/atomic_numbers)

    # Calculate the local bond order paramater
    local_bond_order_paramater = 1 / n_neighbors * Ylm_sum

    return local_bond_order_paramater


def calculate_steinhart(spherical_coords, atomic_numbers, degree_l):
    """
    Calculate the Steinhardt parameter (ql) for a given degree l using atomic information
    provided in spherical coordinates. This function computes ql by summing the squares of
    the local bond order parameters (q_lm) for each order m, from -l to l, and then normalizing
    the sum according to the specified degree l.

    Parameters:
    - spherical_coords (array-like): The spherical coordinates of atoms. This should be an array
      where each element represents the spherical coordinates (r, theta, phi) of each atom.
    - atomic_numbers (array-like): An array of atomic numbers corresponding to each atom represented
      in spherical_coords. This is used to differentiate between different types of atoms when calculating q_lm.
    - degree_l (int): The degree l which specifies the level of angular resolution in the calculation
      of the bond order parameters.

    Returns:
    - float: The calculated Steinhardt parameter ql for the provided degree l.
    """
    q_lm_squared_sum = 0  # Initialize the sum of q_lm values
    order_m = -degree_l  # Start with the lowest order m

    # Iterate over all m values from -l to l, inclusive
    while order_m <= degree_l:
        # Calculate the SP for each m and add it to the sum
        q_lm_squared_sum += np.abs((calculate_lbop_r(spherical_coords, atomic_numbers,
                                                     degree_l, order_m, parity)))**2
        order_m += 1  # Move to the next order m

    # Calculate the overall SP for degree l using the accumulated sum of q_lm values
    ql = np.sqrt((4 * np.pi) / (2 * degree_l + 1)
                 * q_lm_squared_sum)

    # Return the ql for given degree l
    return ql


def calculate_steinhart_sum_from_filepath(file_path, degree_l):
    """
    Calculates the sum of Steinhardt parameters (q_l) up to a given degree (l) for a cluster of atoms.

    This function reads atomic coordinates from a provided file, converts Cartesian coordinates to spherical coordinates,
    and then calculates the Steinhardt parameters (q_l) for each degree up to the specified degree_l. The Steinhardt
    parameters are a measure of the local structural order around an atom in a cluster and are used to characterize
    the local symmetry.

    Parameters:
    - file_path (str): The name of the file containing atomic coordinates. The file should have a specific format
                       where the first line indicates the number of atoms, the second line is a comment (ignored),
                       and subsequent lines contain atomic symbols followed by their x, y, z coordinates.
    - degree_l (int): The maximum degree (l) for which the Steinhardt parameters (q_l) will be calculated. This function
                      will calculate q_l for all degrees from 0 up to and including degree_l.

    Returns:
    - float: The sum of the calculated Steinhardt parameters (q_l) for each degree from 0 up to degree_l.
    """

 # Read atomic coordinates and symbols from the file
    cluster_data = extract_cluster_cif_index(file_path, 0, 5)

    # Get atomic coordinates and removes center atom, at [0,0,0]
    coords, atomic_symbols, atomic_numbers  = cluster_data
    coords = coords[1:]
    atomic_symbols = atomic_symbols[1:]

    # Convert Cartesian coordinates to spherical coordinates
    spherical_coords = cartesian_to_spherical(coords)
    
    # Sum over ql for each degree
    q_l_sum = 0

    while degree_l >= 0:

        q_lm_squarred_sum = 0  # Initialize the sum of q_lm values
        order_m = -degree_l  # Start with the lowest order m

        # Iterate over all m values from -l to l, inclusive
        while order_m <= degree_l:
            # Calculate the SP for each m and add it to the sum
            q_lm_squarred_sum += np.abs(calculate_lbop_r(spherical_coords, atomic_numbers,
                                                         degree_l, order_m, parity))**2
            order_m += 1  # Move to the next order m

        # Calculate the overall SP for degree l using the accumulated sum of q_lm values
        q_l = np.sqrt((4 * np.pi) / (2 * degree_l + 1)
                      * q_lm_squarred_sum)

        q_l_sum += q_l  # Add ql for current degree to the sum

        degree_l -= 1  # Decrease degree_l for the next iteration

    # Return the sum qls for given degree
    return q_l_sum


def calculate_steinhart_sum(spherical_coords, atomic_numbers, degree_l):
    """
    Calculates the sum of Steinhardt parameters (q_l) up to a given degree (l) for a cluster of atoms.

    This function calculates the Steinhardt parameters (q_l) for each degree up to the specified degree_l
    using spherical coordinates and atomic numbers provided as input. The Steinhardt parameters are a measure
    of the local structural order around an atom in a cluster and are used to characterize the local symmetry.

    Parameters:
    - spherical_coords (np.ndarray): Array of spherical coordinates with shape (n, 3) for each atom in the cluster.
    - atomic_numbers (np.ndarray or list): Array or list of atomic numbers corresponding to each atom.
    - degree_l (int): The maximum degree (l) for which the Steinhardt parameters (q_l) will be calculated.
                      This function will calculate q_l for all degrees from 0 up to and including degree_l.

    Returns:
    - float: The sum of the calculated Steinhardt parameters (q_l) for each degree from 0 up to degree_l.
    """

    # Sum over ql for each degree
    q_l_sum = 0

    while degree_l >= 0:
        q_lm_squarred_sum = 0  # Initialize the sum of q_lm values
        order_m = -degree_l  # Start with the lowest order m

        # Iterate over all m values from -l to l, inclusive
        while order_m <= degree_l:
            # Calculate the SP for each m and add it to the sum
            q_lm_squarred_sum += np.abs(calculate_lbop_r(spherical_coords, atomic_numbers,
                                                         degree_l, order_m, parity))**2
            order_m += 1  # Move to the next order m

        # Calculate the overall SP for degree l using the accumulated sum of q_lm values
        q_l = np.sqrt((4 * np.pi) / (2 * degree_l + 1) * q_lm_squarred_sum)

        q_l_sum += q_l  # Add q_l for the current degree to the sum

        degree_l -= 1  # Decrease degree_l for the next iteration

    # Return the sum of q_l for the given degree
    return q_l_sum


def compute_steinhart_vector_from_filepath(file_path, degree_l):
    """
    Compute the Steinhardt parameters for all degrees from 0 up to the specified degree_l
    based on atomic coordinates and types read from an XYZ file. The function calculates
    the Steinhardt parameters for each degree using spherical coordinates.

    Parameters:
    - file_path (str): The path to the XYZ file containing atomic coordinates and element symbols.
      The format of the XYZ file should be such that the first line optionally contains the number
      of atoms, the second line is ignored (or can contain a comment), and subsequent lines should
      contain atomic symbols followed by x, y, z coordinates.
    - degree_l (int): The highest degree (l) of Steinhardt parameters to compute.

    Returns:
    - list: A list of Steinhardt parameters ql for each degree from 0 to degree_l.

    Note:
    - This function assumes the availability of helper functions `read_xyz_file`, `cartesian_to_spherical`,
      and `get_atomic_numbers` which are used to process the file data and convert coordinates.
    - The `calculate_steinhart` function is also required and must be implemented to compute individual
      ql values based on spherical coordinates and atomic numbers.
    """
    
    # Get atomic coordinates and removes center atom, at [0,0,0]
    cluster_data = extract_cluster_cif_index(file_path, 0, 5)
    coords, atomic_symbols, atomic_numbers  = cluster_data

    coords = coords[1:]  # Exclude the center atom's coordinates

    spherical_coords = cartesian_to_spherical(
        coords)  # Convert coordinates
    
    # Get atomic symbols (excluding the central atom)
    atomic_symbols = atomic_symbols[1:]  # Exclude the center atom's symbol

    cluster_name = extract_filename(file_path)

    ql_list = []
    for i in range(degree_l + 1):  # Compute ql for each degree from 0 to degree_l
        ql = calculate_steinhart(spherical_coords, atomic_numbers, i)
        ql_list.append(ql)

    return ql_list, cluster_name


def compute_steinhart_vector_from_structure(structure, degree_l, central_atom):
    """
    Compute the Steinhardt parameters for all degrees from 0 up to the specified degree_l
    based on atomic coordinates and types extracted from a pymatgen.Structure object.
    The function calculates the Steinhardt parameters for each degree using spherical coordinates.

    Parameters:
    - structure (pymatgen.Structure): The structure object containing atomic coordinates and element symbols.
    - degree_l (int): The highest degree (l) of Steinhardt parameters to compute.
    - central_atom_index (int): The index of the central atom for the cluster (default is 0).
    - neighbor_cutoff (float): The cutoff distance or number of neighbors for the cluster extraction.

    Returns:
    - tuple: A tuple containing:
      - list: A list of Steinhardt parameters ql for each degree from 0 to degree_l.
      - str: A name for the cluster extracted from the structure.

    Note:
    - This function assumes the availability of helper functions `helpers.crystalnn_extract_cluster_structure`,
      `cartesian_to_spherical`, and `calculate_steinhart`.
    """

    # Extract the cluster using the central atom and neighbor cutoff
    cluster_data = crystalnn_extract_cluster_structure(structure, central_atom)
    coords, atomic_symbols, atomic_numbers = cluster_data

    # Exclude the center atom's coordinates and symbols
    coords = coords[1:]  # Skip the central atom at [0, 0, 0]
    atomic_symbols = atomic_symbols[1:]  # Skip the central atom's symbol
    atomic_numbers = atomic_numbers[1:]  # Skip the central atom's atomic number

    # Convert coordinates to spherical coordinates
    spherical_coords = cartesian_to_spherical(coords)

    # Generate a cluster name from the structure
    cluster_name = f"Cluster from {structure.composition}"

    # Compute Steinhardt parameters
    ql_list = []
    for l in range(degree_l + 1):  # Compute ql for each degree from 0 to degree_l
        ql = calculate_steinhart(spherical_coords, atomic_numbers, l)
        ql_list.append(ql)

    return ql_list, cluster_name


def compute_steinhart_vector(spherical_coords, atomic_numbers, degree_l, cluster_name="Cluster"):
    """
    Compute the Steinhardt parameters for all degrees from 0 up to the specified degree_l
    based on atomic coordinates and types, assuming the coordinates are given in spherical form
    and atomic numbers are provided.

    Parameters:
    - spherical_coords (np.ndarray): Array of atomic spherical coordinates with shape (n, 3), excluding the central atom.
    - atomic_numbers (list or np.ndarray): List or array of atomic numbers corresponding to the atoms, excluding the central atom.
    - degree_l (int): The highest degree (l) of Steinhardt parameters to compute.
    - cluster_name (str): Name of the cluster being processed.

    Returns:
    - list: A list of Steinhardt parameters ql for each degree from 0 to degree_l.
    - str: The name of the cluster (same as input).

    Note:
    - This function assumes that `calculate_steinhart` is available and used to compute the individual
      ql values based on spherical coordinates and atomic numbers.
    """

    ql_list = []
    for i in range(degree_l + 1):  # Compute ql for each degree from 0 to degree_l
        ql = calculate_steinhart(spherical_coords, atomic_numbers, i)
        ql_list.append(ql)

    return ql_list, cluster_name


def compute_steinhart_vector_from_pyscal(cluster_data, central_atom_number):
    
    #Unpack custer data
    coords, atomic_symbols, atomic_numbers = cluster_data

    # Convert symbols to atomic numbers
    atomic_numbers = [Element(symbol).number for symbol in atomic_symbols]

    # Prepare data for pyscal
    atoms_data = []
    for i, (coord, symbol, number) in enumerate(zip(coords, atomic_symbols, atomic_numbers)):
        atoms_data.append({"id": i + 1, "pos": coord, "type": symbol})


    # Create pyscal System
    system = pc.System()

    # Set the simulation box for the cluster (3x3 matrix format)
    x_min, x_max = min(c[0] for c in coords), max(c[0] for c in coords)
    y_min, y_max = min(c[1] for c in coords), max(c[1] for c in coords)
    z_min, z_max = min(c[2] for c in coords), max(c[2] for c in coords)
    padding = 5  # Add some padding to avoid boundary issues
    system.box = [
        [x_max - x_min + 2 * padding, 0, 0],  # x-dimension
        [0, y_max - y_min + 2 * padding, 0],  # y-dimension
        [0, 0, z_max - z_min + 2 * padding],  # z-dimension
    ]


    # Create atoms with atomic numbers
    atoms = []
    for coord, atomic_number in zip(coords, atomic_numbers):
        atom = pc.Atom(pos=coord, type=atomic_number)  # Use atomic number as 'type'
        atoms.append(atom)

    # Add atoms to the system
    system.add_atoms(atoms)

    # Perform neighbor finding
    system.find_neighbors(method="cutoff", cutoff=3.0)

    # Loop through all atoms to inspect their neighbors
    for i, atom in enumerate(system.atoms):
        neighbors = atom.neighbors  # Get neighbors of the current atom (as indices)
        print(f"Atom {i+1} (Type: {atom.type}) has {len(neighbors)} neighbors:")
        for neighbor_index in neighbors:
            neighbor_atom = system.atoms[neighbor_index]  # Get the neighbor atom using its index
            print(f"    Neighbor Index: {neighbor_index+1}, Type: {neighbor_atom.type}, Position: {neighbor_atom.pos}")
            
    # Calculate bond order parameters for l = 2 to 12
    system.calculate_q(range(2, 13))  # Calculates q2 through q12

    # Get q values for all atoms for l = 2 to 12
    qvals = system.get_qvals(range(2, 13))

    # Extract q values only for atoms of central atom
    ni_qvals = [qvals[i] for i, atom in enumerate(system.atoms) if atom.type == central_atom_number] 

    return ni_qvals


def compute_minkowski_strucutre_metric(degree_l, relative_areas, spherical_coords):
    """
    Computes the q'_l(a) value based on the relative areas and spherical coordinates with (r, theta, phi).

    Parameters:
        l (int): The degree of the spherical harmonics.
        relative_areas (list of float): The relative areas A(f)/A for each facet.
        spherical_coords (list of tuples): Spherical coordinates (r, theta, phi) for each facet.

    Returns:
        float: The computed q'_l(a) value.
    """
    ql_squared = 0.0

    # Loop over m from -l to +l
    for degree_m in range(-degree_l, degree_l + 1):
        # Compute the summation over all facets
        sum_facets = 0.0
        for i, (r, theta, phi) in enumerate(spherical_coords):
            # Get the relative area for this facet
            relative_area = relative_areas[i]

            # Compute the spherical harmonic Y_lm(theta, phi)
            Ylm = calculate_real_sph_harm(degree_m, degree_l, theta, phi)

            # Accumulate the weighted contribution
            sum_facets += relative_area * Ylm

        # Add the square of the modulus of the sum to ql_squared
        ql_squared += np.abs(sum_facets) ** 2

    # Apply the normalization factor
    q_prime_l = np.sqrt((4 * np.pi) / (2 * degree_l + 1) * ql_squared)

    return q_prime_l


def compute_minkowski_structure_vector(max_l, relative_areas, spherical_coords):
    """
    Computes a vector of q'_l(a) values for degrees l = 0, 1, ..., max_l.

    Parameters:
        max_l (int): The maximum degree of the spherical harmonics (inclusive).
        relative_areas (list of float): The relative areas A(f)/A for each facet.
        spherical_coords (list of tuples): Spherical coordinates (r, theta, phi) for each facet.

    Returns:
        list: A vector where each entry corresponds to q'_l(a) for degrees l = 0 to max_l.
    """
    q_prime_vector = []
    
    # Loop over all degrees l from 0 to max_l
    for degree_l in range(max_l + 1):
        # Compute q'_l(a) for the current degree l
        q_prime_l = compute_minkowski_strucutre_metric(degree_l, relative_areas, spherical_coords)
        q_prime_vector.append(q_prime_l)
    
    return q_prime_vector


def compute_voronoi_metrics(structure, site_index, spherical_coords, neighbor_indices):
    """
    Computes Voronoi-based metrics (relative areas and facet angles) using precomputed spherical coordinates.

    Distribute Facet Areas:
    If a facet is shared among multiple neighbors, the area is distributed equally.

    Handle Missing Neighbors:
    Neighbors without a corresponding facet are assigned a small default area.

    Normalization:
    Ensures ∑A(f)/A=1, even if some neighbors are missing or grouped.

    Parameters:
        structure (pymatgen.Structure): The atomic structure.
        site_index (int): Index of the site in the structure for which to compute the metrics.
        spherical_coords (list of tuples): Precomputed spherical coordinates (r, theta, phi) for each neighbor.
        neighbor_indices (list of int): Indices of neighbors corresponding to the spherical coordinates.

    Returns:
        tuple: A tuple containing:
            - relative_areas (list of float): Relative areas A(f)/A for each neighbor.
            - facet_angles (list of tuples): Spherical angles (theta, phi) for each Voronoi facet normal vector.
    """
    from collections import defaultdict

    # Initialize VoronoiNN for Voronoi tessellation
    voronoi_nn = VoronoiNN()
    voronoi_polyhedron = voronoi_nn.get_voronoi_polyhedra(structure, site_index)

    # Dictionary to map neighbors to shared facet areas
    neighbor_facet_areas = defaultdict(float)

    # Step 1: Distribute facet areas among neighbors
    for neighbor_index, properties in voronoi_polyhedron.items():
        if neighbor_index in neighbor_indices:
            # Distribute facet area equally among all matching neighbors
            neighbor_facet_areas[neighbor_index] += properties["area"]

    # Step 2: Initialize relative areas and facet angles
    relative_areas = []
    facet_angles = []
    total_area = 0.0

    for i, neighbor_index in enumerate(neighbor_indices):
        # Get facet area or assign a small default value if the neighbor is missing
        facet_area = neighbor_facet_areas.get(neighbor_index, 0.0)

        # Use precomputed spherical coordinates
        r, theta, phi = spherical_coords[i]
        facet_angles.append((theta, phi))

        # Add facet area to the list and accumulate the total area
        relative_areas.append(facet_area)
        total_area += facet_area

    # Step 3: Normalize relative areas
    if total_area > 0:
        relative_areas = [area / total_area for area in relative_areas]
    else:
        # Assign equal areas if no valid facets were found
        relative_areas = [1.0 / len(neighbor_indices)] * len(neighbor_indices)

    return relative_areas, facet_angles
    

def site_index_by_symbol(structure, symbol):
    """
    Get the site index of the first occurrence of a specified atom by its chemical symbol.

    Parameters:
        structure (pymatgen.Structure): The atomic structure.
        symbol (str): The chemical symbol of the atom to search for (e.g., "Na").

    Returns:
        int: The index of the first site with the specified symbol, or -1 if not found.
    """
    for site_index, site in enumerate(structure):
        if symbol in site.species_string:
            return site_index  # Return the first matching index
    return -1  # Return -1 if no matching site is found


def get_neighbor_indices_crystalnn(structure, site_index):
    """
    Get the neighbor indices for a specific site using CrystalNN.

    Parameters:
        structure (pymatgen.Structure): The atomic structure.
        site_index (int): Index of the target site.

    Returns:
        list: A list of site indices of the neighbors.
    """
    crystal_nn = CrystalNN()
    neighbors = crystal_nn.get_nn_info(structure, site_index)
    neighbor_indices = [neighbor["site_index"] for neighbor in neighbors]
    return neighbor_indices


def extract_filename(file_path):
    """
    Extracts the filename without extension from a given file path.

    Parameters:
    - file_path (str): The complete file path from which the filename is to be extracted.

    Returns:
    - str: The filename without its extension.

    Example:
    If file_path is 'clusters/octohedral.xyz', the function returns 'octohedral'.
    """
    # Use os.path.basename to get the filename with extension from the file path
    file_name_with_ext = os.path.basename(file_path)

    # Use os.path.splitext to remove the file extension and get the filename
    file_name, _ = os.path.splitext(file_name_with_ext)

    return file_name


def order_data(data):
    """
    Organizes each sublist in the provided data such that the tuples are ordered by the second element in descending order.

    Parameters:
    - data (list of lists of tuples): The data to be organized. Each sublist contains tuples of the form (name, value).

    Returns:
    - list of lists of tuples: The organized data with tuples sorted by the second element in descending order within each sublist.
    """
    # Sort each sublist based on the second element of the tuples (the value), in descending order
    sorted_data = [sorted(sublist, key=lambda x: x[1], reverse=True)
                   for sublist in data]

    return sorted_data


def flatten_data(data):
    """
    Flattens a list of lists of tuples into a single list of tuples.

    Parameters:
    - data (list of lists of tuples): The data to be flattened.

    Returns:
    - list of tuples: The flattened data.
    """
    # Use a list comprehension to flatten the list of lists
    flattened_data = [item for sublist in data for item in sublist]
    return flattened_data


def get_oxidation_state_formula(formula):
    """
    Determines the oxidation states of each element in a given formula.
    
    Args:
        formula (str): Chemical formula (e.g., "Fe2O3").
    
    Returns:
        dict: A dictionary mapping each element to its oxidation state(s).
    """
    try:
        # Use pymatgen's Composition to parse the formula
        composition = Composition(formula)
        
        # Try oxidation state guessing
        oxidation_states = composition.oxi_state_guesses()

        if oxidation_states:
            # Return the first guess (most probable based on pymatgen's algorithm)
            return oxidation_states[0]
        else:
            return "Oxidation states could not be determined."
    except Exception as e:
        return f"An error occurred: {e}"


def get_oxidation_state(possible_species, atom):
    """
    Given a list of possible species, this function returns the oxidation state of the specified atom.

    Args:
    - possible_species (list): A list of strings representing species with their oxidation states, e.g., ['O2-', 'V5+', 'Cu+'].
    - atom (str): The symbol of the atom for which to retrieve the oxidation state, e.g., 'V'.

    Returns:
    - float: The oxidation state of the atom, or None if not found.
    """
    for species in possible_species:
        # Check if the species starts with the atom symbol (e.g., "V" for vanadium)
        if species.startswith(atom):
            # Extract the oxidation state, including the sign
            oxidation_state_str = species[len(atom):]
        
            # Handle cases like '2+' or '3-' (sign at the end)
            if oxidation_state_str.endswith('+'):
                # If no number, assume it's 1
                if oxidation_state_str[:-1] == '':
                    oxidation_state = 1.0
                else:
                    oxidation_state = float(oxidation_state_str[:-1])  # Positive oxidation state
            elif oxidation_state_str.endswith('-'):
                # If no number, assume it's -1
                if oxidation_state_str[:-1] == '':
                    oxidation_state = -1.0
                else:
                    oxidation_state = -float(oxidation_state_str[:-1])  # Negative oxidation state

            # Handle cases like '+2' or '-3' (sign at the start)
            elif oxidation_state_str.startswith('+'):
                # If no number, assume it's 1
                if oxidation_state_str[1:] == '':
                    oxidation_state = 1.0
                else:
                    oxidation_state = float(oxidation_state_str[1:])  # Positive oxidation state
            elif oxidation_state_str.startswith('-'):
                # If no number, assume it's -1
                if oxidation_state_str[1:] == '':
                    oxidation_state = -1.0
                else:
                    oxidation_state = float(oxidation_state_str)  # Negative oxidation state

            # If no sign is provided, assume positive oxidation state
            else:
                oxidation_state = float(oxidation_state_str)

            return oxidation_state

    # If no matching species is found, return None
    return None


def get_cluster_properties_old(mp_id, central_atom, api_key=api_key):
    """
    This function retrieves the band gap, oxidation state of the central atom, and density of a material.

    Args:
    - mp_id (str): The Materials Project ID of the material.
    - central_atom (str): The symbol of the central atom (e.g., 'V' for vanadium).
    - api_key (str): Your Materials Project API key.

    Returns:
    - dict: A dictionary containing band gap, oxidation state, and density.
    """
    with MPRester(api_key) as mpr:
        # Search for the material using its MP-ID
        materials = mpr.materials.summary.search(
            material_ids=[mp_id]
        )

        # Select the first material from the returned materials list
        material = materials[0]

        # Initialize a dictionary to store the properties
        properties = {}

        # Get the band gap of the material
        if hasattr(material, 'band_gap'):
            properties['band_gap'] = material.band_gap
        else:
            properties['band_gap'] = "Band gap not available"

        # Get the oxidation state of the central atom
        possible_species = material.possible_species
    
        oxidation_state = get_oxidation_state(possible_species, central_atom)
        if oxidation_state is not None:
            properties['oxidation_state'] = oxidation_state
        else:
            properties['oxidation_state'] = "Oxidation state not available"

        # Add possible pecies to later get the charge dictionary
        properties['possible_species'] = possible_species

        # Get density
        if hasattr(material, 'density'):
            properties['density'] = material.density
        else:
            properties['density'] = "Density not available"

        return properties


def get_cluster_properties(mp_id, api_key=api_key):
    """
    This function retrieves the band gap, oxidation state of the central atom, and density of a material.

    Args:
    - mp_id (str): The Materials Project ID of the material.
    - central_atom (str): The symbol of the central atom (e.g., 'V' for vanadium).
    - api_key (str): Your Materials Project API key.

    Returns:
    - dict: A dictionary containing band gap, oxidation state, and density.
    """
    with MPRester(api_key) as mpr:
        # Search for the material using its MP-ID
        materials = mpr.materials.summary.search(
            material_ids=[mp_id]
        )

        # Select the first material from the returned materials list
        material = materials[0]

        # Initialize a dictionary to store the properties
        properties = {}

        # Get the band gap of the material
        if hasattr(material, 'band_gap'):
            properties['band_gap'] = material.band_gap
        else:
            properties['band_gap'] = "Band gap not available"

        # Get density
        if hasattr(material, 'density'):
            properties['density'] = material.density
        else:
            properties['density'] = "Density not available"

        '''
        # Get possible species
        if hasattr(material, 'possible_species'):
            properties['possible_species'] = material.possible_species
        else:
            properties['possible_species'] = "Possible species not available"
        '''


        return properties


def compute_number_of_unique_ligands(neighbors):
    """
    Returns the number of unique ligand elements in the provided list of neighbors.
    
    Parameters:
    -----------
    neighbors : list of dict
        Each dict is typically returned by something like CrystalNN.get_nn_info(structure, site_index).
        It should have at least a "site" key storing the neighbor site, e.g., neighbor_info["site"].

    Returns:
    --------
    int
        The number of unique ligand elements.
    """
    if not neighbors:
        return 0  # Return 0 if no neighbors
    
    # Extract unique atomic species from neighbors
    unique_ligands = {n["site"].specie.symbol for n in neighbors}  # Set comprehension for unique elements
    
    return len(unique_ligands)


def compute_average_bond_distance(neighbors, center_site):
    """
    Returns the average distance from `center_site` to each neighbor site.

    Parameters
    ----------
    neighbors : list of dict
        Each dict is typically returned by something like CrystalNN.get_nn_info(structure, site_index).
        It should have at least a "site" key storing the neighbor site, e.g. neighbor_info["site"].

    center_site : pymatgen.core.sites.Site or PeriodicSite
        The site (usually the central atom) from which to compute distances.

    Returns
    -------
    float
        The average distance. If neighbors is empty, returns 0.0.
    """
    if not neighbors:
        return 0.0

    # Calculate distance from center_site to each neighbor's site
    distances = [center_site.distance(n["site"]) for n in neighbors]
    return sum(distances) / len(distances)


def compute_bond_length_std(neighbors, center_site):
    """
    Returns the standard deviation of the bond lengths from `center_site` to each neighbor site.

    Parameters
    ----------
    neighbors : list of dict
        Each dict is typically returned by something like CrystalNN.get_nn_info(structure, site_index).
        It should have at least a "site" key storing the neighbor site, e.g. neighbor_info["site"].

    center_site : pymatgen.core.sites.Site or PeriodicSite
        The site (usually the central atom) from which to compute distances.

    Returns
    -------
    float
        The standard deviation of bond lengths. If neighbors is empty, returns 0.0.
    """
    if not neighbors:
        return 0.0

    # Calculate bond lengths
    distances = [center_site.distance(n["site"]) for n in neighbors]

    # Compute and return standard deviation
    return np.std(distances, ddof=1)  # ddof=1 for sample standard deviation


def compute_electronegativity_stats(neighbors):
    """
    Returns the average and standard deviation of the electronegativity of neighbor atoms.

    Parameters
    ----------
    neighbors : list of dict
        Each dict is typically returned by something like CrystalNN.get_nn_info(structure, site_index).
        It should have at least a "site" key storing the neighbor site, e.g. neighbor_info["site"].

    Returns
    -------
    tuple (float, float)
        The average electronegativity and standard deviation.
        If neighbors list is empty or electronegativity is unavailable, returns (0.0, 0.0).
    """
    if not neighbors:
        return 0.0, 0.0

    # Get electronegativity values of neighbor elements
    electronegativities = []
    for n in neighbors:
        element = n["site"].specie.symbol  # Get element symbol
        try:
            electronegativities.append(Element(element).X)  # Get Pauling electronegativity
        except AttributeError:
            continue  # Skip elements without electronegativity

    # Check if we got any valid electronegativities
    if not electronegativities:
        return 0.0, 0.0

    # Compute average and standard deviation
    avg_en = np.mean(electronegativities)
    std_en = np.std(electronegativities, ddof=1)  # Sample standard deviation

    return avg_en, std_en


def read_mp_id_file(file_path):
    """
    Reads a text file containing compound names and their corresponding 
    Materials Project MP-IDs in a specific format, and returns a dictionary.

    The expected file format is:
    # Comment line or header (optional)
    CompoundFormula: MP-ID

    Example:
    NiO: mp-19009
    Fe2O3: mp-19770
    V2O5: mp-25279

    The function skips comment lines that start with a '#' and empty lines.

    Args:
        file_path (str): The path to the text file containing the data.

    Returns:
        dict: A dictionary where the keys are compound formulas (str) 
              and the values are their corresponding Materials Project IDs (str).

    Example:
        >>> read_mp_id_file('compounds.txt')
        {'NiO': 'mp-19009', 'Fe2O3': 'mp-19770', 'V2O5': 'mp-25279'}
    """
    compound_mp_id = {}
    with open(file_path, 'r') as file:
        for line in file:
            # Skip comment or empty lines
            if line.startswith('#') or not line.strip():
                continue
            # Split line by the colon
            parts = line.split(':')
            if len(parts) == 2:
                formula = parts[0].strip()
                mp_id = parts[1].strip()
                compound_mp_id[formula] = mp_id

    return compound_mp_id


def quadrupole_moment(positions, charges):
    """
    Calculate the non traceless form of the quadrupole moment tensor for a system of point charges. 

    Args:
    - positions: Nx3 array, where N is the number of particles, and each row is the (x, y, z) coordinates of a particle.
    - charges: 1D array of length N, where each element is the charge of the corresponding particle.

    Returns:
    - Q: 3x3 numpy array representing the quadrupole moment tensor.
    """
    Q = np.zeros(
        # Initialize the quadrupole moment tensor as a 3x3 zero matrix.
        (3, 3))

    for pos, charge in zip(positions, charges):
        r_x, r_y, r_z = pos

        # Update the Q matrix using the formula.
        Q[0, 0] += charge * (r_x * r_x)
        Q[0, 1] += charge * (r_x * r_y)
        Q[0, 2] += charge * (r_x * r_z)

        Q[1, 0] += charge * (r_y * r_x)
        Q[1, 1] += charge * (r_y * r_y)
        Q[1, 2] += charge * (r_y * r_z)

        Q[2, 0] += charge * (r_z * r_x)
        Q[2, 1] += charge * (r_z * r_y)
        Q[2, 2] += charge * (r_z * r_z)

    return Q


def quadrupole_moment_normalized(positions, charges):
    """
    Calculate the non traceless form of the quadrupole moment tensor for a system of point charges, normalized by the atomic number.

    Args:
    - positions: Nx3 array, where N is the number of particles, and each row is the (x, y, z) coordinates of a particle.
    - charges: 1D array of length N, where each element is the charge of the corresponding particle.

    Returns:
    - Q: 3x3 numpy array representing the normalized quadrupole moment tensor.
    """
    Q = np.zeros(
        # Initialize the quadrupole moment tensor as a 3x3 zero matrix.
        (3, 3))

    # Loop through each position, charge, and atomic number
    for pos, charge in zip(positions, charges):
        r_x, r_y, r_z = pos
        dist = np.sqrt(r_x**2+r_y**2+r_z**2)

        # Update the Q matrix using the normalized formula.
        normalization_factor = charge / (dist)**6

        Q[0, 0] += normalization_factor * (r_x * r_x)
        Q[0, 1] += normalization_factor * (r_x * r_y)
        Q[0, 2] += normalization_factor * (r_x * r_z)

        Q[1, 0] += normalization_factor * (r_y * r_x)
        Q[1, 1] += normalization_factor * (r_y * r_y)
        Q[1, 2] += normalization_factor * (r_y * r_z)

        Q[2, 0] += normalization_factor * (r_z * r_x)
        Q[2, 1] += normalization_factor * (r_z * r_y)
        Q[2, 2] += normalization_factor * (r_z * r_z)

    return Q


def quadrupole_anisotropy_matrix(qxx, qyy, qzz):
    """
    Calculate the quadrupole anisotropy matrix.

    Parameters:
        qxx (float): Quadrupole component along xx.
        qyy (float): Quadrupole component along yy.
        qzz (float): Quadrupole component along zz.

    Returns:
        np.ndarray: 3x3 quadrupole anisotropy matrix.
    """
    # Normalization factor
    normalization = 1
    #(qxx + qyy + qzz) / 3.0

    if normalization == 0:
        raise ValueError("Normalization factor is zero; qxx, qyy, and qzz cannot all be zero.")

    # Initialize the anisotropy matrix with absolute differences normalized
    q_anisotropy_matrix = np.array([
    [0, (np.abs(qxx - qyy)) / normalization, (np.abs(qxx - qzz)) / normalization],
    [(np.abs(qyy - qxx)) / normalization, 0, (np.abs(qyy - qzz)) / normalization],
    [(np.abs(qzz - qxx)) / normalization, (np.abs(qzz - qyy)) / normalization, 0]
])


    return q_anisotropy_matrix


def q_anisotropy_matrix_sum(q_anisotropy_matrix):
    """
    Compute the sum of select off-diagonal elements of an anisotropy matrix.

    This function calculates the sum of the elements located at indices (0, 1), (0, 2),
    and (1, 2) of the input matrix. It assumes that the provided matrix is a 2D array-like
    object with at least 3 rows and 3 columns.

    Parameters:
        q_anisotropy_matrix (array-like): A two-dimensional array or matrix representing
            anisotropy values. The matrix must be indexable with two indices and have dimensions
            that allow access to the elements at (0, 1), (0, 2), and (1, 2).

    Returns:
        float: The sum of the matrix elements at positions (0, 1), (0, 2), and (1, 2).
    """

    sum = q_anisotropy_matrix[0,1] + q_anisotropy_matrix[0,2] + q_anisotropy_matrix[1,2] 

    return sum


def dipole_moment_normalized(positions, charges):
    """
    Compute the normalized dipole moment vector for a system of charges.

    Parameters:
    positions : list of tuples
        A list of 3D position vectors (x, y, z) for the charges.
    charges : list of floats
        A list of charges corresponding to the position vectors.

    Returns:
    numpy.ndarray
        A 3D vector representing the normalized dipole moment.

    Notes:
    ------
    - Positions with a distance of zero are skipped to avoid division by zero.
    - The normalization factor is calculated as charge / (distance^5).
    """

    # Initialize the dipole moment vector as a 3D vector
    P = np.zeros(3)
    
    for pos, charge in zip(positions, charges):
        r_x, r_y, r_z = pos
        dist = np.sqrt(r_x**2 + r_y**2 + r_z**2)
        
        # Avoid division by zero in normalization
        if dist == 0:
            continue

        # Normalize the position vector
        normalization_factor = charge / dist**5
        P[0] += normalization_factor * r_x
        P[1] += normalization_factor * r_y
        P[2] += normalization_factor * r_z

    return P


def dipole_anisotropy_matrix(dipole_vector):
    """
    Create a matrix where the components are the difference squared 
    of the components of the vector normalized by the mean of the components.

    Parameters:
    dipole_vector (array-like): A 3D vector representing the dipole moment.

    Returns:
    numpy.ndarray: A 3x3 matrix as described.
    """
    # Convert the input vector to a NumPy array
    dipole_vector = np.array(dipole_vector)
    
    # Compute the mean of the components
    #mean_value = np.mean(dipole_vector)
    
    # Check for zero mean to avoid division by zero
    #if mean_value == 0:
        #raise ValueError("The mean of the dipole vector components is zero, normalization not possible.")
    
    # Declare the normalization variable
    normalization = 1

    # Initialize the matrix using absolute differences normalized by the mean
    dipole_matrix = np.array([
        [0, np.abs(dipole_vector[0] - dipole_vector[1]) / normalization, np.abs(dipole_vector[0] - dipole_vector[2]) / normalization],
        [np.abs(dipole_vector[1] - dipole_vector[0]) / normalization, 0, np.abs(dipole_vector[1] - dipole_vector[2]) / normalization],
        [np.abs(dipole_vector[2] - dipole_vector[0]) / normalization, np.abs(dipole_vector[2] - dipole_vector[1]) / normalization, 0]
    ])
    
    return dipole_matrix


def p_anisotropy_matrix_sum(dipole_anisotropy_matrix):
    """
    Compute the sum of select off-diagonal elements of a dipole anisotropy matrix.

    This function calculates the sum of the elements located at indices (0, 1), (0, 2),
    and (1, 2) of the input matrix. It assumes that the provided matrix is a 2D array-like
    object with at least 3 rows and 3 columns.

    Parameters:
        dipole_anisotropy_matrix (array-like): A two-dimensional array or matrix representing
            anisotropy values. The matrix must be indexable with two indices and have dimensions
            that allow access to the elements at (0, 1), (0, 2), and (1, 2).

    Returns:
        float: The sum of the matrix elements at positions (0, 1), (0, 2), and (1, 2).
    """

    # Compute the sum of the specific off-diagonal elements
    sum = dipole_anisotropy_matrix[0, 1] + dipole_anisotropy_matrix[0, 2] + dipole_anisotropy_matrix[1, 2]

    return sum


def get_charges_old(possible_species, atomic_symbols):
    """
    Given a list of possible species and atomic symbols, this function returns a list of charges
    (oxidation states) corresponding to each atomic symbol.

    Args:
    - possible_species (list): A list of strings representing species with their oxidation states,
                               e.g., ['O2-', 'V5+'].
    - atomic_symbols (list): A list of atomic symbols (e.g., ['O', 'V', 'H']) for which the charges
                             are to be determined.

    Returns:
    - list: A list of integers representing the charges (oxidation states) for the corresponding atomic symbols.
            If an oxidation state is not found for an atom, the value will be None.

    Example:
    >>> possible_species = ['O2-', 'V5+', 'H1+', 'Fe2+']
    >>> atomic_symbols = ['O', 'V', 'H', 'Fe']
    >>> get_charges(possible_species, atomic_symbols)
    [-2, 5, 1, 2]
    """

    charges = []  # Initialize an empty list to store charges

    # Iterate through each atom in the list of atomic symbols
    for atom in atomic_symbols:
        # Get the charge (oxidation state) using the get_oxidation_state function
        charge = get_oxidation_state(possible_species, atom)
        # Append the charge to the charges list
        charges.append(charge)

    return charges
 

def get_charges(atomic_symbols, oxidation_states):
    """
    Given a list of atomic symbols and a dictionary mapping
    symbols to oxidation states, return a list of charges for
    each atomic symbol in atom_list.
    
    :param atom_list: List of atomic symbols (e.g. ["Nb","Se","Cr"]).
    :param oxidation_states: Dictionary mapping atomic symbols to charges 
                             (e.g. {"Nb": 2.5, "Cr": 3.0, "Se": -2.0}).
    :return: List of charges corresponding to the symbols in atom_list.
    :raises ValueError: if a symbol in atom_list is not found in oxidation_states.
    """
    charge_array = []
    
    for atom in atomic_symbols:
        if atom not in oxidation_states:
            raise ValueError(f"Unknown atomic symbol '{atom}' in the oxidation states dictionary.")
        charge_array.append(oxidation_states[atom])
    
    return charge_array


def get_eigs_of_qm(quadrupole_moment):
    """Work in progress"""
    eigenvalues = linalg.eig(quadrupole_moment)

    #Order the eigen values in some way 

    return eigenvalues


def get_unique_output_folder(base_folder):
    """Generate a unique folder name by adding a numeric suffix if the folder exists."""
    folder = base_folder
    counter = 1
    while os.path.exists(folder):
        folder = f"{base_folder}_{counter}"
        counter += 1
    os.makedirs(folder)
    return folder


def write_factor_dictionary_to_file(factor_dict, filename):
    """
    Writes the factor dictionary to a JSON file.

    Args:
    - factor_dict: Dictionary containing data (including numpy arrays).
    - filename: The name of the file to write the dictionary to.
    """
    print(f"Started writing dictionary to {filename}")
    # Convert the dictionary to a JSON-serializable format
    serializable_dict = convert_to_json_serializable(factor_dict)

    with open(filename, "w") as fp:
        json.dump(serializable_dict, fp, indent=4)  # Use JSON to serialize the dictionary
        fp.flush()  # Ensure the buffer is flushed
        os.fsync(fp.fileno())  # Ensure file is written to disk
    print(f"Done writing dict to {filename}")


class DualWriter:
    def __init__(self, log_file):
        self.terminal = sys.stdout  # Keep the original terminal stdout
        self.log_file = log_file    # File to write logs to

    def write(self, message):
        self.terminal.write(message)    # Write to terminal
        self.log_file.write(message)    # Write to log file

    def flush(self):
        # This flushes the output for both the terminal and the log file
        self.terminal.flush()
        self.log_file.flush()


def convert_to_json_serializable(data):
    """
    Recursively convert data to JSON-serializable format.
    Converts numpy arrays to lists, and handles other non-serializable data types as needed.
    """
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, dict):
        return {key: convert_to_json_serializable(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_json_serializable(element) for element in data]
    elif isinstance(data, tuple):
        return tuple(convert_to_json_serializable(element) for element in data)
    else:
        return data    


def generate_factor_df(factor_dict_dir_path, mat_props=False, dipole=False, quadrupole=False):
    # Initialize an empty list to store rows of data
    data_list = []

    # Iterate through each JSON file in the directory
    for file_path in Path(factor_dict_dir_path).glob("*.json"):
        with open(file_path, "r") as file:
            material_dict = json.load(file)

            # Extract MP-ID and Material Name
            material = material_dict.get("MP-ID", file_path.stem.replace("_factor_dict", ""))
            chem_formula = material_dict.get("Chem Formula", "Unknown")
            cif_name = material_dict.get("CIF Name", "Unkown")

            # Extract Space Group Number
            space_group_number = material_dict.get("Space Group Number", np.nan)  # Default to NaN if missing

            # Initialize row with mandatory fields
            data_row = [material, chem_formula, cif_name, space_group_number]

            # Define column headers
            columns = ["Material", "Chem Formula", "Cif Name", "Space Group Number"]

            if mat_props:

                #Add chemical info bond length number of ligands and electornegativity
                ave_bond_length = material_dict.get("Average Bond Length",0)
                std_bond_length = material_dict.get("Bond Length Std",0)
                num_of_ligands = material_dict.get("Number of Unique Ligands",0)
                ave_en = material_dict.get("Average Electronegativity",0)
                std_en = material_dict.get("Electronegativity Std",0)
                data_row.extend([ave_bond_length, std_bond_length, num_of_ligands, ave_en, std_en])
                columns.extend(["Average Bond Length", "Bond Length Std", "Number of Unique Ligands", "Average Electronegativity", "Std Electronegativity"])


                # Extract band gap, density, oxidation states
                band_gap = material_dict.get("band_gap", 0.0)
                density = material_dict.get("density", 0.0)
                oxidation_states = material_dict.get("oxidation_states", {})
                oxidation_states_str = str(oxidation_states)

                data_row.extend([band_gap, density, oxidation_states_str])
                columns.extend(["Band Gap", "Density", "Oxidation States"])

            if dipole:
                # Extract Dipole Moments
                dipole_moment_norm = np.array(material_dict.get("dipole moment normalized", [0, 0, 0])).flatten()
                normalized_d_anisotropy_matrix = np.array(
                    material_dict.get("normalized dipole anisotropy matrix", [[0] * 3] * 3)
                ).flatten()
                normalized_d_anisotropy_matrix_sum = material_dict.get("normalized dipole anisotropy matrix sum", 0.0)

                data_row.extend([*dipole_moment_norm, *normalized_d_anisotropy_matrix, normalized_d_anisotropy_matrix_sum])
                columns.extend([f"DM Norm {i}" for i in range(3)])
                columns.extend([f"Aniso DM {i}" for i in range(9)])
                columns.append("Aniso Sum DM")

            if quadrupole:
                # Extract Quadrupole Moments
                quadrupole_moment_norm = np.array(material_dict.get("quadrupole moment normalized", [[0] * 3] * 3)).flatten()
                normalized_q_anisotropy_matrix = np.array(
                    material_dict.get("normalized quadrupole anisotropy matrix", [[0] * 3] * 3)
                ).flatten()
                normalized_q_anisotropy_matrix_sum = material_dict.get("normalized quadrupole anisotropy matrix sum", 0.0)

                data_row.extend([*quadrupole_moment_norm, *normalized_q_anisotropy_matrix, normalized_q_anisotropy_matrix_sum])
                columns.extend([f"QM Norm {i}" for i in range(9)])
                columns.extend([f"Aniso QM {i}" for i in range(9)])
                columns.append("Aniso Sum QM")

            # Append row to data list
            data_list.append(data_row)

    # Create DataFrame
    factor_df = pd.DataFrame(data_list, columns=columns)

    # Set 'MP-ID' as the index
    factor_df.set_index("Material", inplace=True)

    return factor_df


def load_anisotropy_matrix(csv_path, center_atom):
    """
    Reads an anisotropy matrix CSV file, filters rows based on the specified center atom, 
    and formats the dataframe.

    Parameters:
        csv_path (str or Path): Path to the CSV file.
        center_atom (str): The central atom to filter by.

    Returns:
        pd.DataFrame: A formatted dataframe containing only the entries with the specified center atom.
    """
    csv_path = Path(csv_path)

    # Read the CSV file, assuming tab-separated values
    anisotropy_matrix_df = pd.read_csv(csv_path, sep='\t')

    # Set the first column as the index
    anisotropy_matrix_df.set_index(anisotropy_matrix_df.columns[0], inplace=True)

    # Filter rows that contain the specified center atom
    anisotropy_matrix_df = anisotropy_matrix_df.loc[anisotropy_matrix_df.index.str.contains(center_atom)]

    # Extract only the first part of the index (before '_')
    anisotropy_matrix_df.index = [name.split('_')[0] for name in anisotropy_matrix_df.index]
    anisotropy_matrix_df.index.name = 'Material'

    return anisotropy_matrix_df


def load_anisotropy_matrix_json(json_file_path):
    """
    Load a JSON file where each key maps to a 3x3 matrix
    (as a list of lists), and return a DataFrame of the
    flattened values (m00..m22).
    """
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    rows = []
    for material, mat3x3 in data.items():
        # mat3x3 is something like:
        # [
        #   [0.0,     0.0,     0.002096],
        #   [0.0,     0.0,     0.002096],
        #   [0.002096,0.002096,0.0     ]
        # ]
        # Unpack or just index:
        m00, m01, m02 = mat3x3[0]
        m10, m11, m12 = mat3x3[1]
        m20, m21, m22 = mat3x3[2]

        rows.append({
            'Material': material,
            'm00': m00, 'm01': m01, 'm02': m02,
            'm10': m10, 'm11': m11, 'm12': m12,
            'm20': m20, 'm21': m21, 'm22': m22
        })

    df = pd.DataFrame(rows)
    return df.set_index('Material')


def print_factor_dict(factor_dict_path):
    """
    Reads a JSON factor dictionary from a given path and prints its contents.

    Parameters:
        factor_dict_path (str or Path): Path to the JSON file.

    Returns:
        dict: The loaded dictionary.
    """
    factor_dict_path = Path(factor_dict_path)

    if not factor_dict_path.exists():
        print(f"Error: File '{factor_dict_path}' not found.")
        return None

    # Load the JSON file
    with open(factor_dict_path, 'r') as file:
        factor_dict = json.load(file)

    # Print the keys and values
    print(f"\nContents of {factor_dict_path.name}:")
    print("-" * 50)
    
    for key, val in factor_dict.items():
        print(f"{key}: {val}")

    return factor_dict


def compute_normed_off_diagonal_sum(anisotropy_spectra_matrix):
    """
    Compute the normalized sum of off-diagonal entries for each row in the given anisotropy spectra matrix.

    Parameters:
    anisotropy_spectra_matrix (pd.DataFrame): A pandas DataFrame containing anisotropy matrix data.

    Returns:
    pd.DataFrame: The original DataFrame with additional columns for off-diagonal sums and their normalized values.
    """

    # Identify the off-diagonal columns
    off_diagonal_cols = ["m01", "m02", "m12"]

    # Sum the off-diagonal entries (row-wise)
    anisotropy_spectra_matrix["Off Diagonal Sum"] = anisotropy_spectra_matrix[off_diagonal_cols].sum(axis=1)

    # Determine the largest off-diagonal entry in each row
    largest_off_diagonal = anisotropy_spectra_matrix["Off Diagonal Sum"].max()

    # Compute the ratio (handling zero-division issues)
    if largest_off_diagonal != 0:
        anisotropy_spectra_matrix["Normed Sum"] = anisotropy_spectra_matrix["Off Diagonal Sum"] / largest_off_diagonal
    else:
        anisotropy_spectra_matrix["Normed Sum"] = 0  # Avoid division by zero

    return anisotropy_spectra_matrix


def compute_off_diagonal_sum(anisotropy_spectra_matrix):
    """
    Compute the sum of off-diagonal entries for each row in the given anisotropy spectra matrix.

    Parameters:
    anisotropy_spectra_matrix (pd.DataFrame): A pandas DataFrame containing anisotropy matrix data.

    Returns:
    pd.DataFrame: The original DataFrame with an additional column for off-diagonal sums.
    """

    # Identify the off-diagonal columns
    off_diagonal_cols = ["m01", "m02", "m12"]

    # Sum the off-diagonal entries (row-wise)
    anisotropy_spectra_matrix["Anisotropy Matrix Sum"] = anisotropy_spectra_matrix[off_diagonal_cols].sum(axis=1)

    return anisotropy_spectra_matrix


def compute_normed_spacegroup_number(factor_df):
    """
    Computes the normalized space group number by dividing 
    each space group number by 230.

    Parameters:
    factor_df (pd.DataFrame): A pandas DataFrame containing a column 
                              "Space Group Number" with integer values.

    Returns:
    pd.DataFrame: The input DataFrame with an added column 
                  "Normed Spacegroup Number" containing the 
                  normalized values.
    """

    if "Space Group Number" not in factor_df.columns:
        raise KeyError("The input DataFrame must contain a 'Space Group Number' column.")

    normalization = 1 / 230
    factor_df["Normed Spacegroup Number"] = normalization * factor_df["Space Group Number"]

    return factor_df


def filter_matching_mpids(factor_df, anisotropy_df):
    """
    Filters two DataFrames to keep only rows with matching MP-IDs.

    Parameters:
        factor_df (pd.DataFrame): The factor DataFrame indexed by MP-ID.
        anisotropy_df (pd.DataFrame): The anisotropy DataFrame indexed by MP-ID.

    Returns:
        tuple: Filtered DataFrames containing only matching MP-IDs.
    """
    # Find common MP-IDs in both DataFrames
    common_mp_ids = factor_df.index.intersection(anisotropy_df.index)
    
    # Filter both DataFrames to keep only the rows with matching MP-IDs
    filtered_factor_df = factor_df.loc[common_mp_ids]
    filtered_anisotropy_df = anisotropy_df.loc[common_mp_ids]
    
    return filtered_factor_df, filtered_anisotropy_df


def remove_nan_entries(factor_df, anisotropy_df):
    """
    Removes rows with NaN values from the factor DataFrame and filters the anisotropy DataFrame accordingly.

    Parameters:
        factor_df (pd.DataFrame): The factor DataFrame indexed by MP-ID.
        anisotropy_df (pd.DataFrame): The anisotropy DataFrame indexed by MP-ID.

    Returns:
        tuple: Cleaned factor DataFrame and corresponding anisotropy DataFrame.
    """
    # Drop rows with NaN values from the factor DataFrame
    cleaned_factor_df = factor_df.dropna()
    
    # Extract the MP-IDs of the valid rows
    valid_mp_ids = cleaned_factor_df.index
    
    # Filter the anisotropy DataFrame to keep only rows matching the valid MP-IDs
    cleaned_anisotropy_df = anisotropy_df.loc[anisotropy_df.index.isin(valid_mp_ids)]
    
    return cleaned_factor_df, cleaned_anisotropy_df


def align_dataframes_by_index(factor_df, anisotropy_df):
    """
    Align two DataFrames by their indices, ensuring they have the same set of
    rows in the same order.

    Args:
        df1 (pd.DataFrame): First DataFrame (indexed by Material).
        df2 (pd.DataFrame): Second DataFrame (indexed by Material).

    Returns:
        (pd.DataFrame, pd.DataFrame): The aligned DataFrames, filtered to the
        intersection of their indices and sorted row-by-row.
    """
    # Find the common indices (intersection of MPIDs)
    common_indices = factor_df.index.intersection(anisotropy_df.index)
    
    if common_indices.empty:
        print("Warning: No common indices. Returning empty DataFrames.")
    
    # Filter and sort both DataFrames by the common indices
    factor_df_aligned = factor_df.loc[common_indices].sort_index()
    anisotropy_df_aligned = anisotropy_df.loc[common_indices].sort_index()
    
    return factor_df_aligned, anisotropy_df_aligned


def align_dataframes(factor_df, anisotropy_spectra_matrix):
    """
    Cleans and aligns the factor DataFrame and anisotropy spectra DataFrame by:
    1. Filtering only the rows with matching MP-IDs.
    2. Removing rows with NaN values in the factor DataFrame and filtering anisotropy accordingly.
    3. Ensuring both DataFrames have the same indices in the same order.

    Parameters:
        factor_df (pd.DataFrame): The factor dictionary DataFrame indexed by MP-ID.
        anisotropy_spectra_matrix (pd.DataFrame): The anisotropy spectra DataFrame indexed by MP-ID.

    Returns:
        tuple: Cleaned and aligned (factor_df, anisotropy_spectra_matrix).
    """
    factor_df, anisotropy_spectra_matrix = filter_matching_mpids(factor_df, anisotropy_spectra_matrix)
    factor_df, anisotropy_spectra_matrix = remove_nan_entries(factor_df, anisotropy_spectra_matrix)
    factor_df, anisotropy_spectra_matrix = align_dataframes_by_index(factor_df, anisotropy_spectra_matrix)
    return factor_df, anisotropy_spectra_matrix

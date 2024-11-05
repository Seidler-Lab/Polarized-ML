from pymatgen.core import Structure
import numpy as np
from scipy.special import sph_harm
import os
from mp_api.client import MPRester
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core import Element

parity = True  # Include parity or not for the plot
degree_l = 40

# List of transition metals based on their atomic numbers
transition_metals = [Element(sym).symbol for sym in [
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", 
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", 
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg"]]

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


def extract_cluster(cif_file, index_number, cluster_radius=3):
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

    # Find all sites (atoms) within the specified radius around the atom at 'index_number'
    sites = symmetrized_structure.get_sites_in_sphere(symmetrized_structure.cart_coords[index_number],
                                                      cluster_radius)

    # Create a new structure consisting of only the atoms within the cluster
    cluster_structure = Structure.from_sites(sites)

    # Initialize lists to hold the coordinates, atomic symbols, and atomic numbers
    coords = []
    atomic_symbols = []
    atomic_numbers = []

    # Reverse the structure so that the central atom is the first one in the final list
    # This ensures that the central atom (the one at index_number) is the first element
    for site in reversed(cluster_structure):
        coords.append(site.coords)  # Append the coordinates of each atom
        # Append the atomic symbol (e.g., "H", "O")
        atomic_symbols.append(site.specie.symbol)
        atomic_numbers.append(site.specie.Z)  # Append the atomic number (Z)

    # Convert the lists to numpy arrays for easier manipulation and center the central atom at (0,0,0)
    coords = np.array(translate_coords(coords))
    atomic_symbols = np.array(atomic_symbols)
    atomic_numbers = np.array(atomic_numbers)

    # Return the coordinates, atomic symbols, and atomic numbers
    return coords, atomic_symbols, atomic_numbers


def extract_cluster(cif_file, atomic_symbol, cluster_radius=3):
    """
    Extracts a cluster of atoms around the first occurrence of the specified atomic symbol in a crystal structure from a CIF file.
    The central atom (specified by atomic_symbol) will be the first in the returned arrays.

    Parameters:
    - cif_file (str): Path to the CIF file containing the crystal structure.
    - atomic_symbol (str): Atomic symbol (e.g., "Si", "O") of the atom to center the cluster around.
    - cluster_radius (float): Radius of the cluster in angstroms within which atoms will be selected.

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

    # Find all sites within the specified radius around the chosen atom
    sites = symmetrized_structure.get_sites_in_sphere(symmetrized_structure.cart_coords[chosen_atom_index],
                                                      cluster_radius)

    # Create a new structure consisting of only the atoms within the cluster
    cluster_structure = Structure.from_sites(sites)

    # Initialize lists to hold the coordinates, atomic symbols, and atomic numbers
    coords = []
    atomic_symbols = []
    atomic_numbers = []

    # Reverse the structure so that the central atom is the first one in the final list
    for site in reversed(cluster_structure):
        coords.append(site.coords)  # Append the coordinates of each atom
        atomic_symbols.append(site.specie.symbol)  # Append the atomic symbol
        atomic_numbers.append(site.specie.Z)  # Append the atomic number

    # Convert the lists to numpy arrays for easier manipulation and center the central atom at (0,0,0)
    coords = np.array(translate_coords(coords))
    atomic_symbols = np.array(atomic_symbols)
    atomic_numbers = np.array(atomic_numbers)

    # Return the coordinates, atomic symbols, and atomic numbers
    return coords, atomic_symbols, atomic_numbers


def extract_cluster(structure, atomic_symbol, cluster_radius=3):
    """
    Extracts a cluster of atoms around the first occurrence of the specified atomic symbol in a crystal structure.
    The central atom (specified by atomic_symbol) will be the first in the returned arrays.

    Parameters:
    - cif_file (str): Path to the CIF file containing the crystal structure.
    - atomic_symbol (str): Atomic symbol (e.g., "Si", "O") of the atom to center the cluster around.
    - cluster_radius (float): Radius of the cluster in angstroms within which atoms will be selected.

    Returns:
    - coords (np.ndarray): A 2D array of shape (n, 3) where 'n' is the number of atoms in the cluster.
      Each row contains the x, y, z coordinates of the corresponding atom.
    - atomic_symbols (np.ndarray): A 1D array containing the atomic symbols (e.g., "H", "O") of the atoms in the cluster,
      with the central atom's symbol first.
    - atomic_numbers (np.ndarray): A 1D array containing the atomic numbers (Z) of the atoms in the cluster,
      with the central atom's atomic number first.
    """

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

    # Find all sites within the specified radius around the chosen atom
    sites = symmetrized_structure.get_sites_in_sphere(symmetrized_structure.cart_coords[chosen_atom_index],
                                                      cluster_radius)

    # Create a new structure consisting of only the atoms within the cluster
    cluster_structure = Structure.from_sites(sites)

    # Initialize lists to hold the coordinates, atomic symbols, and atomic numbers
    coords = []
    atomic_symbols = []
    atomic_numbers = []

    # Reverse the structure so that the central atom is the first one in the final list
    for site in reversed(cluster_structure):
        coords.append(site.coords)  # Append the coordinates of each atom
        atomic_symbols.append(site.specie.symbol)  # Append the atomic symbol
        atomic_numbers.append(site.specie.Z)  # Append the atomic number

    # Convert the lists to numpy arrays for easier manipulation and center the central atom at (0,0,0)
    coords = np.array(translate_coords(coords))
    atomic_symbols = np.array(atomic_symbols)
    atomic_numbers = np.array(atomic_numbers)

    # Return the coordinates, atomic symbols, and atomic numbers
    return coords, atomic_symbols, atomic_numbers


def crystalnn_extract_cluster(cif_file, atomic_symbol):
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
    cluster_sites = [symmetrized_structure[chosen_atom_index]]

    # Collect the neighbor sites
    for neighbor in neighbors:
        cluster_sites.append(symmetrized_structure[neighbor['site_index']])

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
    return coords, atomic_symbols, atomic_numbers


def crystalnn_extract_cluster(structure, atomic_symbol):
    """
    Extracts a cluster of atoms around the first occurrence of the specified atomic symbol in a crystal structure.
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
    cluster_sites = [symmetrized_structure[chosen_atom_index]]

    # Collect the neighbor sites
    for neighbor in neighbors:
        cluster_sites.append(symmetrized_structure[neighbor['site_index']])

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
    return coords, atomic_symbols, atomic_numbers


def voronoi_extract_cluster(cif_file, atomic_symbol):
    """
    Extracts a cluster of atoms around the first occurrence of the specified atomic symbol in a crystal structure from a CIF file.
    The central atom (specified by atomic_symbol) will be the first in the returned arrays. The cluster is based on Voronoi neighbors.

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

    # Initialize Voronoi nearest neighbors calculator
    voronoi_nn = VoronoiNN()

    # Get the Voronoi neighbors of the selected atom
    neighbors = voronoi_nn.get_nn_info(symmetrized_structure, chosen_atom_index)

    # Include the central atom in the cluster
    cluster_sites = [symmetrized_structure[chosen_atom_index]]

    # Collect the neighbor sites
    for neighbor in neighbors:
        cluster_sites.append(symmetrized_structure[neighbor['site_index']])

    # Initialize lists to hold the coordinates, atomic symbols, and atomic numbers
    coords = []
    atomic_symbols = []
    atomic_numbers = []

    # Append the central atom first, followed by its neighbors (no reversing this time)
    for site in cluster_sites:
        coords.append(site.coords)  # Append the coordinates of each atom
        atomic_symbols.append(site.specie.symbol)  # Append the atomic symbol
        atomic_numbers.append(site.specie.Z)  # Append the atomic number

    # Convert the lists to numpy arrays for easier manipulation and center the central atom at (0,0,0)
    coords = np.array(translate_coords(coords))
    atomic_symbols = np.array(atomic_symbols)
    atomic_numbers = np.array(atomic_numbers)

    # Return the coordinates, atomic symbols, and atomic numbers
    return coords, atomic_symbols, atomic_numbers


def voronoi_extract_cluster(structure, atomic_symbol):
    """
    Extracts a cluster of atoms around the first occurrence of the specified atomic symbol in a crystal structure.
    The central atom (specified by atomic_symbol) will be the first in the returned arrays. The cluster is based on Voronoi neighbors.

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

    # Initialize Voronoi nearest neighbors calculator
    voronoi_nn = VoronoiNN()

    # Get the Voronoi neighbors of the selected atom
    neighbors = voronoi_nn.get_nn_info(symmetrized_structure, chosen_atom_index)

    # Include the central atom in the cluster
    cluster_sites = [symmetrized_structure[chosen_atom_index]]

    # Collect the neighbor sites
    for neighbor in neighbors:
        cluster_sites.append(symmetrized_structure[neighbor['site_index']])

    # Initialize lists to hold the coordinates, atomic symbols, and atomic numbers
    coords = []
    atomic_symbols = []
    atomic_numbers = []

    # Append the central atom first, followed by its neighbors (no reversing this time)
    for site in cluster_sites:
        coords.append(site.coords)  # Append the coordinates of each atom
        atomic_symbols.append(site.specie.symbol)  # Append the atomic symbol
        atomic_numbers.append(site.specie.Z)  # Append the atomic number

    # Convert the lists to numpy arrays for easier manipulation and center the central atom at (0,0,0)
    coords = np.array(translate_coords(coords))
    atomic_symbols = np.array(atomic_symbols)
    atomic_numbers = np.array(atomic_numbers)

    # Return the coordinates, atomic symbols, and atomic numbers
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

    # Apply translation: subtract the translation vector from all coords
    translated_coords = coords - translation_vector

    return translated_coords


def detect_transition_metal(structure):
    """
    Detect the first occurrence of a transition metal in the given structure.
    
    Args:
        structure (Structure): A pymatgen Structure object representing the crystal.
    
    Returns:
        str: The symbol of the transition metal, or None if no transition metal is found.
    """
    for site in structure:
        if site.specie.symbol in transition_metals:
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
        transition_metal = detect_transition_metal(structure)
        
        if transition_metal is None:
            print(f"No transition metal found in {cif_file}. Skipping...")
            continue

        print(f"Processing {cif_file} for transition metal {transition_metal} (MP-ID: {mp_id})...")
        
        # Call the extract_crystalnn_cluster function
        coords, symbols, numbers = crystalnn_extract_cluster(file_path, transition_metal)
        
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
      and spherical harmonic parameters. Weighted by a factor of 1/r
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
        Ylm_sum = np.sum(
            calculate_real_sph_harm(order_m, degree_l, theta, phi)*1/r*1/atomic_numbers)

    # Compute without considering parity
    else:
        # Sum over all neighbors
        Ylm_sum = np.sum(
            np.abs(calculate_real_sph_harm(order_m, degree_l, theta, phi))*1/r*1/atomic_numbers)

    # Calculate the local bond order paramater
    local_bond_order_paramater = 1 / n_neighbors * Ylm_sum

    return local_bond_order_paramater.real


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


def calculate_steinhart_sum(file_path, degree_l):
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
    cluster_data = extract_cluster(file_path, 0, 5)

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


def compute_steinhart_vector(file_path, degree_l):
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
    cluster_data = extract_cluster(file_path, 0, 5)
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


def get_oxidation_state(possible_species, atom):
    """
    Given a list of possible species, this function returns the oxidation state of the specified atom.

    Args:
    - possible_species (list): A list of strings representing species with their oxidation states, e.g., ['O2-', 'V5+', 'Cu+'].
    - atom (str): The symbol of the atom for which to retrieve the oxidation state, e.g., 'V'.

    Returns:
    - int: The oxidation state of the atom, or None if not found.
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
                    oxidation_state = 1
                else:
                    oxidation_state = int(oxidation_state_str[:-1])  # Positive oxidation state
            elif oxidation_state_str.endswith('-'):
                # If no number, assume it's -1
                if oxidation_state_str[:-1] == '':
                    oxidation_state = -1
                else:
                    oxidation_state = -int(oxidation_state_str[:-1])  # Negative oxidation state

            # Handle cases like '+2' or '-3' (sign at the start)
            elif oxidation_state_str.startswith('+'):
                # If no number, assume it's 1
                if oxidation_state_str[1:] == '':
                    oxidation_state = 1
                else:
                    oxidation_state = int(oxidation_state_str[1:])  # Positive oxidation state
            elif oxidation_state_str.startswith('-'):
                # If no number, assume it's -1
                if oxidation_state_str[1:] == '':
                    oxidation_state = -1
                else:
                    oxidation_state = int(oxidation_state_str)  # Negative oxidation state

            # If no sign is provided, assume positive oxidation state
            else:
                oxidation_state = int(oxidation_state_str)

            return oxidation_state

    # If no matching species is found, return None
    return None


def get_cluster_properties(mp_id, central_atom, api_key="VS3hLdCF3oL9aiuzPSH03BXjW5QNSmBj"):
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


def quadrupole_moment_normalized(positions, charges, atomic_numbers):
    """
    Calculate the non traceless form of the quadrupole moment tensor for a system of point charges, normalized by the atomic number.

    Args:
    - positions: Nx3 array, where N is the number of particles, and each row is the (x, y, z) coordinates of a particle.
    - charges: 1D array of length N, where each element is the charge of the corresponding particle.
    - atomic_numbers: 1D array of length N, containing the atomic number of each particle.

    Returns:
    - Q: 3x3 numpy array representing the normalized quadrupole moment tensor.
    """
    Q = np.zeros(
        # Initialize the quadrupole moment tensor as a 3x3 zero matrix.
        (3, 3))

    # Loop through each position, charge, and atomic number
    for pos, charge, atomic_number in zip(positions, charges, atomic_numbers):
        r_x, r_y, r_z = pos

        # Update the Q matrix using the normalized formula.
        normalization_factor = charge / atomic_number / (r_x**2 + r_y**2 + r_z**2)**2

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


def get_charges(possible_species, atomic_symbols):
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

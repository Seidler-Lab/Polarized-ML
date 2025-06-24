import pprint
import numpy as np
from pathlib import Path
from collections import namedtuple, defaultdict

def create_file_tuple(cif_file_path, data_df, feff_log1_pairs, json_dictionary, file_status):
    """
    Creates a namedTuple(File_Tuple) object. Intended to be created during the file gathering step.

    ARGS:
    cif_file_path: Path object to the cif file for this calculation
    reference_df: pandas dataframe of the reference spcetra
    data_df: pandas dataframe of the data spectra
    file_status: status of the gathering file section of the process

    RETURNS:
    namedTuple(File_Tuple)
    """
    file_tuple = namedtuple("File_Tuple", ["CIF_File_Path", "Data_DF", "Feff_and_log1_file_list", "JSON_Dictionary", "File_Status"])

    return file_tuple(cif_file_path, data_df, feff_log1_pairs, json_dictionary, file_status)

# def create_interpolation_tuple(interpolated_df, interpolation_status):
#     """
#     Creates a namedTuple(Interpolation_Tuple). Intended to be created during the interpolation step
    
#     ARGS:
#     interpolated_df: pandas dataframe of the interpolated data that is created
#         between our reference_df and our data_df
#     interpolation_status: status of the interpolation process
#         True: passed all checks
#         str: error message of some sort (hopefully)
#         other: something is not being passed correctly

#     RETURNS:
#     namedTuple(Interpolation_Tuple)
#     """
#     interpolation_tuple = namedtuple("Interpolation_Tuple", ["Interpolated_DF", "Interpolation_Status"])

#     return interpolation_tuple(interpolated_df, interpolation_status)

def create_statistical_measurement_tuple(absolute_difference_score, anisotropy_matrix=None, anisotropy_matrix_total=None, absolute_difference_status=None):
    """
    Creates a namedTuple(Statistical_Measurement_Tuple) object

    ARGS:
    absolute_difference_score: after doing the absolute difference, provides a score of the sum of the three
        upper diagonal numbers. A different part of the process will have us set the threshold as we see fit,
        this is just to demonstrate that it has been through the process

    anisotropy_matrix: 3x3 matrix

    anisotropy_matrix_total: float. total of the upper three diagonal elements

    absolute_difference_status: status of the absolute difference part of the process
        True: the score has pass vs the statistical measurement we are taking
        str: a description of the failure
        other: something unexpected
    
    RETURNS:
    namedTuple(Statistical_Measurement_tuple)
    """
    statistical_measurement_tuple = namedtuple("Statistical_Measurement_Tuple", ["Absolute_Difference_Score", "Anisotropy_Matrix", "Anisotropy_Matrix_Toal", "Absolute_Difference_Status"])
    
    return statistical_measurement_tuple(absolute_difference_score, anisotropy_matrix, anisotropy_matrix_total, absolute_difference_status)

def create_full_calculation_tuple(file_tuple:tuple = None, interpolation_tuple:tuple = None, statistical_measurement_tuple:tuple = None):
    """
    Intended to be used at the end of each of our steps. In this way we can show which calculations fail at what points
    We set the default input to be None to indicate that point has not been reached yet.
    At the end, we check for all namedTuples statuses, then send to the anisotropy matrix calculation.
    
    ARGS:
    file_tuple: namedTuple(File_Tuple)
    interpolation_tuple: namedTuple(Interpolation_Tuple)
    statistical_measurement_tuple: namedTuple(Statistical_Measurement_Tuple)
    
    RETURNS:
    namedTuple(Calculation Tuple)
    """
    calculation_tuple = namedtuple("Calculation_Tuple", ("File_tuple", "Interpolation_tuple", "Statistical_measurement_tuple"))

    return calculation_tuple(file_tuple, interpolation_tuple, statistical_measurement_tuple)

def concatenated_statuses(*args):
    """
    Whenever we get to a new phase of the sorting, we check to see if any of the tuples returned
    statuses that are not True.
    If this is the case, we take them out of the good_file_list, and place them into the bad_file_list.

    ARGS:
    good_file_list: list of a tuple of tuples, we unpack them, add a new tuple.
        if the new tuple does not have a True value in the status, we throw that shit out.

    bad_file_list: The trash can we carry through the process.
    
    *args: takes any number of positional arguments as statuses from various 
        error checking steps

    RETURNS:
    returns True if all positional arguments evaluate to True

    returns a concatenated string of all the error messages that are created
    """
    try:
        if all(args):
            return True
    
        else:
            error = [s for s in (args) if isinstance(s, str) or isinstance(s,False) or isinstance(s,None)]
            return error
    
    except Exception as e:
        print(e)

def create_array_from_file_after_target_string(file_path:Path, target:str)-> np.array:
    """
    Reads a file until a certain string.  
    ARGS:
    file_path: Path-like object
    target: string to start the reading from

    RETURNS:
    potentials_array: A mapping of the potentials into an array
    """
    # print("This is the file we are trying to turn contents into an array: ", file_path.name)
    # print(f"We are going to look under {target} part of the file and started reading its contents")

    block = []
    record = False

    with open(file_path, 'r') as f:
        for line in f:
            if not record:
                if target in line:
                    record = True
                continue

            stripped = line.strip()
            
            if stripped == "" or not (stripped[0].isdigit() or stripped[0] == '-' or stripped[0] == '.'):
                break

            parts = stripped.split()
            block.append([x for x in parts])

    return block

def insert_matrix_to_json(data, key:str, matrix):
    """
    Inserts a 3x3 matrix (as NumPy array or list) into a dictionary under the given key.

    Args:
        data (dict): The dictionary to update.
        key (str): The key to assign the matrix to.
        matrix (array-like): A 3x3 matrix as a NumPy array or list.

    Returns:
        dict: The updated dictionary.
    """
    matrix = np.asarray(matrix)
    
    if matrix.shape != (3, 3):
        raise ValueError("Matrix must be 3x3.")
    
    data[key] = matrix.tolist()
    return data

def turn_list_into_dictionary(list_of_lists, index_of_inner_lists_for_keys, index_of_inner_lists_for_values):
    """ 
    Given two indices of a list, turns into a dictionary. The first argument will be the key
    and the second argument will be the value of that key.

    args: 
    key_of_dict_from_list_index: list index that will be the key 
    value_of_key_from_list_index: list index that will be the value 
    """
    default_dict = defaultdict()

    for list in list_of_lists:
        default_dict[list[index_of_inner_lists_for_keys]] = list[index_of_inner_lists_for_values]

    return default_dict

def generate_feff_json_dict(log1_array, feff_inp_array, feff_cluster_array):
    """
    Generates a JSON-serializable dictionary containing:
      - Average charge transfer per element number
      - Atom coordinates with associated element numbers
    """
    # Step 1: Map potential number → charge from log1_array
    potential_to_charge = {entry[0]: float(entry[1]) for entry in log1_array}

    # Step 2: Map potential number → element number from feff_inp_array
    potential_to_element = {entry[0]: entry[1] for entry in feff_inp_array}

    # Step 3: Build FEFF Charges dictionary: element number → list of charges
    element_to_charges = defaultdict(list)
    for potential, charge in potential_to_charge.items():
        element = potential_to_element.get(potential)
        if element:
            element_to_charges[element].append(charge)

    # Step 4: Take the mean of each element's charges
    feff_charges = {element: sum(charges)/len(charges) for element, charges in element_to_charges.items()}

    # Step 5: Create Atoms Coordinates list: [x, y, z, element_number]
    atom_coordinates = []
    for atom in feff_cluster_array:
        x, y, z, potential = atom
        element = potential_to_element.get(potential)
        if element:
            atom_coordinates.append([x, y, z, element])

    # Combine into final dictionary
    return {
        "FEFF Charges": feff_charges,
        "Atoms Coordinates": atom_coordinates
    }

def create_charges_json_dictionary_from_feff_and_log1_arrays(log1_dat_file, feff_inp_file):
    """
    For a single file_tuple from a good_calculation_tuple, reads both the 
    feff.inp and log1_dat to get their charges. Uses other functions to read in
    parts of their individual files to arrays and compares the two to 

    ARGS:
    good_calculation_tuple: tuple that has passed all of the tests in the pipeline

    RETURNS:
    charges_json_dictionary: a json dictionary with the mp-id_absorbing_atom as the
    primary key, with a dictionary of unique atoms with values of the means of their
    charges, not counting the absorber atom
    """

    # print(f"This is the log1 path \n {log1_dat_file}, \n and this is the feff_inp path \n {feff_inp_file}")

    df_log1_dat = create_array_from_file_after_target_string(log1_dat_file, target='Charge transfer:  type  charge')
    df_feff_inp = create_array_from_file_after_target_string(feff_inp_file, target='POTENTIALS')
    df_feff_cluster = create_array_from_file_after_target_string(feff_inp_file, target='ATOMS')

    # print(f"This is the log1_array \n {df_log1_dat}, \n This is the feff_inp_array \n {df_feff_inp}, \n This the feff_cluster_arra {df_feff_cluster}")
    if not df_feff_inp or not df_log1_dat or not df_feff_cluster:
        status = "Either the log1.dat or the feff.inp did not have the proper string, and thus did not populate"
        # create_file_tuple(cif_file_path, data_df, log1_dat_file, feff_inp_file, status)
        return {}
    else:
        # charge_dictionary = calculate_mean_charges(df_log1_dat, df_feff_inp)
        json_dict_for_cluster = generate_feff_json_dict(df_log1_dat, df_feff_inp,df_feff_cluster)
        return json_dict_for_cluster


def read_corvus_cfavg_xes_out_file_to_numpy(corvus_cfavg_xes_out_file_path: Path):
    """
    Takes a Path object of a Corvus.cfavg.xes.out file and turns it into numpy arrays.
    Makes the Energy column as the first array. Does an error check to see if the isotropic
    column is average of the xpolarization, ypolarization, and zpolarization columns (as expected).

    ARGS: 
    corvus_cfavg_xes_out_file_path: Path object

    RETURN:
    dict of numpy arrays with keys: 'Energy', 'x_polarization', 'y_polarization', 'z_polarization', 'Isotropic'
    """

    print(f"Processing the Corvus.cfavg.xes.out file to numpy arrays {corvus_cfavg_xes_out_file_path.parent.name}")
    if corvus_cfavg_xes_out_file_path is None:
        print("THIS IS NONE THIS IS NONE THIS IS NONE THIS IS NONE THIS IS NONE THIS IS NONE THIS IS NONE")
        return None
    
    # Load data skipping the header (if any), assume whitespace delimiter
    # The file has 5 columns: Energy, x, y, z, Isotropic
    data = np.loadtxt(corvus_cfavg_xes_out_file_path)

    # Columns by index:
    # 0: Energy
    # 1: x_polarization
    # 2: y_polarization
    # 3: z_polarization
    # 4: Isotropic

    energy = data[:, 0]
    x_polarization = data[:, 1]
    y_polarization = data[:, 2]
    z_polarization = data[:, 3]
    isotropic = data[:, 4]

    polarization_mean = (x_polarization + y_polarization + z_polarization) / 3.0

    if not np.allclose(polarization_mean, isotropic, atol=1e-6):
        raise ValueError("The three polarization columns do not average out to be the Isotropic column.")

    print("This corvus.cfavg.xes.out data was successfully loaded:", corvus_cfavg_xes_out_file_path.parent.name)

    return {
        "Energy": energy,
        "x_polarization": x_polarization,
        "y_polarization": y_polarization,
        "z_polarization": z_polarization,
        "Isotropic": isotropic,
    }

def find_pertinent_files_from_calc_directory(parent_dir:str, absorbing_element:str):
    """
    Looks inside of the parent_dir variable declared at the top of this
    jupyter notebook and globs all of the pertinent files (Corvus.cfavg.xes.out, .cif)

    ARGS:
    parent_dir: String found at top of this notebook, is the single point access for everything
    absorbing_element: String found at top of this notebook, describes the absorbing atom that we
    want to get from the data set

    RETURNS:
    two list of tuples to be unpacked for a calculation tuple. 
    Will return a good tuple list that is tuples (Path object, Dataframe, status=None), 
    and a bad tuple list that is tuples (unknown, unknown, status = False or 'reference) 
    """

    list_of_matching_directories = list(Path(parent_dir).rglob(f"*_{absorbing_element}"))
    total_number = len(list_of_matching_directories)
    print("This is what we will report as the total amount of calculations: ", total_number)

    good_file_tuple_list = []
    bad_file_tuple_list = []

    for directory_match in list_of_matching_directories:

        data_spectra_path = None
        data_spectra_df = None
        cif_path = None
        log1_dat_file_path = None
        feff_inp_file_path = None
        json_dictionary = f"{directory_match}/{directory_match.name}.json"
        print("This is the json dictionary path in file_reader: ", json_dictionary)
        status = None
        
        # if any(str(directory_match).endswith(f"{reference}_{absorbing_element}") for reference in reference_cifs_dictionary.values()):
        #     print("This is one of the reference files for another element. Skipping", directory_match)
        #     status = 'reference for other material'
        #     file_tuple = create_file_tuple(directory_match, reference_df, data_spectra_df, log1_dat_file_path, feff_inp_file_path, status)
        #     bad_file_tuple_list.append(create_full_calculation_tuple(file_tuple = file_tuple))
        #     continue

        try:
            feff_log1_pairs = []
            for path in directory_match.rglob("*"):

                if path.suffix == ".cif":
                    cif_path = path
                    print("This is the path to the .cif file", path.parent)

                if path.name == "Corvus.cfavg.out":
                    data_spectra_path = path
                    print(data_spectra_path)
                    print(type(data_spectra_path))
                    
                    print("This is the path to the Corvus.cfavg.xes.out file", path.parent)

                if path.name == "log1.dat":
                    corresponding_feff_path = path.parent / 'feff.inp'
                    print("This is the path to the log1.dat file", path.parent)
                    print("This is the corresponding path to the feff.inp for this log1.dat file: ", corresponding_feff_path)
                    if corresponding_feff_path.exists():
                        feff_log1_pairs.append((path, corresponding_feff_path))

                # if path.name == "feff.inp":

                #     print("This is the path to the feff.inp file", path.parent)
                #     feff_inp_file_path = path

            if data_spectra_path is not None:
                    data_spectra_df = read_corvus_cfavg_xes_out_file_to_numpy(data_spectra_path)
            
            if any(x is None for x in [cif_path, data_spectra_df, feff_log1_pairs]):
                status = "One or more required files missing"
                file_tuple = create_file_tuple(cif_path, data_spectra_df, feff_log1_pairs, json_dictionary, status)
                bad_file_tuple_list.append(create_full_calculation_tuple(file_tuple=file_tuple))
                continue
 

            status = True
            
            #print(data_spectra_df.shape)
            file_tuple = create_file_tuple(cif_path, data_spectra_df, feff_log1_pairs, json_dictionary, status)
            good_file_tuple_list.append(create_full_calculation_tuple(file_tuple = file_tuple))
        
        except Exception as e:
            print("An error occured", e)
            cif_path = cif_path 
            data_spectra_df = data_spectra_df
            log1_dat_file_path = log1_dat_file_path
            feff_inp_file_path = feff_inp_file_path
            status = f"An error has occured: {e}"
            file_tuple = create_file_tuple(cif_path, data_spectra_df, feff_log1_pairs, json_dictionary, status) 
            bad_file_tuple_list.append(create_full_calculation_tuple(file_tuple=file_tuple))
            
    print("This is the amount of good cif file and dataframe tuples that we created: ", len(good_file_tuple_list))
    print("This is the amount of BAD cif file and dataframe tuples that we created: ", len(bad_file_tuple_list))
    return good_file_tuple_list, bad_file_tuple_list, total_number

def good_calc_list_bad_calc_list_status(good_calculation_list:list[tuple], bad_calculation_list:list[tuple], total_number:int):
    """
    Print a status statement of the good list and the bad list.
    Also does a check on total conservation to make sure we aren't losing anything
    
    ARGS:
    good_calculation_list: List of tuples. These are all of the calculations that are passing the current step of the process
    bad_calculation_list: List of tuples. These are all of the calculations that have failed for one reason or another
    total number: int. Ensures total conservation to make sure we aren't leaving anything behind 
    
    RETURNS:
    statement, and warning that conservation is being fulfilled
    """

    statement = (
        print("#######################################"),
        print("############### STATUS ################"),
        print("#######################################"),
        print(f"Good List: {len(good_calculation_list)}"),
        print(f"Bad List: {len(bad_calculation_list)}"),
        print(f"Total number of calculations: {total_number}"),
        print("#######################################"),
        print("########### HAVE A SMILEY DAY #########"),
        print("#######################################")
    )

    if len(good_calculation_list) + len(bad_calculation_list) != total_number:
        print("WARNING WARNING WARNING: We are losing calculations somewhere")

    return statement

def sorting_good_calc_and_bad_calc_lists(good_calculation_list:list[tuple], bad_calculation_list:list[tuple], total_number:int):
    """
    Goes through the good calculation list, if all of the tuples that exist
    inside have tuples with a True in its [-1] (where we want the status of every tuple),
    then it will stay inside of the good list. If there are tuples inside that have a 
    status that is other than True, it is placed into the bad calculation list.

    ARGS:
    good_calculation_list, bad_calculation_list: list of tuples of tuples
    total_number: int, total of both lists
    RETURNS:
    """

    for current_tuple in good_calculation_list:
        
        if all(
            isinstance(inner_tuple, tuple) and inner_tuple[-1] is True 
            for inner_tuple in current_tuple
            if inner_tuple is not None):
            continue 
            #print("All tuples with this calculation_tuple have True status in their last entry")           
            #print(current_tuple)

        else:
            #print(current_tuple)
            good_calculation_list.remove(current_tuple)
            bad_calculation_list.append(current_tuple)
            #print("Some tuples do not have a True status, taking to the trash")
        
    good_calc_list_bad_calc_list_status(good_calculation_list, bad_calculation_list, total_number)
    
    return good_calculation_list, bad_calculation_list 

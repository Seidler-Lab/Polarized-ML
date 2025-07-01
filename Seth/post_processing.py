import file_reader_n_json_creater as file
import pprint
#import glob
from collections import defaultdict
from pathlib import Path
import numpy as np
import json
import sys

TARGET_DIRECTORY = Path(sys.argv[1]).absolute()
ABSORBING_ATOM = sys.argv[2]

reference_cifs_dictionary = {
 "Sc": "mp-644481", 
 "Ti": "mp-1215", 
 "V": "mp-18937", 
 "Cr": "mp-19177", 
 "Mn": "mp-510408",
 "Fe": "mp-19770", 
 "Co": "mp-22408", 
 "Ni": "mp-19009", 
 "Cu": "mp-704645", 
 "Zn": "mp-2133" 
  }



def file_contains(file_path:Path, target:str)->bool:
    """
    Checks to see if given file_path has target string

    ARGS:
    file_path: Path-like object
    target: string to look for in file_path

    RETURNS:
    True or False depending on whether it is in the file or not
    """
    with open(file_path, 'r') as f:
        return any(target in line for line in f)

def try_float(x):
    """
    Attempts to make an entry a float. Otherwise handles the error and returns
    the input as is
    
    ARGS:
    x: anything
    RETURNS:
    -x:float if it can be turned into one
    -x:stays the same if it cant be converted
    """
    try:
        return float(x)
    
    except ValueError:
        return x

def calculate_mean_charges(df_log1, df_feff_inp)->dict:
    """
    Matches row by index in column zero of both data frames, and adds element symbols
    as keys to a dictionary, and then averages the charges of all the keys

    ARGS:
    df_log1: the dataframe created from parsing the calculation's log1.dat file
    feff_inp_df: the dataframe created from parsing the calculation's feff.inp file

    RETURNS:
    dictionary: dictionary of the calculation's unique elements and the average of their charges
    """

    element_charges = defaultdict(list)

    df_log1[0] = df_log1[0].astype(float)
    df_feff_inp[0] = df_feff_inp[0].astype(float)

    for _, (idx, charge) in df_log1[[0,1]].iterrows():
        match = df_feff_inp[df_feff_inp[0] == idx]
        if not match.empty:
            element = match.iloc[0][2]
            element_charges[element].append(charge)

    return {el: np.mean(vals) for el, vals in element_charges.items()}


def xes_average(data_1, data_2, data_3):
    '''
    Calculates the integral of the average of 3 orthogonally polarized XES spectra.

    Parameters:
    data_1 (pandas.DataFrame): The DataFrame containing the X-ray absorption data for polarization 1.
    data_2 (pandas.DataFrame): The DataFrame containing the X-ray absorption data for polarization 2.
    data_3 (pandas.DataFrame): The DataFrame containing the X-ray absorption data for polarization 3.

    Return:
    integrated_average (float): The integral of the average of the 3 spectra.
    '''
    # Calculate the average of Delta mu(E) values
    average_delta_muE = (data_1 + data_2 + data_3) / 3

    # Integrate the average Delta mu(E) over the energy range
    integrated_average = np.sum(average_delta_muE)

    return integrated_average


def anisotropy_parameter(xes_difference, xes_average):
    '''
    Calculate the anisotropy parameter, which is the quotient of the XES difference and the XES average.

    Parameters:
    xes_difference (float): The integrated absolute difference of Δμ(E).
    xes_average (float): The integral of the average of the 3 spectra.

    Returns:
    float: The anisotropy parameter.
    '''
    if xes_average == 0:
        raise ValueError(
            "The xes_average must not be zero to avoid division by zero.")

    return xes_difference / xes_average

def xes_integrated_abs_difference(data_1, data_2):
    ''' 
    Calculate the integrated absolute difference of Δμ(E) for data_1 and data_2.

    Data 1 and 2 have orthogonal polarizations.

    Parameters:
    data_1 (pandas.DataFrame): The DataFrame containing the X-ray absorption data for polarization 1.
    data_2 (pandas.DataFrame): The DataFrame containing the X-ray absorption data for polarization 2.

    Returns:
    difference (float): The integrated absolute difference of Δμ(E) between the two datasets.
    '''
    # Calculate the absolute difference of Delta mu(E)
    integrated_abs_difference = np.sum(np.abs(data_1 - data_2)**2)**(1/2)

    # Integrate the absolute difference over the energy range
    #integrated_abs_difference = np.sum(abs_difference)

    return integrated_abs_difference

def anisotropy_matrix(data_x, data_y, data_z):
    '''
    Calculate a 3x3 anisotropy matrix where each entry represents the anisotropy parameter 
    for the difference between two datasets divided by the average of all three datasets.

    Parameters:
    data_x (pandas.DataFrame): The DataFrame containing the X-ray absorption data for polarization x.
    data_y (pandas.DataFrame): The DataFrame containing the X-ray absorption data for polarization y.
    data_z (pandas.DataFrame): The DataFrame containing the X-ray absorption data for polarization z.

    Returns:
    numpy.ndarray: A 3x3 anisotropy matrix.
    '''
    # Calculate the XES average of all three datasets
    xes_avg = xes_average(data_x, data_y, data_z)

    # Initialize a 3x3 matrix
    anisotropy_mat = np.zeros((3, 3))

    # Define the pairs for which to calculate the differences
    diff_1 = xes_integrated_abs_difference(data_x, data_y)
    diff_2 = xes_integrated_abs_difference(data_x, data_z)
    diff_3 = xes_integrated_abs_difference(data_y, data_z)

    anisotropy_mat[0][0] = 0
    anisotropy_mat[0][1] = anisotropy_parameter(diff_1, xes_avg)
    anisotropy_mat[0][2] = anisotropy_parameter(diff_2, xes_avg)

    anisotropy_mat[1][0] = anisotropy_parameter(diff_1, xes_avg)
    anisotropy_mat[1][1] = 0
    anisotropy_mat[1][2] = anisotropy_parameter(diff_3, xes_avg)

    anisotropy_mat[2][0] = anisotropy_parameter(diff_2, xes_avg)
    anisotropy_mat[2][1] = anisotropy_parameter(diff_3, xes_avg)
    anisotropy_mat[2][2] = 0


    # Fill the anisotropy matrix with the anisotropy parameters
    #for i, (data1, data2) in enumerate(pairs):
    #    diff = xes_integrated_abs_difference(data1, data2)
    #    anisotropy_mat[i][(i+1) % 3] = anisotropy_parameter(diff, xes_avg)
    #    anisotropy_mat[(i+1) % 3][i] = anisotropy_mat[i][(i+1) % 3]  # Symmetric entries 

    return anisotropy_mat

def main(TARGET_DIRECTORY, ABSORBING_ATOM):
    """ This is the main function of the post_processing module"""

    good_calculation_tuple_list, bad_tuple_list, _ = file.find_pertinent_files_from_calc_directory(TARGET_DIRECTORY, ABSORBING_ATOM)
    # print("This is the good file_tuple list ", good_calculation_tuple_list)
    print("This is the bad tuples list")
    #for tuple in bad_tuple_list:
    #    pprint.pprint(tuple)

    for good_calculation_tuple in good_calculation_tuple_list:
        # print("This is a tuple from good file tuple", good_calculation_tuple)
        # print("This is the length of the tuple ", len(good_calculation_tuple))
        file_tuple, _, _ = good_calculation_tuple
        _, corvus_cfavg_file, feff_log1_pairs, json_dictionary, _ = file_tuple

        #spectral anisotropy matrix
        #corvus_cfavg_xes_out_dictionary = file.read_corvus_cfavg_xes_out_file_to_numpy(corvus_cfavg_file)  
        print("Starting the anisotropy matrix")     
        anisotropy_mat = anisotropy_matrix(corvus_cfavg_file['x_polarization'],
                        corvus_cfavg_file['y_polarization'],
                        corvus_cfavg_file['z_polarization'])
        
        # Load the existing JSON file
        with open(json_dictionary, 'r') as f:
            data = json.load(f)
        
        print("This is the json we are trying to load into: ", data)

        # Insert matrix into the dictionary
        data = file.insert_matrix_to_json(data, 'Avg Spectral Anisotropy Matrix', anisotropy_mat)

        #feff charges
        print("Starting the writing of the FEFF Charges for: ", good_calculation_tuple)
        for i, (log1_dat_file, feff_inp_file) in enumerate(feff_log1_pairs, start=1):
            print("This is the file pair: ", feff_log1_pairs)
            print("This is the log1 dat file: ", log1_dat_file)
            print("This is the feff.inp file ", feff_inp_file)

            
            cluster_key = f"{ABSORBING_ATOM}{i}"
            print("This is the cluster key: ", cluster_key)

            charge_dictionary = file.create_charges_json_dictionary_from_feff_and_log1_arrays(log1_dat_file, feff_inp_file)
            # pprint.pprint("This is the charge dictionary coming from creating the charges: ", charge_dictionary)

            data[cluster_key] = charge_dictionary

        # Write the updated dict back to the JSON file
        with open(json_dictionary, 'w') as f:
            json.dump(data, f, indent=4)


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: python Full_Polarization.py <target_directory> <absorbing_atom>")
        sys.exit(1)
    
    main(TARGET_DIRECTORY, ABSORBING_ATOM)
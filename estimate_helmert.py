##################
# --- HEADER --- #
##################

__author__ = "Mike Halbheer"
__version__ = "alpha.0.7"
__status__ = "Development"


###################
# --- IMPORTS --- #
###################

import numpy as np
from numpy.linalg import inv
import os
import sys


#####################
# --- FUNCTIONS --- #
#####################

def main(input_path):

    ### Load the coordinates generated in AutoCAD
    coords = np.loadtxt(input_path, delimiter=',', usecols=(1, 2, 3))
    
    ## Seperate the coordinates into input and reference coodinate pairs
    coords_input = coords[coords[:, -1] == 0, :-1]
    coords_ref = coords[coords[:, -1] == 1, :-1]

    ## Load scale if it was defined
    scale = coords[coords[:, -1] == 2, 0][0]

    # If scale is known apply it ahead of time to remove it from parameter estimation
    if not np.isnan(scale):
        estimate_scale = False
        coords_input *= scale
    else:
        estimate_scale = True

    # Helmert transformation with center of gravity
    center_point = get_zero_shift(coords_input)
    
    # Reduce coordinates
    coords_input = coords_input - center_point
    coords_ref = coords_ref - center_point

    ### Define Parameter estimation

    ## Create observation vector
    l = coords_ref.ravel()

    ## Create A matrix
    A = np.zeros((coords_input.shape[0] * 2, 4))
    A[::2, 0] = 1
    A[1::2, 1] = 1
    A[:, 2] = coords_input.ravel()

    coords_input_a = coords_input[:, [1, 0]]
    coords_input_a[:, 1]= coords_input_a[:, 1] * -1
    
    A[:, 3] = coords_input_a.ravel()

    # Estimate parameters
    x = inv(A.T @ A) @ A.T @ l

    # Extract relevant parameters from the x vector
    translation = x[:2]

    phi = np.arctan2(x[3], x[2])

    m = np.sqrt(np.sum(x[2:]**2)) if estimate_scale else scale

    # Calculate the residuals and sigma a posteriori
    residuals = A @ x - l
    sigma_posteriori = np.sqrt((residuals.T @ residuals) / (l.shape[0] - 4))
    K_xx = sigma_posteriori**2 * inv(A.T @ A)
    sigma_x = np.diagonal(K_xx)

    ## Log the transformation
    logfile_path = os.path.join(os.path.dirname(input_path), os.path.splitext(os.path.basename(input_path))[0][:-7] + '_log.txt')

    with open(logfile_path, 'a') as logfile:
        logfile.write('\nHelmerttransformation\n=====================\n')
        logfile.write('Transformationsparameter\n----------------------\n')
        logfile.write('Alle Transformationsparameter verstehen sich in einem geodätischen Koordinatensystem.\n')
        logfile.write('Rotationsrichtung ist mathematisch positiv in einem geodätischen Koordinatensystem.\n')
        logfile.write(f'Drehpunkt (y, x)\t{center_point[0]}\t{center_point[1]}\n')
        logfile.write(f'Translation (y, x)\t{translation[0]}\t{translation[1]}\n')
        logfile.write(f'Rotation (rad, gon, deg)\t{phi}\t{phi * 200 / np.pi}\t{phi * 180 / np.pi}\n')
        logfile.write(f'Massstab\t{m}\n')

        logfile.write('\nResiduen\n--------\n')
        logfile.write('Punktnummer\tY\tX\n')
        for i in range(0, coords_input.shape[0]):
            logfile.write(f'{i}\t{residuals[2*i] * 1000:.1f}mm\t{residuals[2*i+1] * 1000:.1f}mm\n')

        logfile.write('\nStandardabweichungen der Parameter\n----------------------------------\n')
        logfile.write(f't_y\t{sigma_x[0]*1000:.1f}mm\n')
        logfile.write(f't_x\t{sigma_x[1]*1000:.1f}mm\n')
        logfile.write(f'm\t{sigma_x[2]:.1f}\n')
        logfile.write(f'mgon\t{sigma_x[3]*200 / np.pi:.1f}mgon\n')

    ## Create a CSV file to pass transformation parameters to AutoCAD
    trafo_path = os.path.join(os.path.dirname(input_path), os.path.splitext(os.path.basename(input_path))[0][:-7] + '_trafo.txt')

    # Define the parameter array
    trafo_array = np.append(translation, np.array([phi, m, center_point[0], center_point[1]]))

    # Dump the parameters to the CSV file
    np.savetxt(trafo_path, trafo_array[np.newaxis, ...], delimiter=',')

def get_zero_shift(coords:np.ndarray) -> np.ndarray:
    '''
    Function to calculate the center point of the coordinates

    Parameters
    ----------
    coords : numpy Array
        The coordinates to get the center for

    Returns
    -------
    center_point : numpy Array
        The center point of the coordinates
    '''

    center_point = np.average(coords, axis=0)

    return center_point

if __name__ == '__main__':
    main(sys.argv[1])
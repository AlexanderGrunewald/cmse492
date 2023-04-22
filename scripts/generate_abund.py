import numpy as np
import pandas as pd
import csv
import sys
import os
from typing import List, Dict, Tuple, Union, Optional, Callable
from scipy.stats import gamma as gamma_dist



def sample_dirichlet2(cube: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """
    Samples from a Dirichlet distribution with given alpha parameters.

    Parameters
    ----------
    cube : numpy.ndarray
        A 1D array of random numbers from a uniform distribution, used for sampling.
    alphas : numpy.ndarray
        A 1D array of alpha parameters for the Dirichlet distribution.
        Each alpha value controls the concentration of the corresponding output sample.

    Returns
    -------
    numpy.ndarray
        A 1D array of random numbers from the Dirichlet distribution,
        where each value represents the probability of the corresponding output category.

    Examples
    --------
    >>> sample_dirichlet2(np.random.rand(8), np.array([1, 2, 3, 4, 5, 6, 7, 8]))
    array([0.10360234, 0.06044682, 0.04772003, 0.14713447, 0.13287672,
           0.10051183, 0.07499862, 0.13853791, 0.09686599, 0.09728427])
    """
    y = gamma_dist.ppf(cube, alphas)
    x = y / y.sum()
    return x


def generate_distribution(n: int, alphas: np.ndarray, names: list) -> None:
    """
    Generate a csv file with n rows and 8 columns of random numbers from a Dirichlet distribution.
    Parameters
    ----------
    n : int
        Number of rows in the csv file.
    alphas : np.ndarray
        A 1D array of alpha parameters for the Dirichlet distribution.
    names : list
        A list of names for the columns of the csv file.
    Returns
    -------
    None

    Examples
    --------
    >>> generate_distribution(100, np.array([1, 2, 3, 4, 5, 6, 7, 8]), ['O', 'Si', 'S', 'Mg', 'Ca', 'Ti+Cr', 'Fe', 'Ni56'])
    """
    data = pd.DataFrame(columns=names)
    for i in range(n):
        data.loc[i] = sample_dirichlet2(np.random.rand(8), alphas)
    # dump data to csv file
    # if there is no folder named data_predictor, create one
    if not os.path.exists('/mnt/home/grunew14/Documents/project/data_predictor'):
        os.makedirs('/mnt/home/grunew14/Documents/project/data_predictor')

    if not os.path.exists('/mnt/home/grunew14/Documents/project/data_predictor/abundances.csv'):
        data.to_csv('/mnt/home/grunew14/Documents/project/data_predictor/abundances.csv', index=False)

def main(num_samples: int):
    names = ['O', 'Si', 'S', 'Mg', 'Ca', 'Ti+Cr', 'Fe', 'Ni56']
    generate_distribution(num_samples, np.ones(8), names)

if __name__ == '__main__':
    num_samples = int(sys.argv[1])
    main(num_samples)
import numpy as np
import pandas as pd
import sys
import csv
import os
from typing import List, Dict, Tuple, Union, Optional, Callable
from scipy.stats import gamma as gamma_dist
from copy import deepcopy
import tardis
from tardis import run_tardis
from tardis.io.config_reader import Configuration


####################
# This script generates the data for the predictor and target
# The predictor is a csv file named abundances.csv with 100 rows and 8 columns
# The target is a csv file named targets{i}.csv with 2 rows and 1000 columns
# where i is the row number of the predictor
####################


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
    if not os.path.exists('data_predictor'):
        os.makedirs('data_predictor')

    if not os.path.exists('data_predictor/abundances.csv'):
        data.to_csv('data_predictor/abundances.csv', index=False)


def generate_config(abundance_samples: pd.Series) -> Configuration:
    """
    Generate a Configuration object with the specified abundances.

    Parameters
    ----------
    abundance_samples : pd.Series
        A Series of abundances in the following format:
            O       0.000000
            Si      0.000000
            S       0.000000
            Mg      0.000000
            Ca      0.000000
            Ti+Cr   0.000000
            Fe      0.000000
            Ni56    0.000000
        The index is the element name and the value is the abundance.
        The abundances should be in the same order as the `names` list in `generate_distribution`.
        The abundances should sum to 1.

    Returns
    -------
    Configuration
        A Configuration object with the abundances set to the specified values. The object is generated from the
        configuration template in `tardis_example.yml`. The abundances are set to the following:
        config.model.abundances =
            {
                'type': 'uniform',
                'O': abundance_samples['O'],
                'Si': abundance_samples['Si'],
                'S': abundance_samples['S'],
                'Mg': abundance_samples['Mg'],
                'Ca': abundance_samples['Ca'],
                'Ti': abundance_samples['Ti+Cr'] / 2,
                'Cr': abundance_samples['Ti+Cr'] / 2,
                'Fe': abundance_samples['Fe'],
                'Ni56': abundance_samples['Ni56']
            }
    """
    names = ['O', 'Si', 'S', 'Mg', 'Ca', 'Ti+Cr', 'Fe', 'Ni56']
    config_template = Configuration.from_yaml("tardis_example.yml")
    config = deepcopy(config_template)
    config.model.abundances = {'type': 'uniform'}
    abund = dict(zip(names, abundance_samples))
    abund['Ti'] = abund['Ti+Cr'] / 2
    abund['Cr'] = abund['Ti']
    del abund['Ti+Cr']
    config.model.abundances.update(abund)
    return config


def run_model(config: Configuration):
    """
    Run the model with the given Configuration object.
    Parameters
    ----------
    config : Configuration
        A Configuration object with the abundances set to
        config.model.abundances
    Returns
    -------
    tardis.model.Radial1DModel
        A Radial1DModel object.
    """
    return run_tardis(config, atom_data='kurucz_cd23_chianti_H_He.h5')


def main(row_number: int = 0)-> None:
    """
    Generates a csv file for each row in `data_predictor/abundances.csv` containing the model outputs for the
    corresponding abundance configuration, and saves them in the `data_target` directory.

    Parameters
    ----------
    num_samples : int, optional
        The number of rows to process from the `abundances.csv` file (default is 100).

    Returns
    -------
    None

    Examples
    --------
    To generate model outputs for 50 abundance configurations, run:
    >>> main(num_samples=50)
    """
    # generate_distribution(100, np.ones(8), ['O', 'Si', 'S', 'Mg', 'Ca', 'Ti+Cr', 'Fe', 'Ni56'])
    # return
    #names = ['O', 'Si', 'S', 'Mg', 'Ca', 'Ti+Cr', 'Fe', 'Ni56']
    #generate_distribution(num_samples, np.ones(8), names)

    # generate the targets by running the model and saving each output to a csv file named simulation{i}.csv
    # if there is no folder named data_target, create one
    if not os.path.exists('/mnt/home/grunew14/Documents/project/data_target'):
        os.makedirs('/mnt/home/grunew14/Documents/project/data_target')

    abundances = pd.read_csv('/mnt/home/grunew14/Documents/project/data_predictor/abundances.csv')

    row = abundances.iloc[row_number,:]
    config = generate_config(row)
    model = run_model(config)
    if not os.path.exists('/mnt/home/grunew14/Documents/project/data_target/targets{}.csv'.format(row_number)):
        with open(os.path.join('/mnt/home/grunew14/Documents/project/data_target', 'targets{}.csv'.format(row_number)), 'w') as target:
            t_rad = model.model.t_rad.value
            w = model.model.w
            # file should be in long format with columns t_rad# and w#
            # where # is the wavelength index
            writer = csv.writer(target)
            writer.writerow(['t_rad{}'.format(i) for i in range(len(t_rad))] +
                            ['w{}'.format(i) for i in range(len(w))])
            writer.writerow(t_rad.tolist() + w.tolist())


if __name__ == '__main__':
    index = int(sys.argv[1])
    main(index)


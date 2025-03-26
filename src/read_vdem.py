import numpy as np
import pandas as pd
from pathlib import Path

def read_vdem(shuffled=True, only_dm=True, standardize=True):
    """
    Read V-Dem dataset and return it as a DataFrame
    Args:
        shuffled: If true, the rows are permutated randomly (but consistenly each time)
        only_dm: If true, only keep dimensions that are used for diffusion map
        standardize: If true, standardize data, i.e. scale it to have zero mean and standard deviation one
            For right now, this is only possible if only_dm is True

    Returns:
        A pandas DataFrame containing the V-Dem dataset
    """
    projectroot = Path(__file__).parent.parent
    filename = projectroot / "data" / "data_vdem_dm.csv"
    vdem = pd.read_csv(filename)
    if only_dm:
        use_columns = [col for col in vdem.columns if col.startswith('v2') and col != 'v2x_polyarchy']
        vdem = vdem[use_columns]
        if standardize:
            vdem = (vdem - np.mean(vdem, axis=0)) / np.std(vdem, axis=0)
    if standardize and not only_dm:
        raise ValueError("can only standardize if only_dm is set")
    if shuffled:
        n = len(vdem)
        rng = np.random.default_rng(seed=1)
        indices=rng.permutation(n)
        vdem = vdem.iloc[indices]
    return vdem

import numpy as np
import pandas as pd
from pathlib import Path

def read_vdem(shuffled=True, only_vdem=True):
    """
    Read V-Dem dataset and return it as a DataFrame
    Args:
        shuffled: If true, the rows are permutated randomly (but consistenly each time)
        only_dm: If true, only include columns that are used for diffusion map
        only_vdem: If true, only include columns from the V-Dem dataset proper. Do not include, i.e. diffusion
        components or principal components
        standardize: If true, standardize data, i.e. scale it to have zero mean and standard deviation one
            For right now, this is only possible if only_dm is True

    Returns:
        A pandas DataFrame containing the V-Dem dataset
    """
    projectroot = Path(__file__).parent.parent
    filename = projectroot / "data" / "data_vdem_dm.csv"
    vdem = pd.read_csv(filename)
    if only_vdem:
        use_columns = (["country_name", "country_text_id", "country_id", "year", "COWcode"] +
                       [col for col in (vdem.columns) if (col.startswith('v2'))])
        vdem = vdem[use_columns]
    feature_columns = [col for col in vdem.columns if col.startswith('v2') and col != 'v2x_polyarchy']
    vdem.attrs.update({"feature_columns": feature_columns})
    if shuffled:
        n = len(vdem)
        rng = np.random.default_rng(seed=1)
        indices=rng.permutation(n)
        vdem = vdem.iloc[indices]
    return vdem

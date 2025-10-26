import numpy as np
import pandas as pd
from pathlib import Path

def read_vdem(shuffled=True):
    """
    Read V-Dem dataset and return it as a DataFrame
    Args:
        shuffled: If true, the rows are permutated randomly (but consistenly each time)
        components or principal components
            For right now, this is only possible if only_dm is True

    Returns:
        A pandas DataFrame containing the V-Dem dataset
    """
    projectroot = Path(__file__).parent.parent
    filename = projectroot / "data" / "vdem.csv"
    vdem = pd.read_csv(filename)
    feature_columns = [col for col in vdem.columns if col.startswith('v2') and col != 'v2x_polyarchy']
    vdem.attrs.update({"feature_columns": feature_columns})
    if shuffled:
        n = len(vdem)
        rng = np.random.default_rng(seed=1)
        indices=rng.permutation(n)
        vdem = vdem.iloc[indices]
    return vdem

import numpy as np
import pandas as pd
from pathlib import Path

def read_ising(filename, shuffled=False):
    """
    Read Ising configuration dataset and return it as a DataFrame
    """
    projectroot = Path(__file__).parent.parent
    filename = projectroot / "data" / "ising_configurations" / filename
    ising = pd.read_csv(filename)
    # Consider all columns with a numeric name feature columns, i.e. they contain configurations
    # Other columns may contain supplementary information
    feature_columns = []
    additional_columns = []
    for col in ising.columns:
        if col.isdigit():
            feature_columns.append(col)
        else:
            additional_columns.append(col)
    if shuffled:
        n = len(ising)
        rng = np.random.default_rng(seed=1)
        indices=rng.permutation(n)
        ising = ising.iloc[indices]
    return ising[feature_columns], ising[additional_columns]

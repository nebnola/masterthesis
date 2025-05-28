from pathlib import Path
from typing import Mapping, Iterable

import pandas as pd

from src.dr_decoders import DRDecoders


def compare_runs(filenames: Mapping | Iterable, directory="", label_name="method") -> pd.DataFrame:
    """
    Compare DRDecoders runs stored in different files and return combined data frame
    Args:
        filenames: Either a filename:label mapping or an iterable of filenames. In the second case, the filename
            itself is used as the label. The label is used in the label_name column of the returned data frame.
        directory: If given, is prepended to each filename
        label_name: The name of the column containing the labels. Default "method"

    Returns:
        A combined data frame of all DRDecoder runs
    """
    dfs = []
    try:
        # works if filenames is a dictionary
        items = filenames.items()
    except AttributeError:
        # fallback
        items = [(fn, fn) for fn in filenames]
    for fn, label in items:
        p = Path(directory) / Path(fn)
        decoders = DRDecoders.from_file(p).decoders
        decoders[label_name] = label
        dfs.append(decoders)
    return pd.concat(dfs)

VDEM_RUNS = ["vdem0", "vdem_large_eps", "vdem_nc3_opt", "vdem_pca0", "vdem_tsne"]
VDEM_DIR = "../data/models"
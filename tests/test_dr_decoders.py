import pandas as pd
from src.dr_decoders import DRDecoders
import pytest

data = pd.DataFrame({"name": ["Alpha", "Bravo", "Charlie", "Delta"],
                     "year": [1901, 1902, 1903, 1904],
                     "length": [1.7, 2.4, 0.3, 0.85],
                     "width": [0.22, 0.15, 0.19, 0.20],
                     "weight": [25.8, 27.1, 22.0, 12.7],
                     })
feature_columns = ["length", "width", "weight"]
data.attrs.update({"feature_columns": feature_columns})

def test_features():
    decoders = DRDecoders(data)
    assert decoders.data.equals(data)
    assert list(decoders.features.columns) == feature_columns
    assert decoders.features.shape == (4,3)
    assert decoders.features.equals(data[feature_columns])

def test_features_std():
    decoders = DRDecoders(data)
    assert list(decoders.features_std.columns) == feature_columns
    assert decoders.features_std.shape == (4,3)
    var_mean = decoders.features_std.mean().values
    var_std = decoders.features_std.std(ddof=0).values
    for mean, std in zip(var_mean, var_std):
        assert mean == pytest.approx(0)
        assert std == pytest.approx(1)

#def test_df():
    # TODO
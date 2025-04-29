import pickle
from typing import override, Self

import numpy as np
import pandas as pd

from src.diffusion_map import DiffusionMap
from src.neural_network import Trainer


class DRDecoders:
    """
    A class to contain decoders for a dimensionality reduction
    """

    def __init__(self, data: pd.DataFrame, encoding: pd.DataFrame=None, parameters=None, description=None) -> None:
        """

        Args:
            data: The original data
            encoding: The encoding of the data, which could be obtained e.g. by running a dimensionality reduction
            algorithm on the data
            parameters: The underlying parameters used, if any, to generate the data. Relevant mostly for synthetic data
            description: A description of the run. This is only stored and can be set freely
        """
        self.data = data
        # feature_columns is a list of all columns that are used for training - as opposed to columns which contain
        # e.g. labels or supplementary information
        try:
            self.feature_columns = data.attrs["feature_columns"]
        except KeyError:
            self.feature_columns = None
        self.encoding = encoding
        self.parameters = parameters
        self.decoders = pd.DataFrame()
        self.description = description

    def add_decoder(self, decoder: Trainer, **kwargs) -> None:
        """Add a decoder to the structure. Use arbitrary keyword arguments to label it with attributes
        Use the same structure (name and type) for the attributes for all decoders, however this is not enforced
        Do not use the following keys:
        decoder
        training_loss
        test_loss
        """
        # TODO: might use xarray instead
        self.decoders = self.decoders._append(kwargs | {'decoder': decoder}, ignore_index=True)

    def get_decoders(self, **kwargs) -> list[Trainer]:
        """
        Get all decoders matching the key value pairs passed in as keyword arguments.
        """
        decoder_filter = True
        for key, val in kwargs.items():
            decoder_filter = decoder_filter & (self.decoders[key] == val)
        return list(self.decoders.loc[decoder_filter, 'decoder'])

    def get_decoder(self, **kwargs) -> Trainer:
        """
        Get decoder matching the key value pairs passed in as keyword arguments.
        Intended for cases where there is only one decoder matching these parameters
        """
        return self.get_decoders(**kwargs)[0]

    def decode(self, **kwargs) -> pd.DataFrame:
        """Get reconstruction of original data using the decoder matching the keyword arguments"""
        # TODO write reconstruction for all decoders into one datastructure (xarray? DataFrame with MultiIndex?)
        decoder = self.get_decoder(**kwargs)
        test_data = decoder.test_dataloader.dataset.tensors[0]
        output = decoder.model(test_data)
        # get index. Assume test data is at the end of the data frame
        size = len(output)
        return pd.DataFrame(output.detach().numpy(), index=self.data.index[-size:], columns=list(self.feature_columns))

    def get_training_size(self, **kwargs) -> int:
        """Get training size that was used for training a decoder given by kwargs"""
        decoder = self.get_decoder(**kwargs)
        return len(decoder.train_dataloader.dataset)

    def test_decoders(self) -> None:
        """Test decoders and store the result in the decoders structure"""
        # TODO only test decoders which have not been tested?
        training_losses, test_losses = [], []
        for i in range(len(self.decoders)):
            run = self.decoders["decoder"][i]
            training_loss, test_loss = run.test()
            training_losses.append(training_loss)
            test_losses.append(test_loss)
        self.decoders["training_loss"] = training_losses
        self.decoders["test_loss"] = test_losses

    @property
    def features(self):
        """Get features of dataset. I.e. only the columns which are supposed to be used for training"""
        if self.feature_columns:
            return self.data[self.feature_columns]
        return self.data

    @property
    def features_std(self):
        """Get standardized features"""
        features = self.features
        return (features - np.mean(features, axis=0)) / np.std(features, axis=0, ddof=0)

    @property
    def df(self) -> pd.DataFrame:
        dfs_to_be_joined = []
        if self.parameters is not None:
            dfs_to_be_joined.append(self.parameters)
        if self.encoding is not None:
            dfs_to_be_joined.append(self.encoding)
        if len(dfs_to_be_joined) == 0:
            return self.data
        return self.data.join(dfs_to_be_joined)

    def copy(self, include_encoding=True) -> Self:
        """
        Copy data and (optionally) map to a new object. The decoders are not copied.
        Does not perform a deep copy. The data, encoding and parameters attributes are still shared!
        """

        cls = type(self)
        new = cls(data=self.data, parameters=self.parameters)
        if include_encoding:
            new.encoding = self.encoding
        return new

    def to_file(self, filename: str) -> None:
        """Save DRRun with all its data and the decoders"""
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def from_file(cls, filename: str) -> Self:
        with open(filename, 'rb') as file:
            return pickle.load(file)

    @classmethod
    def from_dummy_data(cls):
        # TODO
        pass


class DMDecoders(DRDecoders):
    @override
    def __init__(self, data: pd.DataFrame, dm: DiffusionMap = None, encoding: pd.DataFrame = None, parameters=None,
                 description=None):
        """

        Args:
            data: The original data
            dm: The DiffusionMap object applied to the data
            encoding: The encoding of the data, i.e. the DiffusionMap applied with a given t
            parameters: The underlying parameters used, if any, to generate the data. Relevant mostly for synthetic data
            description: A description of the run. This is only stored and can be set freely
        """
        super().__init__(data, encoding, parameters, description)
        self.dm = dm

    def calculate_dmap(self, t=None, standardize=True, *args, **kwargs):
        """Calculate DiffusionMap. If t is set, also set encoding"""
        if standardize:
            dm = DiffusionMap(np.array(self.features_std), *args, **kwargs)
        else:
            dm = DiffusionMap(np.array(self.features), *args, **kwargs)
        self.dm = dm
        if t is not None:
            self.set_dmap(t)

    def set_dmap(self, t):
        """Calculate and set the encoding attribute from an already calculated DiffusionMap"""
        dmap = self.dm.dmap(t)
        column_names = [f"dc{i}" for i in range(1, dmap.shape[1] + 1)]
        self.encoding = pd.DataFrame(dmap, index=self.data.index, columns=column_names)

        return self

    @override
    def copy(self, include_encoding=True):
        """
        Copy data and (optionally) map to a new object. The decoders are not copied.
        Does not perform a deep copy. The data, dm and dmap attributes are still shared!
        """
        new = super().copy(include_encoding=include_encoding)
        if include_encoding:
            new.dm = self.dm
        return new

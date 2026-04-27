import numbers
import pickle
from typing import override, Self, Iterable, Callable

import numpy as np
import pandas as pd
from torch import nn

from src.diffusion_map import DiffusionMap
from src.neural_network import TrainerABC, TrainParams, TrainerNew


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

    def add_decoder(self, decoder: TrainerABC | None, **kwargs) -> None:
        """Add a decoder to the structure. Use arbitrary keyword arguments to label it with attributes
        Use the same structure (name and type) for the attributes for all decoders, however this is not enforced
        Do not use the following keys:
        decoder
        training_loss
        test_loss
        """
        # TODO: might use xarray instead
        new_row = pd.DataFrame([kwargs | {'decoder': decoder}])
        self.decoders = pd.concat([self.decoders, new_row], ignore_index=True)

    def get_decoders(self, **kwargs) -> list[TrainerABC]:
        """
        Get all decoders matching the key value pairs passed in as keyword arguments.
        """
        decoder_filter = True
        for key, val in kwargs.items():
            decoder_filter = decoder_filter & (self.decoders[key] == val)
        return list(self.decoders.loc[decoder_filter, 'decoder'])

    def get_decoder(self, **kwargs) -> TrainerABC:
        """
        Get decoder matching the key value pairs passed in as keyword arguments.
        Intended for cases where there is only one decoder matching these parameters
        """
        return self.get_decoders(**kwargs)[0]

    def train_decoder(self, model: nn.Module, input_components: Iterable | None, train_params: TrainParams,
                      standardize=True, verbosity=0, attributes={}) -> TrainerABC:
        """
        Train decoder to reconstruct features from given components of the encoding and add it to decoders
        Args:
            model: the model to be trained
            input_components: The components of the encoding that are used as inputs. If None, use all components
            train_params: The training parameters to be used
            standardize: If true, use standardized features (mean 0, standard deviation 1) as labels
            verbosity:
            attributes: dictionary of key-value pairs associated with the decoder. Can be chosen freely apart from the
                restrictions in DRDecoders.add_decoder
        """
        if input_components is None:
            inputs = self.encoding
        else:
            input_components = list(input_components)
            # check if input components are indeces or column names
            if all(isinstance(c, numbers.Integral) for c in input_components):
                inputs = np.array(self.encoding)[:, input_components]
            else:
                inputs = np.array(self.encoding[input_components])
        if standardize:
            labels = self.features_std
        else:
            labels = self.features
        trainer = TrainerNew(model, train_params).train(inputs, labels, verbosity=verbosity)
        self.add_decoder(trainer, **attributes)
        return trainer

    def train_decoders_incremental(self, create_model: Callable[[int], nn.Module], train_params: TrainParams,
                                   components: Iterable, start=0,
                                   components_name = "components",
                                   n_components_name="ndims", standardize=True) -> None:
        """
        Train decoders to reconstruct features from different numbers of components of the encoding
        Args:
            create_model: Function to create the model. Receives the number of components
            train_params: training parameters to use for training
            components: The components to go through. Decoders are trained for [components[0]], then [components[0],
                components[1]], and so on
            start: How many components to start with. Default 0. If set to something larger than 0, training ist
                started with [components[0], ..., components[start-1]]
            components_name: The name for the column in the data frame that indicates the components used. Default
                "components"
            n_components_name: The name for the column in the data frame that indicates the number of components
                used. Default "ndims"
            standardize: Whether to use standardized features for training. Default True
        """
        components = list(components)
        for ndims in range(start, len(components) + 1):
            if ndims == 0:
                # Add MSE for zero dimension
                features = self.features_std if standardize else self.features
                var = np.mean(np.var(features.to_numpy(), axis=0))
                self.add_decoder(None, **{components_name: (), n_components_name: 0, 'test_loss': var,
                                        'training_loss': var} )
                continue
            use_components = tuple(components[:ndims])
            print(f"Training with components {use_components}")
            model = create_model(ndims)
            self.train_decoder(model, use_components, train_params, standardize, verbosity=0,
                               attributes = {
                                   components_name: use_components,
                                   n_components_name: ndims
                               })
        self.test_decoders()


    def decode(self, **kwargs) -> pd.DataFrame:
        """Get reconstruction of original data using the decoder matching the keyword arguments"""
        # TODO write reconstruction for all decoders into one datastructure (xarray? DataFrame with MultiIndex?)
        decoder = self.get_decoder(**kwargs)
        test_data = decoder.test_dataloader.dataset.tensors[0]
        output = decoder.model(test_data)
        # get index. Assume test data is at the end of the data frame
        size = len(output)
        columns = self.feature_columns or self.data.columns
        return pd.DataFrame(output.detach().numpy(), index=self.data.index[-size:], columns=list(columns))

    def get_training_size(self, **kwargs) -> int:
        """Get training size that was used for training a decoder given by kwargs"""
        decoder = self.get_decoder(**kwargs)
        return len(decoder.train_dataloader.dataset)

    def test_decoders(self) -> None:
        """Test decoders and store the result in the decoders structure"""
        # Makes sure indeces are unique so that the following code works
        self.decoders.reset_index(drop=True, inplace=True)
        # create loss columns if they don't exist yet
        if 'training_loss' not in self.decoders.columns:
            self.decoders['training_loss'] = np.nan
        if 'test_loss' not in self.decoders.columns:
            self.decoders['test_loss'] = np.nan
        for i in range(len(self.decoders)):
            row = self.decoders.loc[i]
            # check if loss is already written
            if (pd.isna(row['training_loss']) or
                    pd.isna(row['test_loss'])):
                trainer = row['decoder']
                if not pd.isna(trainer):
                    training_loss, test_loss = trainer.test()
                    self.decoders.loc[i, ['training_loss', 'test_loss']] = [training_loss, test_loss]

    def add_norm_loss(self, n_components_name="ndims") -> None:
        """Add normalized loss column to decoders"""
        ndims0 = self.decoders[self.decoders[n_components_name] == 0]
        if len(ndims0) != 1:
            raise ValueError("There needs to be exactly one row with 0 dimensions")
        var = ndims0.loc[0, "test_loss"]
        self.decoders["test_loss_norm"] = self.decoders["test_loss"] / var

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

import pickle

import numpy as np
import pandas as pd

from src.diffusion_map import DiffusionMap


class DMRun:
    def __init__(self, data, parameters=None, description=None):
        self.data = data
        self.parameters = parameters
        self.dm = None
        self.dmap = None
        self.decoders = pd.DataFrame()
        self.description = description

    def calculate_dmap(self, t=None, *args, **kwargs):
        """Calculate DiffusionMap. If t is set, also set dmap"""
        # TODO: introduce option to only use some columns
        dm = DiffusionMap(np.array(self.data), *args, **kwargs)
        self.dm = dm
        if t is not None:
            self.set_dmap(t)

    def set_dmap(self, t):
        """Calculate and set the dmap attribute from an already calculated DiffusionMap"""
        dmap = self.dm.dmap(t)
        column_names = [f"dc{i}" for i in range(1, dmap.shape[1] + 1)]
        self.dmap = pd.DataFrame(dmap, columns=column_names)

        return self

    def add_decoder(self, decoder, **kwargs):
        """Add a decoder to the structure. Use arbitrary keyword arguments to label it with attributes
        Use the same structure (name and type) for the attributes for all decoders, however this is not enforced
        Do not use the following keys:
        decoder
        training_loss
        test_loss
        """
        # TODO: might use xarray instead
        self.decoders = self.decoders._append(kwargs | {'decoder': decoder}, ignore_index=True)

    def get_decoder(self, **kwargs):
        """Get decoder matching the key value pairs passed in as keyword arguments.
        Intended for cases where there is only one decoder matching these parameters
        """
        decoder_filter = True
        for key, val in kwargs.items():
            decoder_filter = decoder_filter & (self.decoders[key] == val)
        return self.decoders.loc[decoder_filter, 'decoder'].item()

    def decode(self, **kwargs):
        """Get reconstruction of original data using the decoder matching the keyword arguments"""
        # TODO write reconstruction for all decoders into one datastructure (xarray? DataFrame with MultiIndex?)
        decoder = self.get_decoder(**kwargs)
        # assume the training
        test_data = decoder.test_dataloader.dataset.tensors[0]
        output = decoder.model(test_data)
        return pd.DataFrame(output.detach().numpy(), columns=list(self.data.columns))

    def get_training_size(self, **kwargs):
        """Get training size that was used for training a decoder given by kwargs"""
        decoder = self.get_decoder(**kwargs)
        return len(decoder.train_dataloader.dataset)

    def test_decoders(self):
        """Test decoders and store the result in the decoders structure"""
        training_losses, test_losses = [], []
        for i in range(len(self.decoders)):
            run = self.decoders["decoder"][i]
            training_loss, test_loss = run.test()
            training_losses.append(training_loss)
            test_losses.append(test_loss)
        self.decoders["training_loss"] = training_losses
        self.decoders["test_loss"] = test_losses

    @property
    def df(self):
        dfs_to_be_joined=[]
        if self.parameters is not None:
            dfs_to_be_joined.append(self.parameters)
        if self.dmap is not None:
            dfs_to_be_joined.append(self.dmap)
        if len(dfs_to_be_joined) == 0:
            return self.data
        return self.data.join(dfs_to_be_joined)

    def copy(self, include_dmap = True):
        """
        Copy data and (optionally) map to a new object. The decoders are not copied.
        Does not perform a deep copy. The data, dm and dmap attributes are still shared!
        """

        new = DMRun(data=self.data, parameters=self.parameters)
        if include_dmap:
            new.dm = self.dm
            new.dmap = self.dmap
        return new

    def to_file(self, filename):
        """Save DMRun with all its data and the decoders"""
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def from_file(cls, filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)


    @classmethod
    def from_dummy_data(cls):
        #TODO
        pass

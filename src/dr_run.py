import pickle

import pandas as pd


class DRRun:
    """
    A class to contain decoders for a dimensionality reduction
    TODO: integrate with DMRun. DMRun should probably inherit from DRRun
    """

    def __init__(self, data: pd.DataFrame, encoding=None, parameters=None, description=None):
        """

        Args:
            data: The original data
            encoding: The encoding of the data, which could be obtained e.g. by running a dimensionality reduction
            algorithm on the data
            parameters: The underlying parameters used, if any, to generate the data. Relevant mostly for synthetic data
            description: A description of the run. This is only stored and can be set freely
        """
        self.data = data
        self.encoding = encoding
        self.parameters = parameters
        self.decoders = pd.DataFrame()
        self.description = description

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
        dfs_to_be_joined = []
        if self.parameters is not None:
            dfs_to_be_joined.append(self.parameters)
        if self.encoding is not None:
            dfs_to_be_joined.append(self.encoding)
        if len(dfs_to_be_joined) == 0:
            return self.data
        return self.data.join(dfs_to_be_joined)

    def copy(self, include_encoding=True):
        """
        Copy data and (optionally) map to a new object. The decoders are not copied.
        Does not perform a deep copy. The data, encoding and parameters attributes are still shared!
        """

        cls = type(self)
        new = cls(data=self.data, parameters=self.parameters)
        if include_encoding:
            new.encoding = self.encoding
        return new

    def to_file(self, filename):
        """Save DRRun with all its data and the decoders"""
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def from_file(cls, filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)

    @classmethod
    def from_dummy_data(cls):
        # TODO
        pass

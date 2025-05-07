from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Self, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class FeedForward(nn.Module):
    """
    Feed-forward network
    Used e.g. for "decoding" output from dimensionality reduction algorithm back to original data
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_layers=[50, 50], activation_function=nn.ReLU):
        super().__init__()
        stack = [
            nn.Linear(input_dim, hidden_layers[0]),
            activation_function(),
        ]
        for i in range(len(hidden_layers) - 1):
            stack += [
                nn.Linear(hidden_layers[i], hidden_layers[i + 1]),
                activation_function(),
            ]
        stack += [
            nn.Linear(hidden_layers[-1], output_dim),
        ]
        self.stack = nn.Sequential(*stack)

    @classmethod
    def uniform(cls, input_dim: int, output_dim: int, hidden_layers: int = 2, hidden_layer_size: int = 50,
                activation_function=nn.ReLU):
        hidden_layers = [hidden_layer_size] * hidden_layers
        return cls(input_dim, output_dim, hidden_layers, activation_function)

    def forward(self, x):
        return self.stack(x)


class TrainerABC(ABC):
    """Abstract class subclassed by Trainer and TrainerNew during the migration"""
    pass

    @abstractmethod
    def test(self) -> Tuple[float, float]:
        pass


class Trainer(TrainerABC):
    """
    Contains the necessary logic for training the model
    TODO: make it possible to continue training
    """
    device = "cpu"

    def __init__(
            self,
            model: nn.Module,
            inputs,
            labels,
            training_size: int,
            loss_fn,
            epochs: int,
            weight_decay=0,
            batch_size: int = 32,
            lr: float = None,
            scheduler: type = None,
            scheduler_kwargs: dict = None,
            early_stopping_patience: int | None = None,
    ):
        """

        Args:
            model:
            inputs:
            labels:
            training_size:
            loss_fn:
            epochs:
            weight_decay:
            batch_size:
            lr:
            scheduler:
            scheduler_kwargs:
            early_stopping_patience: Used for early stopping when using ReduceLROnPlateau.
                Early stopping is activated when early_stopping_patience is set to an integer.
                After ReduceLROnPlateau reaches its last learning rate, training is continued for early_stopping_patience
                epochs, then stopped
        """
        model = model.to(self.device)
        self.model = model
        inputs = np.array(inputs)
        labels = np.array(labels)
        train_dataset = TensorDataset(torch.tensor(inputs[:training_size, ...]),
                                      torch.tensor(labels[:training_size, ...]))
        test_dataset = TensorDataset(torch.tensor(inputs[training_size:, ...]),
                                     torch.tensor(labels[training_size:, ...]))
        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        if len(test_dataset) <= 5000:
            test_batch_size = 5000
        else:
            # divide test into equal sized batches of at most 5000
            n_batches = np.ceil(len(test_dataset)/5000)
            test_batch_size = int(np.ceil(len(test_dataset)/n_batches))
        self.test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size)
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.lr = lr
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        # if no scheduler is set, use exponential scheduler with gamma=1 (amounts to a constant learning rate)
        if scheduler is None:
            scheduler = torch.optim.lr_scheduler.ExponentialLR
            scheduler_kwargs = dict(gamma=1)
        if scheduler_kwargs is None:
            scheduler_kwargs = dict()
        self.scheduler = scheduler(self.optimizer, **scheduler_kwargs)
        if early_stopping_patience is not None and scheduler != torch.optim.lr_scheduler.ReduceLROnPlateau:
            raise Warning("early_stopping has no effect when not using ReduceLROnPlateau as learning rate scheduler")
        self.early_stopping_patience = early_stopping_patience

    def epoch(self, verbosity=3):
        """Train the model for one epoch"""
        size = len(self.train_dataloader.dataset)
        self.model.train()
        for batch, (X, y) in enumerate(self.train_dataloader):
            X, y = X.to(self.device), y.to(self.device)

            # Compute prediction error
            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            if verbosity >= 3:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]\r", end="")
        if verbosity >= 3:
            print()

    def train(self, verbosity=3, history=False):
        """Train the model for the given number of epochs

        :param verbosity: verbosity level. Level 0: do not print anything. Level 1: print epoch. Level 2: Test and print accuracy every epoch. Level 3: Print Updated Loss during each epoch
        :param history: If True keep a history of the training loss, test loss, learning rate as a function of the number of epochs
        """
        if history:
            # TODO: use data frame as training record
            training_record = np.empty((self.epochs, 4))
            training_record[:, 0] = range(1, self.epochs + 1)  # first column is just the number of epochs
        early_stopping_count = 0
        for t in range(self.epochs):
            if verbosity >= 1:
                print(f"Epoch {t + 1}")
            self.epoch(verbosity=verbosity)
            training_loss, test_loss = self.test()
            if history:
                training_record[t, 1] = training_loss
                training_record[t, 2] = test_loss
                training_record[t, 3] = self.scheduler.get_last_lr()[0]
            if verbosity >= 2:
                print(f"Training Loss: {training_loss:>0.5f}, Test Loss: {test_loss:>0.5f}")
                print(f"Learning Rate: {self.scheduler.get_last_lr()}")

            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(training_loss)
                if self.early_stopping_patience is not None:
                    if self.scheduler.get_last_lr()[0] <= self.scheduler.default_min_lr:
                        early_stopping_count += 1
                    if early_stopping_count > self.early_stopping_patience:
                        training_record = training_record[:(t+1),:]
                        break
            else:
                self.scheduler.step()
        if history:
            self.training_record = training_record

    def test(self):
        self.model.eval()
        training_loss, test_loss = 0, 0
        with torch.no_grad():
            for X, y in self.train_dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                training_loss += self.loss_fn(pred, y).item()
            for X, y in self.test_dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()
        return training_loss / len(self.train_dataloader), test_loss / len(self.test_dataloader)


@dataclass
class TrainParams:
    """
    Data class for containing training parameters. Does not include model parameters such as the network layers
    """
    training_size: int
    loss_fn: type
    epochs: int
    batch_size: int
    weight_decay: float
    lr: float
    scheduler: type
    scheduler_kwargs: dict

    early_stopping_patience: int | None
    """Used for early stopping when using ReduceLROnPlateau.
    Early stopping is activated when early_stopping_patience is set to an integer.
    After ReduceLROnPlateau reaches its last learning rate, training is continued for early_stopping_patience
    epochs, then stopped"""

class TrainerNew(TrainerABC):

    model: nn.Module
    training_record: pd.DataFrame
    train_dataloader: DataLoader
    test_dataloader: DataLoader

    device = "cpu"

    def __init__(
            self,
            model: nn.Module,
            train_params: TrainParams
    ):
        model = model.to(self.device)
        self.model = model
        self.train_params = train_params

        self.loss_fn = train_params.loss_fn()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=train_params.lr,
                                          weight_decay=train_params.weight_decay)
        # if no scheduler is set, use exponential scheduler with gamma=1 (amounts to a constant learning rate)
        scheduler = train_params.scheduler
        scheduler_kwargs = train_params.scheduler_kwargs
        if train_params.scheduler is None:
            scheduler = torch.optim.lr_scheduler.ExponentialLR
            scheduler_kwargs = dict(gamma=1)
        if scheduler_kwargs is None:
            scheduler_kwargs = dict()
        self.scheduler = scheduler(self.optimizer, **scheduler_kwargs)
        if train_params.early_stopping_patience is not None and scheduler != torch.optim.lr_scheduler.ReduceLROnPlateau:
            raise Warning("early_stopping has no effect when not using ReduceLROnPlateau as learning rate scheduler")


    def epoch(self, verbosity=3):
        """Train the model for one epoch"""
        size = len(self.train_dataloader.dataset)
        self.model.train()
        for batch, (X, y) in enumerate(self.train_dataloader):
            X, y = X.to(self.device), y.to(self.device)

            # Compute prediction error
            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            if verbosity >= 3:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]\r", end="")
        if verbosity >= 3:
            print()

    def train(self, inputs, labels, verbosity=0) -> Self:
        """Train the model for the number of epochs given in train_params

        Args:
            inputs: The inputs to the model
            labels: The labels, or targets that the model is trained to match
            verbosity: verbosity level. Level 0: do not print anything. Level 1: print epoch.
                Level 2: Test and print accuracy every epoch. Level 3: Print Updated Loss during each epoch
        """
        inputs = np.array(inputs)
        labels = np.array(labels)
        training_size = self.train_params.training_size
        train_dataset = TensorDataset(torch.tensor(inputs[:training_size, ...], dtype=torch.float64),
                                      torch.tensor(labels[:training_size, ...], dtype=torch.float64))
        test_dataset = TensorDataset(torch.tensor(inputs[training_size:, ...], dtype=torch.float64),
                                     torch.tensor(labels[training_size:, ...], dtype=torch.float64))
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.train_params.batch_size, shuffle=True)
        if len(test_dataset) <= 5000:
            test_batch_size = 5000
        else:
            # divide test into equal sized batches of at most 5000
            n_batches = np.ceil(len(test_dataset)/5000)
            test_batch_size = int(np.ceil(len(test_dataset)/n_batches))
        self.test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size)

        self.training_record = pd.DataFrame()
        self.continue_training(self.train_params.epochs, verbosity)
        return self

    def continue_training(self, epochs, verbosity=0) -> Self:
        training_losses = []
        test_losses = []
        learning_rates = []

        early_stopping_count = 0
        for t in range(epochs):
            if verbosity >= 1:
                print(f"Epoch {t + 1}")
            self.epoch(verbosity=verbosity)
            training_loss, test_loss = self.test()
            training_losses.append(training_loss)
            test_losses.append(test_loss)
            learning_rates.append(self.scheduler.get_last_lr()[0])
            if verbosity >= 2:
                print(f"Training Loss: {training_loss:>0.5f}, Test Loss: {test_loss:>0.5f}")
                print(f"Learning Rate: {self.scheduler.get_last_lr()}")

            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(training_loss)
                if self.train_params.early_stopping_patience is not None:
                    if self.scheduler.get_last_lr()[0] <= self.scheduler.default_min_lr:
                        early_stopping_count += 1
                    if early_stopping_count > self.train_params.early_stopping_patience:
                        break
            else:
                self.scheduler.step()
        self.training_record = pd.concat([self.training_record, pd.DataFrame(dict(
            training_loss = training_losses,
            test_loss = test_losses,
            learning_rate = learning_rates))], ignore_index=True)
        self.training_record["epoch"] = self.training_record.index + 1
        return self


    def test(self):
        """Test the trained model

        Returns:
            training_loss, test_loss
        """
        self.model.eval()
        training_loss, test_loss = 0, 0
        with torch.no_grad():
            for X, y in self.train_dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                training_loss += self.loss_fn(pred, y).item()
            for X, y in self.test_dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()
        # TODO account for varying batch sizes
        return training_loss / len(self.train_dataloader), test_loss / len(self.test_dataloader)

    def plot_training(self):
        """Plot training progress (loss by epoch) and learning rate"""
        loss_record = self.training_record
        fig, ax = plt.subplots()

        ax.plot(loss_record["epoch"], loss_record["training_loss"], label='training loss')
        ax.plot(loss_record["epoch"], loss_record["test_loss"], label='test loss')
        ax.set_xlabel("epochs")
        ax.set_ylabel("loss")

        axlr = ax.twinx()
        axlr.plot(loss_record["epoch"], loss_record["learning_rate"], label='learning rate')
        axlr.set_ylabel("learning rate")
        axlr.semilogy()

        fig.legend()

        print(f"final test loss: {loss_record.iloc[-1]["test_loss"]}")
        return fig, ax
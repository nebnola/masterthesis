import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class DecodeDMModel(nn.Module):
    """
    Model for "decoding" output from diffusion map back to original data
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


class DecodeDM:
    """
    Contains the necessary logic for training the model
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
            batch_size: int,
            lr: float = None,
            scheduler: type = None,
            scheduler_kwargs: dict = None,
            weight_decay=0,
    ):
        model = model.to(self.device)
        self.model = model
        inputs = np.array(inputs)
        labels = np.array(labels)
        train_dataset = TensorDataset(torch.tensor(inputs[:training_size, ...]),
                                      torch.tensor(labels[:training_size, ...]))
        test_dataset = TensorDataset(torch.tensor(inputs[training_size:, ...]),
                                     torch.tensor(labels[training_size:, ...]))
        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        self.test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
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
                # TODO: we don't really need a test data loader, do it all at once
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()
        return training_loss / len(self.train_dataloader), test_loss / len(self.test_dataloader)

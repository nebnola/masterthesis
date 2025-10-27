This package contains code for my thesis "Quantifying the Accuracy of Diffusion Map for Dimensionality Reduction Using
Neural Networks" In particular, the neural reconstruction error (NRE) is implemented.

## Project structure

`src`: Core of the functionality, including implementations of the diffusion map and the NRE, alongside some utility functions  
`data`: Contains all the datasets used in the thesis  
`fput` and `isingmodel`: Code to simulate the FPUT system and Ising model and generate the respective datasets

## Project Setup

```
pip install -r requirements.txt
```

In the root directory:
```
pip install -e .
```
to make the `src` package available everywhere

## Basic usage

```python
from src.dr_decoders import DRDecoders
from src.neural_network import FeedForward, TrainParams

decoders = DRDecoders(data_original, data_reduced)
def create_model(input_dimensions: int):
    # You can return any nn.Module of suitable input and output dimensions here
    # p is the dimensionality of the original data space
    return FeedForward(input_dimensions, p, hidden_layers=[50,50])
train_params = TrainParams(
    ...
)
decoders.train_decoders_incremental(create_model, train_params, components=range(10))
```

The NREs are then found in the `decoders.decoders` DataFrame in the column `"test_loss"`
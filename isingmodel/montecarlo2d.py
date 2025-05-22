import numba
import random
import numpy as np

"""
This implementation is adapted from https://github.com/ising-model/ising-model-python, where it is released under the
MIT License

MIT License

Copyright (c) 2023 JinwooLim

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
"""

@numba.jit(nopython=True)
def _metropolis(spin, L, beta):
    for _ in range(L ** 2):
        # randomly sample a spin
        x, y = random.randint(0, L - 1), random.randint(0, L - 1)
        s = spin[x, y]

        # Sum of the spins of nearest neighbors
        xpp = (x + 1) if (x + 1) < L else 0
        ypp = (y + 1) if (y + 1) < L else 0
        xnn = (x - 1) if (x - 1) >= 0 else (L - 1)
        ynn = (y - 1) if (y - 1) >= 0 else (L - 1)
        R = spin[xpp, y] + spin[x, ypp] + spin[xnn, y] + spin[x, ynn]

        # Check Metropolis-Hastings algorithm for more details
        dH = 2 * s * R  # Change of the Hamiltionian after flippling the selected spin
        if dH < 0:      # Probability of the flipped state is higher -> flip the spin
            s = -s
        elif np.random.rand() < np.exp(-beta * dH): # Flip randomly according to the temperature
            s = -s
        spin[x, y] = s


class MonteCarlo2D:
    def __init__(self, size: int, eqstep: int, n_samples: int, sample_step: int):
        """

        Args:
            size:
            eqstep: The number of equalization steps to make at the beginning.
            n_samples: The number of samples to take
            sample_step: The number of MC steps to take between samples.
        """
        self.L = size
        self.eqstep = eqstep
        self.n_samples = n_samples
        self.sample_step = sample_step
        self.area = self.L ** 2

    def _init_spin(self):
        return 2 * np.random.randint(2, size=(self.L, self.L)) - 1
    
    # Calculate energy using neighbors
    def _calc_energy(self, spin):
        R = np.roll(spin, 1, axis=0) + np.roll(spin, -1, axis=0) + np.roll(spin, 1, axis=1) + np.roll(spin, -1, axis=1)
        return np.sum(-R * spin) / (4 * self.area)
    
    def _calc_magnetization(self, spin):
        return np.sum(spin) / self.area
    
    def simulate(self, beta):
        # This will store the configurations
        configurations = np.empty((self.n_samples, self.L, self.L))
        # Initialize the lattice randomly
        spin = self._init_spin()
        # Equilibration steps
        for _ in range(self.eqstep):
            _metropolis(spin, self.L, beta)
        # Monte Carlo steps
        for i in range(self.n_samples):
            for _ in range(self.sample_step):
                _metropolis(spin, self.L, beta)
            configurations[i] = spin

        return configurations

    def simulate_independent(self, beta):
        """Simulate runs which are all initialized independently"""
        configurations = np.empty((self.n_samples, self.L, self.L))
        for i in range(self.n_samples):
            spin = self._init_spin()
            for _ in range(self.eqstep):
                _metropolis(spin, self.L, beta)

            configurations[i] = spin

        return configurations

    def simulate_independent_betas(self, betas):
        """Simulate runs with different temperatures which are all initialized independently
        This ignores the n_samples and sample_step attributes
        """
        configurations = np.empty((len(betas), self.L, self.L))
        for i, beta in enumerate(betas):
            spin = self._init_spin()
            for _ in range(self.eqstep):
                _metropolis(spin, self.L, beta)

            configurations[i] = spin

        return configurations


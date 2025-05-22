import numpy as np
import pandas as pd

from montecarlo2d import MonteCarlo2D


def export_configurations(conf, fn):
    df = pd.DataFrame(conf.reshape(conf.shape[0], -1))
    df.to_csv(fn, index=False)

if __name__ == "__main__":
    # T = 2.0
    conf = MonteCarlo2D(30, eqstep=1000, n_samples=15000, sample_step=1).simulate_independent(1/2.0)
    export_configurations(conf, "../data/ising_configurations/ising_2_0.csv")
    # T = 3.0
    conf = MonteCarlo2D(30, eqstep=500, n_samples=12000, sample_step=1).simulate_independent(1/3.0)
    export_configurations(conf, "../data/ising_configurations/ising_3_0.csv")
    # Critical point
    conf = MonteCarlo2D(30, eqstep=1000, n_samples=12000, sample_step=1).simulate_independent(1/2.27)
    export_configurations(conf, "../data/ising_configurations/ising_t_c.csv")
    # Mixing different temperatures in dataset
    Ts = np.linspace(1.5, 3, 12000)
    conf = MonteCarlo2D(30, eqstep=5000, n_samples=12000, sample_step=1).simulate_independent_betas(1/Ts)
    df = pd.DataFrame(conf.reshape(conf.shape[0], -1))
    df["T"] = Ts
    df.to_csv("../data/ising_configurations/ising_mixed.csv", index=False)
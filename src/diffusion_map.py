import numpy as np
import copy

from scipy import sparse
from scipy import linalg
from sklearn.metrics.pairwise import pairwise_distances


class DiffusionMap:
    def __init__(self, data, n_eigenv: int, epsilon=None, nnn=None, alpha_normalization=0, zero_threshold=1e-20,
                 eigen_solver="sparse"):
        """
        Instantiate Diffusion Map from data with a Gaussian kernel.
        Call DiffusionMap.dmap() to actually calculate diffusian map for a given t

        :param data: Input data. Has to be of shape (N_observations, N_dimensions)
        :param n_eigenv: Number of eigenvectors to use. Maximum of N_observations-1, since the first eigenvector is not counted.
        :param epsilon: Width of the Gaussian kernel.
        :param nnn: Only keep n nearest neighbors. All other entries of the adjacency matrix are set to zero.
        If not set, use all connections.
        :param alpha_normalization: value of alpha when setting L=D^(-alpha)WD^(-alpha), as explained in Coifman, Lafon (2006)
            Common choices are:
            alpha=0: No normalization (default)
            alpha=1/2: Random walk in a potential U, where U is such that the density of points on the manifold is given by exp(-U)
            alpha=1: Unbiased random walk on the data manifold. This gets rid of the influence of the density of data points.
        :param zero_threshold: Set all values in Laplacian below the zero threshold to zero. This drastically improves performance while
            not affecting the accuracy if zero_threshold is small enough
            It is normalized to the maximal value of the Laplacian
            default: 1e-20
            Has no effect if set to zero
        :eigen_solver: which solver to use for the eigenvalue problem.
            "sparse": use the solver from scipy.sparse. Can be used to only compute the specified number of leading
            eigenvectors. (default)
            "linalg": use the solver from scipy.linalg. Computes all eigenvalues. Usually slower
        """
        self.data = data
        self.epsilon = epsilon
        self.nnn = nnn
        self.alpha_normalization = alpha_normalization

        if eigen_solver == "sparse" and n_eigenv >= data.shape[0] - 1:
            raise ValueError("Cannot use sparse eigenvector calculation to get all eigenvectors. Use 'linalg' instead")

        distance = pairwise_distances(data, metric="euclidean")

        # kernel matrix
        W = np.exp(-distance ** 2 / epsilon)
        if nnn is not None:
            W = self.n_nearest_neighbors(W, nnn)

        if alpha_normalization == 0:
            L = W
        else:
            d_malpha = np.sum(W, axis=1) ** (-alpha_normalization)  # normalization factors
            L = d_malpha[:, np.newaxis] * W * d_malpha  # equivalent to D^(-alpha) W D^(-alpha)

        if zero_threshold != 0:
            threshold = zero_threshold * np.max(L)
            L[L < threshold] = 0

        # TODO: Factor out calculation of L in case you need it
        trace = np.sum(L)  # trace of L for normalization purposes

        if n_eigenv == 0:
            return

        d = np.sum(L, axis=1)  # normalization factors
        d_12 = 1 / np.sqrt(d)
        # M_s, the symmetrized version of transition matrix M:
        Ms = d_12[:, np.newaxis] * L * d_12  # equivalent to D^(-1/2) L D^(-1/2) since D is diagonal
        if eigen_solver == "sparse":
            Ms = sparse.csr_matrix(Ms)
            eigenvals, eigenvecs = sparse.linalg.eigs(Ms, k=n_eigenv + 1, which='LR')
        elif eigen_solver == "linalg":
            eigenvals, eigenvecs = linalg.eig(Ms, left=False, right=True)
        else:
            raise ValueError("eigen_solver must be one of ('sparse', 'linalg')")
        idx = np.argsort(eigenvals)[::-1]
        eigenvals = np.real(eigenvals[idx])
        eigenvecs = np.real(eigenvecs[:, idx])
        eigenvals = eigenvals[:n_eigenv + 1]
        eigenvecs = eigenvecs[:, :n_eigenv + 1]  # only keep appropriate amount of eigenvectors
        eigenvecs /= np.linalg.norm(eigenvecs, axis=0)  # normalize eigenvectors

        phis = 1 / d_12[:, np.newaxis] / np.sqrt(trace) * eigenvecs
        psis = d_12[:, np.newaxis] * np.sqrt(trace) * eigenvecs

        if phis[0, 0] < 0:
            phis[:, 0] *= (-1)

        self.eigenvals = eigenvals
        self.eigenvecs = eigenvecs
        self.phis = phis
        self.psis = psis

    def dmap(self, t, n_eigenv: int = None):
        """
        Calculate diffusian map for given number of time steps
        :param t: Number of time steps
        :param n_eigenv: Number of eigenvalues to keep. Must be less than n_eigenv used in the init function
        """
        dmap = self.eigenvals ** t * self.psis
        if n_eigenv is None:
            n_eigenv = len(self.eigenvals) - 1
        if n_eigenv + 1 > len(self.eigenvals):
            raise ValueError(f"Cannot use {n_eigenv} eigenvalues, at most {len(self.eigenvals) - 1} available")
        return dmap[:, 1:n_eigenv + 1]

    @classmethod
    def n_nearest_neighbors(cls, L: np.ndarray, nnn: int, safe=True):
        """
        Take only n nearest neighbors of adjacency matrix, set all other entries to 0
        If distances/weights are equal, behaviour is undefined
        :param L: The kernel matrix, shape (n_observations, n_observations). Assumption is that the entries are a monotonically decreasing function of the distance
        :param nnn: The number of nearest neighbors to keep. Includes the point itself
        :param safe: If set to True (default), make a copy of the input array. If set to False, the input array is changed in-place and cannot be re-used
        """
        if safe:
            L = copy.deepcopy(L)
        idx = L.shape[0] - nnn  # index that separates the n largest values from the rest
        thresholds = np.partition(L, idx, axis=1)[:, idx]  # list of thresholds. All smaller values are set to zero
        L[L < thresholds[:, np.newaxis]] = 0
        return np.maximum(L, L.T)

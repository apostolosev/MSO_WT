import math
import itertools
import numpy as np

from pywt import WaveletPacket
from scipy import signal, linalg, fftpack

"""
--------------------------------------------
Cost function to perform best basis selection
---------------------------------------------
"""


def entropy(x):
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    x = x[x != 0]

    return -np.sum(np.abs(x) ** 2 * np.log2(np.abs(x) ** 2))


"""
--------------------------------------
Kernel function to measure similarity
--------------------------------------
"""


def hist(x, y):
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Signal dimension must be 1")
    elif len(x) != len(y):
        raise ValueError("Signals must be of equal size")

    return np.sum(np.min(np.array([x, y]), axis=0))


"""
-----------------------------------
Best Basis Selection implementation
-----------------------------------
"""


class Node:
    def __init__(self, data=None, name='', left=None, right=None, depth=0, cost=0):
        self.data = np.copy(data)
        self.name = name
        self.left = left
        self.right = right
        self.depth = depth
        self.cost = cost

    # Method to add a right child
    def add_right(self, data):
        del self.right
        self.right = Node(data=data, name=self.name + 'd', depth=self.depth + 1)

    # Method to add a left child
    def add_left(self, data):
        del self.left
        self.left = Node(data=data, name=self.name + 'a', depth=self.depth + 1)

    # Method to return the right child
    def get_right(self):
        return self.right

    # Method to return the left child
    def get_left(self):
        return self.left

    # Method to return both children of the node
    def get_children(self):
        children = [self.left, self.right]
        children = [node for node in children if node is not None]
        return children

    # Method to delete the right child
    def delete_right(self):
        self.right = None

    # Method to delete the left child
    def delete_left(self):
        self.left = None

    # Method to delete all the children of the tree
    def delete(self):
        if self.left is not None:
            self.left.delete()
            self.delete_left()
        if self.right is not None:
            self.right.delete()
            self.delete_right()

    # Method to print node information
    def info(self):
        print("Node name: {}".format(self.name))
        print("Depth: {}\n".format(self.depth))

    # Method to check if the node has children
    def has_children(self):
        return len(self.get_children()) > 0

    # Method to perform best basis search
    def prune(self, cost_func):

        if not self.has_children():
            self.cost = cost_func(self.data)
            return self.cost
        else:
            C_right = self.right.prune(cost_func)
            C_left = self.left.prune(cost_func)
            C_self = cost_func(self.data)

            if C_self < C_right + C_left:
                self.cost = C_self
                self.delete_left()
                self.delete_right()
            else:
                self.cost = C_right + C_left

            return self.cost

    # Method to construct a wavelet packet tree
    def construct(self, wp):

        # Delete the previous tree structure
        self.name = wp.path
        self.data = np.copy(wp.data)
        # self.cost = func(wp.data)

        if wp.level != wp.maxlevel:
            self.left = Node(depth=self.depth + 1)
            self.right = Node(depth=self.depth + 1)

            self.left.construct(wp['a'])
            self.right.construct(wp['d'])

    def get_leafs(self):
        # Method to reconstruct a wavelet packet representation
        if self.has_children() is False:
            return [self.name]
        else:
            return list(itertools.chain(self.left.get_leafs(), self.right.get_leafs()[::-1]))


"""
------------------------------
Bellman's k-segments algorithm
------------------------------
"""


# Function to precompute the distance matrix and the means
def prepare_ksegments(series, weights):
    # Sequence length
    N = series.shape[1]
    # Sequence dimension
    d = series.shape[0]

    # Initialize the helper matrices
    W = np.diag(weights)
    Q = np.diag(weights * np.linalg.norm(series, ord=2, axis=0) ** 2)
    S = np.zeros((N, N, d))

    # Initialize the means
    means = np.zeros_like(S)
    for i in range(N):
        S[i, i, ...] = weights[i] * series[..., i]
        means[i, i, ...] = series[..., i]

    # Initialize the subsequence distances
    dists = np.zeros((N, N))

    for i in range(N):
        for j in range(N - i):
            r = i + j

            # Helper matrices
            W[j, r] = W[j, r - 1] + W[r, r]
            Q[j, r] = Q[j, r - 1] + Q[r, r]
            S[j, r, ...] = S[j, r - 1, ...] + S[r, r, ...]

            # Means and distances
            means[j, r, ...] = S[j, r, ...] / W[j, r]
            dists[j, r] = Q[j, r] - np.linalg.norm(S[j, r, ...], ord=2) ** 2 / W[j, r]

    return dists, means


# Function to perform k-segmentation
def ksegments(series, weights, k):
    # Sequence length
    N = series.shape[1]
    # Sequence dimension
    d = series.shape[0]

    # Precompute distances and means
    dists, means = prepare_ksegments(series, weights)

    k_seg_dist = np.zeros((k, N + 1))
    k_seg_path = np.zeros((k, N))
    k_seg_dist[0, 1:] = dists[0, :]

    k_seg_path[0, :] = 0
    for i in range(k):
        k_seg_path[i, :] = i

    for i in range(1, k):
        for j in range(i, N):
            choices = k_seg_dist[i - 1, :j] + dists[:j, j]
            best_index = np.argmin(choices)
            best_val = np.min(choices)

            k_seg_path[i, j] = best_index
            k_seg_dist[i, j + 1] = best_val

    labels = np.zeros(N, dtype=int)
    rhs = len(labels) - 1
    count = 0

    for i in reversed(range(k)):
        lhs = k_seg_path[i, rhs]
        labels[int(lhs):rhs] = count
        rhs = int(lhs)
        count += 1

    return labels


"""
-----------------------------------------------------------------
Class to implement the Maximal Spectral Overlap Wavelet Transform
-----------------------------------------------------------------
"""


class MSOWT:
    def __init__(self, nIMFs=3):
        self.nIMFs = nIMFs

    # Extract the optimal wavelet packet components
    def features(self, x):

        if x.ndim != 1:
            raise ValueError("Signal dimension must be 1")
        n = len(x)

        # Standarize the time series
        mean = np.mean(x)
        std = np.std(x)
        x = (x - mean) / std

        # Define the maximum tree depth
        maxDepth = int(np.floor(np.log2(n)) - 2)

        # Create a wavelet packet object
        wp = WaveletPacket(data=x, wavelet='dmey', mode='reflect', maxlevel=maxDepth)

        # Perform Best Basis Selection
        bbs = Node()
        bbs.construct(wp)
        bbs.prune(entropy)

        if bbs.has_children():
            paths = bbs.get_leafs()
        else:
            paths = [n.path for n in wp.get_level(maxDepth, 'freq')]

        m = len(paths)

        # Project the signal onto the best basis
        X = np.zeros((m, n), dtype=x.dtype)
        for k, path in enumerate(paths):
            wp_k = WaveletPacket(data=None, wavelet='dmey', mode='reflect', maxlevel=maxDepth)
            wp_k[path] = wp[path].data
            x_k = wp_k.reconstruct(update=False)
            x_k = x_k[0:n]
            X[k, ...] = x_k

        # Recover the original signal before standardization
        X[0, ...] = X[0, ...] * std + mean
        X[1:m, ...] = X[1:m, ...] * std

        # Estimate the power spectral density using Welchs method
        window = signal.get_window('parzen', math.floor(n))
        S = np.abs(fftpack.fft(X * window, axis=1)[..., 0:int(n / 2)])

        return X, S

    # Calculate the affinity matrix
    def affinity(self, S):

        if S.ndim != 2:
            raise ValueError("Feature matrix dimension must be equal to 2")

        m = S.shape[0]
        n = S.shape[1]
        W = np.zeros((m, m))
        for k in range(m):
            for l in range(k, m):
                # Histogram intersection kernel calculation
                W[k, l] = 4 * np.pi / n * hist(S[k, ...], S[l, ...])
                W[l, k] = W[k, l]

        # Graph stationary distribution
        D = np.diag(np.sum(W, axis=1))
        weights = D.diagonal() / np.linalg.norm(D.diagonal(), 1)

        return W, weights

    # Calculate the Laplacian embedding
    def embed(self, W):

        # Graph Laplacian matrix
        D = np.diag(np.sum(W, axis=1))
        L = D - W

        # Laplacian matrix eigenvalues
        eigValues, eigVectors = linalg.eig(L, D)
        eigValues = eigValues.real
        eigVectors = eigVectors.real
        eigVectors = eigVectors[:, np.argsort(eigValues)]

        # Calculate the Laplacian embedding
        Y = eigVectors[:, 1:self.nIMFs]
        l2_norm = np.linalg.norm(Y, axis=1, ord=2)
        for k in range(Y.shape[0]):
            Y[k, ...] = Y[k, ...] / max(l2_norm[k], 1e-3)

        return Y

    # Perform the decomposition
    def decompose(self, x):

        if x.ndim != 1:
            raise ValueError("Signal dimension must be equal to 1")

        # Calculate the feature matrix
        X, S = self.features(x)

        # Calculate the affinity matrix
        W, weights = self.affinity(S)

        # Embed the data
        Y = self.embed(W)

        # Perform k-segmentation
        labels = ksegments(Y.T, weights, k=self.nIMFs)

        # Calculate the IMFs
        IMFs = np.zeros((self.nIMFs, X.shape[1]), dtype=x.dtype)
        for k in range(len(weights)):
            IMFs[labels[k], ...] += X[k, ...]

        # Sort the IMFs based on their central frequency
        IMFs = IMFs[::-1]

        return IMFs

    # Function to calculate the corresponding frequency bands
    def freq_bands(self, paths):

        # Create a list of boundary frequencies
        freqs = np.zeros(len(paths) + 1)
        k = 0

        # Number of wavelet packet components
        N = len(paths)

        # For every component
        for i in range(N):
            # Calculate the bandwidth of the filter
            j = len(paths[i])
            bw = np.pi * 2 ** (-j)

            freqs[i + 1] = freqs[i] + bw

        return freqs


# Function to calculate the instantaneous phase of a signal
def angle(x):
    # Calculate the analytical extension
    x_a = signal.hilbert(x)

    # Estimate the instantaneous phase
    phi = np.unwrap(np.angle(x_a))

    return phi


# Function to calculate the mean phase coherence between two signals
def MPC(x, y):
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError('Signal dimension must be equal to 1')
    elif len(x) != len(y):
        raise ValueError('Signals must have equal length')

    # Time series length
    N = len(x)

    # Estimate the instantaneous phase
    phi_x = angle(x)
    phi_y = angle(y)

    # plt.hist((phi_x - phi_y + np.pi) % 2 * np.pi - np.pi, bins=40, facecolor='y', edgecolor='k')
    # plt.show()

    return 1 / N * np.abs(np.sum(np.exp(1j * (phi_x - phi_y))))


# Function to calculate the mean phase coherence between two multicomponent signals
def MC_MPC(x, y, K):
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError('Signal dimension must be equal to 1')
    elif len(x) != len(y):
        raise ValueError('Signals must have equal length')

    # Construct the complex signal
    z = x + 1j * y

    # Decompose the signals using MSO-WT 
    msowt = MSOWT(nIMFs=K)
    IMFs = msowt.decompose(z)
    xIMFs = IMFs.real
    yIMFs = IMFs.imag

    # Calculate the mean phase coherence array
    R = np.empty(K)
    for k in range(K):
        R[k] = MPC(xIMFs[k, ...], yIMFs[k, ...])

    # Return the maximum value
    return np.max(R)

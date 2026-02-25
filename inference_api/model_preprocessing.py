# model_preprocessing.py

# imports
import numpy as np
from torch.utils.data import Dataset

class ZScoreNormalizer:
    """
    Z-score normalizer for spatiotemporal tensors.

    This class computes a global mean and standard deviation from the
    training data and applies standardization to inputs and targets.
    It also supports inverse transformation back to physical units.
    """

    def __init__(self, eps=1e-8):
        """
        Initialize the normalizer.

        Parameters
        ----------
        eps : float, optional
            Small constant added to the standard deviation to avoid
            division by zero. Default is 1e-8.
        """

        self.mu = None
        self.sigma = None
        self.eps = eps

    def fit(self, X):
        """
        Compute mean and standard deviation from training data.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor used for fitting the normalizer.
            Expected shape: (N, T, C, H, W) or compatible.

        Returns
        -------
        self : ZScoreNormalizer
            Fitted normalizer.
        """

        self.mu = X.mean()
        self.sigma = X.std() + self.eps
        return self

    def transform(self, X):
        """
        Apply z-score normalization.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor to be normalized.

        Returns
        -------
        torch.Tensor
            Normalized tensor.
        """

        if self.mu is None or self.sigma is None:
            raise RuntimeError("Normalizer must be fitted first.")

        return (X - self.mu) / self.sigma

    def fit_transform(self, X):
        """
        Fit the normalizer and apply normalization.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor used for fitting and transformation.

        Returns
        -------
        torch.Tensor
            Normalized tensor.
        """

        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        """
        Convert normalized values back to the original physical scale.

        Parameters
        ----------
        X : torch.Tensor
            Normalized tensor.

        Returns
        -------
        torch.Tensor
            Tensor in the original physical units.
        """

        if self.mu is None or self.sigma is None:
            raise RuntimeError("Normalizer must be fitted first.")
        return X * self.sigma + self.mu


class DHWDataset(Dataset):
    """
    PyTorch Dataset for Deep Heatwave (DHW) prediction.

    Each sample consists of a temporal sequence of spatial fields
    and the corresponding target field at the next time step.
    """

    def __init__(self, X, y):
        """
        Initialize the dataset.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor with shape (N, T, C, H, W),
            where N is the number of samples and T is the sequence length.
        y : torch.Tensor
            Target tensor with shape (N, C, H, W).
        """

        self.X = X
        self.y = y

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """

        return self.X.shape[0]

    def __getitem__(self, idx):
        """
        Retrieve a single sample from the dataset.

        Parameters
        ----------
        idx : int
            Index of the sample.

        Returns
        -------
        tuple
            (X_i, y_i), where:
            - X_i has shape (T, C, H, W)
            - y_i has shape (C, H, W)
        """

        return self.X[idx], self.y[idx]


def make_sequences(data, seq_len=12):
    """
    Create input-output sequences for supervised temporal learning.

    Given a time-ordered spatial dataset, this function builds
    sliding windows of length `seq_len` as inputs and the subsequent
    time step as the target.

    Parameters
    ----------
    data : numpy.ndarray
        Input array with shape (time, latitude, longitude).
    seq_len : int, optional
        Length of the input temporal sequence. Default is 12.

    Returns
    -------
    X : numpy.ndarray
        Input sequences with shape (N, seq_len, 1, H, W).
    y : numpy.ndarray
        Target fields with shape (N, 1, H, W).
    """

    X, y = [], []

    for t in range(seq_len, len(data)):
        X.append(data[t-seq_len:t])  # 12 months
        y.append(data[t])            # next month

    X = np.array(X)[:, :, None, :, :]
    y = np.array(y)[:, None, :, :]

    return X, y
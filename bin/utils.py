from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import AutoMinorLocator
from scipy.special import binom
from scipy.stats import moment

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times New Roman"],  # or any other serif font you prefer
        "font.size": 14,  # Set the default font size
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.major.width": 1,
        "ytick.major.width": 1,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "xtick.minor.width": 1,
        "ytick.minor.width": 1,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)

plt.minorticks_on()
# Set the minor tick frequency globally
plt.gca().xaxis.set_minor_locator(AutoMinorLocator(2))
plt.gca().yaxis.set_minor_locator(AutoMinorLocator(2))


def bootstrap(x, Nboot, binsize):
    rng = np.random.default_rng()
    boots = []
    x = x.reshape(-1, binsize, *x.shape[1:])
    for i in range(Nboot):
        x_bin = x[rng.integers(len(x), size=len(x))]
        boots.append(np.mean(x_bin, axis=(0, 1)))
    return np.mean(boots), np.std(boots)


def jackknife(samples: np.ndarray):
    """Return mean and estimated lower error bound."""
    means = []

    for i in range(samples.shape[0]):
        means.append(np.delete(samples, i, axis=0).mean(axis=0))

    means = np.asarray(means)
    mean = means.mean(axis=0)
    error = np.sqrt((samples.shape[0] - 1) * np.mean(np.square(means - mean), axis=0))

    return mean, error


def grab(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def stats(samples):
    return np.mean(samples, axis=0), np.std(samples, axis=0) / np.sqrt(samples.shape[0])


class Scaler:
    """
    Used to scale the data to zero mean and unit variance for training performance purposes
    """

    def __init__(self) -> None:
        self.mean = 0
        self.std = 1.0

    def transform(self, input_data, axis=0):
        self.mean = np.mean(input_data, axis=axis)
        self.std = np.std(input_data, axis=axis)
        return (input_data - self.mean) / self.std

    def invert_transform(self, input_data):
        return input_data * self.std + self.mean


def nth_cumulants(data, order, axis=0):
    """
    Calculate the first few cumulants of a dataset up to the specified order.

    Parameters:
    data (array-like): Input data
    order (int): The highest order of cumulants to calculate

    Returns:
    list: A list of cumulants from 1 to the specified order
    """
    cumulants_list = [np.mean(data, axis=axis)]  # First cumulant is the mean

    for n in range(2, order + 1):
        kappa = moment(data, n, axis=axis)
        if n > 2:
            for i in range(1, n):
                kappa -= (
                    binom(n - 1, i - 1)
                    * cumulants_list[i - 1]
                    * moment(data, n - i, axis=axis)
                )
        cumulants_list.append(kappa)

    return cumulants_list


# def jackknife_cumulants(data, order, axis=0):
# 	n_samples = len(data)
# 	jackknife_cumulants = []

# 	for i in range(n_samples):
# 		jackknife_sample = np.delete(samples, i, axis=axis)
# 		jackknife_cumulants.append(nth_cumulants(jackknife_sample, order))

# 	jackknife_cumulants = np.squeeze(jackknife_cumulants)
# 	cumulants_means = np.mean(jackknife_cumulants, axis=axis)

# 	jackknife_errors = np.sqrt(n_samples - 1) * np.mean((jackknife_cumulants - cumulants_means)**2, axis=axis)

# 	return cumulants_means, jackknife_errors


def bootstrap_cumulants(data, order, axis=0, n_boot=1000):
    n_samples = len(data)
    boot_cumulants = []

    for i in range(n_boot):
        boot_sample = np.random.choice(data, size=n_samples, replace=True)
        boot_cumulants.append(nth_cumulants(boot_sample, order, axis=axis))

    boot_cumulants = np.squeeze(boot_cumulants)
    cumulants_means = np.mean(boot_cumulants, axis=axis)

    boot_errors = np.std(boot_cumulants, axis=axis)

    return cumulants_means, boot_errors


# from scipy.special import binom

# def nth_cumulants(data, order, axis=0):
#     """
#     Calculate the first few cumulants of a dataset up to the specified order.

#     Parameters:
#     data (array-like): Input data
#     order (int): The highest order of cumulants to calculate

#     Returns:
#     list: A list of cumulants from 1 to the specified order
#     """
#     cumulants_list = [np.mean(data, axis=axis)]  # First cumulant is the mean

#     for n in range(2, order + 1):
#         kappa = moment(data, n, axis=axis)
#         if n > 2:
#             for i in range(1, n):
#                 kappa -= (binom(n - 1, i - 1) * cumulants_list[i - 1] *
#                           moment(data, n-i, axis=axis))
#         cumulants_list.append(kappa)

#     return cumulants_list

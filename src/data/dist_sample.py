import numpy as np
from scipy.stats import norm, expon

# TODO: If we want to make this cleaner, we could define a class that samples given some
# parameters and the scipy probability function, and has discretize as a method.

def discretize(data: np.ndarray, bins: int):
    """
    Returns the probabilities of the discrete distribution given by the data
    split into the specified number of bins.

    The bins are determined by equally spaced blocks between the data's minimum 
    and maximum values.
    """
    # Get the value of the probability density function of the equally spaced bins
    density, _ = np.histogram(data, bins=bins, range=(data.min(), data.max()), density=True)

    # Normalize density values since they don't add up to 1. See NumPy's documentation.
    density_norm = density / density.sum()

    return density_norm


def sample_norm(mu: float | int, sigma: float | int, n: int, bins: int, seed: int = 1925) -> np.ndarray:
    """
    Samples from the normal distribution and splits the samples into bins.
    """
    # Generate the random variables
    samples = norm.rvs(loc=mu, scale=sigma, size=n, random_state=seed)

    # Split the values into bins
    samples_discrete = discretize(samples, bins=bins)

    return samples_discrete


def sample_expon(lambda_val: float | int, n: int, bins: int, seed: int = 1925) -> np.ndarray:
    """
    Samples from the exponential distribution and splits the samples into bins.
    The probability density functions is:
    
        f(x; lambda) = lambda * exp(-lambda * x); x >= 0, lambda > 0

    Arguments:
        lambda_val - Distribution parameter
        n - Number of samples to generate
        bins - Number of bins to discretize the samples
    """
    # Generate the random variables
    samples = expon.rvs(scale=1/lambda_val, size=n, random_state=seed)

    # Split the values into bins
    samples_discrete = discretize(samples, bins=bins)

    return samples_discrete

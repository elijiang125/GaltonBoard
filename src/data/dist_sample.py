import numpy as np
from scipy.stats import norm, expon


def discretize(data: np.ndarray, bins: int):
    """
    Returns the probabilities of the discrete distribution given by the data
    split into the specified number of bins.

    The bins are determined by equally spaced blocks between the data's minimum 
    and maximum values.
    """
    # Get the value of the probability density function of the equally spaced bins
    hist_density, _ = np.histogram(data, bins=bins, range=(data.min(), data.max()), density=True)


def sample_expon(lambda: float | int, n: int, bins: int, seed: int = 1925) -> np.ndarray:
    """
    Samples from the exponential distribution and splits the samples into bins.
    The probability density functions is:
    
        f(x; lambda) = lambda * exp(-lambda * x); x >= 0, lambda > 0

    Arguments:
        lambda - Distribution parameter
        n - Number of samples to generate
        bins - Number of bins to discretize the samples
    """
    # Generate the random variables
    samples = expon.rvs(scale=1/lambda, size=n, random_state=seed)

    # Split the values into bins
    samples_discrete = discretize(samples)

    return samples_discrete

import numpy as np
from scipy.stats import norm, expon


def sample_expon(lambda: float | int = 1, n: int = 10000, seed: int = 1925) -> np.array:
    """
    Samples from the exponential distribution. The probability
    density functions is:
    
        f(x; lambda) = lambda * exp(-lambda * x); x >= 0, lambda > 0

    Arguments:
        lambda - Distribution parameter
        n - Number of samples to generate
    """
    # Generate the random variables
    samples = expon.rvs(scale=1/lambda, size=n, random_state=seed)

    # Save them to file
    return

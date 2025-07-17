from math import asin, sqrt 


def triangular_number(n):
    """
    Computes the triangular number T_n
    """
    return int(n*(n + 1)/2)

def angle_from_prob(p: float):
    """
    Returns the angle phi that produces a probability of p when Rx(phi) is applied.
    """
    return 2*asin(sqrt(p))

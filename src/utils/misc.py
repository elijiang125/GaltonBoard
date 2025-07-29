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


def count_mcm(levels):
    """
    Returns the number of mid-circuit measurements for a circuit of given levels.

    Calculation is based on the number provided by the paper with some adjustments
    for the way in which our circuits are built.
    """
    # Number of pegs
    num_pegs = triangular_number(levels - 1)

    # For row i, (i-1)-many reset gates
    row_resets = [row_i - 1 for row_i in range(1, levels)]

    return int(num_pegs - 1 + sum(row_resets))

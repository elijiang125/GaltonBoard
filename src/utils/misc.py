import scipy as sp
from math import asin, sqrt
from omegaconf import DictConfig

def triangular_number(n: int) -> int:
    """
    Computes the triangular number T_n
    """
    return int(n*(n + 1)/2)

def angle_from_prob(p: float) -> float:
    """
    Returns the angle phi that produces a probability of p when Rx(phi) is applied.
    """
    return 2*asin(sqrt(p))


def count_mcm(levels: int) -> int:
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


def infer_levels(data_mat: sp.sparse.spmatrix) -> int:
    """
    Infers the number of levels on the circuit used to produce the simulations.

    Assumes each row corresponds to one simulations.
    """
    n_cols = data_mat.shape[1]

    # Check which number of levels make sense
    lvl = 2  # Starts with 2 because this is the minimum for a circuit
    n_states = 2**lvl
    Rx_n = triangular_number(lvl - 1)

    while (Rx_n + n_states) != n_cols:
        lvl += 1
        n_states = 2**lvl
        Rx_n = triangular_number(lvl - 1)

    return lvl


def add_mixed_states(state_obs: list):
    """
    Takes a list of observed values from pure states and returns
    the corresponding list of all possible states' values. 
    
    Mixed state observations are set to 0.
    """
    # Number of qubits is the number of pure states
    num_qubits = int(len(state_obs))
    
    # Add mixed states
    all_obs = [0 for i in range(2**num_qubits)]  # Initialize possible states' observations
    for idx, val in enumerate(state_obs):
        all_obs[2**idx] = val  # Update the corresponding pure state

    return all_obs

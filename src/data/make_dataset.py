import logging
import numpy as np
import joblib as jlb
from tqdm import tqdm
from pathlib import Path
from scipy import sparse
from sklearn.model_selection import train_test_split

from circuit import quantum_galton_board as qgb
from utils.misc import triangular_number


def sim_run(levels: int, num_shots: int, Rx_n: int) -> np.ndarray:
    """
    Run a single circuit simulation.

    Returns a NumPy array of size (Rx_n + 2**levels, )
    """
    # Generate a random vector of probabilities and corresponding angles
    p_vals = np.random.uniform(low=0, high=1, size=Rx_n)  # [0, 1)

    # Get the observed counts from running the circuit
    qc = qgb.build_galton_circuit(levels=levels, num_shots=num_shots, bias=p_vals)
    measured_states = qc()  # NumPy array of all possible states' probs in lexicographic order

    return sparse.hstack([p_vals, measured_states], format="lil")


def simulate_qgb(levels: int, num_shots: int, sims_n: int, results_path: Path) -> None:
    """
    Simulates the Quantum Galton Board (QGB) sims_n times.

    Arguments:
        levels - Number of levels for the simulated QGBs.
        num_shots - Number of shots to the simulated QGBs.
        sims_n - Number of times to simulate the QGB.
        results_path - Path to directory to store the CSV file with the results.

    Saves a Compressed Sparse Row (CSR) matrix of shape (sims, Rx_n + 2**levels) :
        - p-values, one for each Rx gate (prob1, ..., probn)
        - states of the outcomes (e.g. '100', '010', '001')

    The file is a compressed npz file.
    """
    Rx_n = triangular_number(levels - 1)  # Number of Rx gates used
    sparse_mat = sparse.lil_matrix((sims_n, Rx_n + 2**levels))
    
    # Run the QGB circuit sims_n times
    num_cpus = round(jlb.cpu_count()/2)
    obs_list = jlb.Parallel(n_jobs=num_cpus)(jlb.delayed(sim_run)(levels, num_shots, Rx_n) for sim in tqdm(range(sims_n)))

    # Substitute corresponding values on the sparse matrix
    for sim_idx, sim_row in enumerate(obs_list):
        sparse_mat[sim_idx, :] = sim_row

    # Convert to efficient format for computation
    sparse_csr = sparse_mat.tocsr()

    # Save the results from all simulations
    filepath = results_path.joinpath(f"obs_probs_levels{levels}_shots{num_shots}_sims{sims_n}.npz")
    filepath.parent.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(file=filepath, matrix=sparse_csr, compressed=True)


def process_simulations():
    """
    
    """
    pass

import logging
import numpy as np
import joblib as jlb
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split

from circuit import quantum_galton_board as qgb
from utils.misc import triangular_number


def sim_run(levels: int, num_shots: int, Rx_n: int) -> np.ndarray:
    """

    """
    # Generate a random vector of probabilities and corresponding angles
    p_vals = np.random.uniform(low=0, high=1, size=Rx_n)  # [0, 1)

    # Get the observed counts from running the circuit
    qc = qgb.build_galton_circuit(levels=levels, num_shots=num_shots, bias=p_vals, return_probs=False)
    obs_counts = qc()

    # Create the corresponding row of the final NumPy array
    obs_row = np.zeros((1, Rx_n + 2**levels))  # Row vector
    idxs = list(range(Rx_n)) + [int(idx_bin, 2) for idx_bin in obs_counts.keys()]  # Fill only the entries we got
    obs_row[0, idxs] = np.append(p_vals, list(obs_counts.values()))

    return obs_row


def simulate_qgb(levels: int, num_shots: int, sims_n: int, results_path: Path) -> None:
    """
    Simulates the Quantum Galton Board (QGB) sims_n times.

    Arguments:
        levels - Number of levels for the simulated QGBs.
        num_shots - Number of shots to the simulated QGBs.
        sims_n - Number of times to simulate the QGB.
        results_path - Path to directory to store the CSV file with the results.

    Saves a NumPy array of shape (sims_n, Rx_n + 2**levels) where
        - Rx_n is the number of Rx gates used by the circuit.
        - levels is also the number of qubits measured at the end of the circuit.

    The file is a NumPy non-compressed npz file.
    """
    Rx_n = triangular_number(levels - 1)  # Number of Rx gates used
    
    # Run the QGB circuit sims_n times
    num_cpus = round(jlb.cpu_count()/2)
    arr_list = jlb.Parallel(n_jobs=num_cpus)(jlb.delayed(sim_run)(levels, num_shots, Rx_n) for sim in tqdm(range(sims_n)))
    
    # Stack all rows of observed counts
    obs_arr = np.vstack(arr_list)

    # Save the results from all simulations
    filename = results_path.joinpath(f"obs_counts_levels{levels}_shots{num_shots}_sims{sims_n}.npz")
    np.savez(filename, obs_arr)


def process_simulations():
    """
    
    """
    pass

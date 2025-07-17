import logging
import numpy as np
import pandas as pd
import joblib as jlb
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split

from circuit import quantum_galton_board as qgb
from utils.misc import triangular_number


def sim_run(levels: int, num_shots: int, Rx_n: int) -> dict[str, int | float]:
    """

    """
    # Generate a random vector of probabilities and corresponding angles
    p_vals = np.random.uniform(low=0, high=1, size=Rx_n)  # [0, 1)

    # Get the observed counts from running the circuit
    qc = qgb.build_galton_circuit(levels=levels, num_shots=num_shots, bias=p_vals, return_probs=False)
    obs_counts = qc()

    # Add the probabilities used
    probs = {f"prob{idx+1}": pval for idx, pval in enumerate(p_vals)}

    return {**probs, **obs_counts}


def simulate_qgb(levels: int, num_shots: int, sims_n: int, results_path: Path) -> None:
    """
    Simulates the Quantum Galton Board (QGB) sims_n times.

    Arguments:
        levels - Number of levels for the simulated QGBs.
        num_shots - Number of shots to the simulated QGBs.
        sims_n - Number of times to simulate the QGB.
        results_path - Path to directory to store the CSV file with the results.

    Saves a Pandas DataFrame with columns:
        - p-values, one for each Rx gate (prob1, ..., probn)
        - states of the outcomes (e.g. '100', '010', '001')

    The file is a CSV file.
    """
    Rx_n = triangular_number(levels - 1)  # Number of Rx gates used
    
    # Run the QGB circuit sims_n times
    num_cpus = round(jlb.cpu_count()/2)
    obs_list = jlb.Parallel(n_jobs=num_cpus)(jlb.delayed(sim_run)(levels, num_shots, Rx_n) for sim in tqdm(range(sims_n)))

    # Create a DataFrame of the observations
    obs_df = pd.DataFrame(obs_list).fillna(0)

    # Save the results from all simulations
    filepath = results_path.joinpath(f"obs_counts_levels{levels}_shots{num_shots}_sims{sims_n}.csv")
    filepath.parent.mkdir(parents=True, exist_ok=True)
    obs_df.to_csv(filepath, index=False)


def process_simulations():
    """
    
    """
    pass

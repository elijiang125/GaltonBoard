import logging
import numpy as np
import scipy as sp
import joblib as jlb
from tqdm import tqdm
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from circuit import quantum_galton_board as qgb
from utils.misc import triangular_number, infer_levels


def sim_run(levels: int, num_shots: int, Rx_n: int, add_noise: bool = False) -> np.ndarray:
    """
    Run a single circuit simulation.

    Returns a NumPy array of size (Rx_n + 2**levels, )
    """
    # Generate a random vector of probabilities and corresponding angles
    p_vals = np.random.uniform(low=0, high=1, size=Rx_n)  # [0, 1)

    # Get the observed counts from running the circuit
    qc = qgb.build_galton_circuit(levels=levels, num_shots=num_shots, bias=p_vals, add_noise=add_noise)
    measured_states = qc()  # NumPy array of all possible states' probs in lexicographic order

    return sp.sparse.hstack([p_vals, measured_states], format="lil")


def simulate_qgb(levels: int, num_shots: int, sims_n: int, raw_basename: str, add_noise: bool = False) -> Path:
    """
    Simulates the Quantum Galton Board (QGB) sims_n times.

    Arguments:
        levels - Number of levels for the simulated QGBs.
        num_shots - Number of shots to the simulated QGBs.
        sims_n - Number of times to simulate the QGB.
        raw_basename - Absolute path with basename used for the saved file.
        add_noise - Whether to run the circuit with a noise model.

    Saves a COO sparse matrix of shape (sims, Rx_n + 2**levels) :
        - p-values, one for each Rx gate (prob1, ..., probn)
        - states of the outcomes (e.g. '100', '010', '001')

    The file is a compressed npz file.
    """
    # Make an empty sparse matrix
    Rx_n = triangular_number(levels - 1)  # Number of Rx gates used
    sparse_mat = sp.sparse.lil_matrix((sims_n, Rx_n + 2**levels))  # List-of-lists because we need to slice it
    
    # Run the QGB circuit sims_n times
    num_cpus = round(jlb.cpu_count()/2)
    obs_list = jlb.Parallel(n_jobs=num_cpus)(jlb.delayed(sim_run)(levels, num_shots, Rx_n, add_noise) for sim in tqdm(range(sims_n)))

    # Substitute corresponding values on the sparse matrix
    for sim_idx, sim_row in enumerate(obs_list):
        sparse_mat[sim_idx, :] = sim_row

    # Save the results from all simulations
    coo_mat = sparse_mat.tocoo()  # Change for saving only
    filepath = Path(f"{raw_basename}.npz")
    filepath.parent.mkdir(parents=True, exist_ok=True)
    sp.sparse.save_npz(file=filepath, matrix=coo_mat, compressed=True)

    return filepath


def dataset_split(dataset_path: Path, outputs_path: Path, train_size: float, test_size: float, val_size: float) -> None:
    """
    Splits the dataset saved on the given path with the specified sizes.

    Saves the splits as sparse matrices into separate compressed npz files.
    """
    # Check that the sizes add up to 1
    if (train_size + test_size + val_size) != 1.0:
        print("Error: train, validation and size don't add up to 1")
        return

    # Load the dataset
    data_mat = sp.sparse.load_npz(dataset_path).toarray()

    # Separate features and targets
    levels = infer_levels(data_mat)
    Rx_n = triangular_number(levels - 1)

    targets_mat = data_mat[:, :Rx_n]  # Circuit's observations (e.g. probabilities)
    features_mat = data_mat[:, Rx_n:]  # Probabilities/biases used for the pegs

    # Split to get training data
    temp_size = test_size + val_size
    features_train, features_temp, targets_train, targets_temp = train_test_split(features_mat, 
                                                                                  targets_mat, 
                                                                                  test_size=temp_size, 
                                                                                  random_state=1925)

    # Further split to get testing and validation data
    features_val, features_test, targets_val, targets_test = train_test_split(features_temp, 
                                                                              targets_temp, 
                                                                              test_size=test_size,
                                                                              random_state=1925)

    # Fit the scaler with the training data
    scaler = StandardScaler()
    scaler.fit(features_train)
    
    # Transform all features with the same scaler to avoid data leakage
    features_train = scaler.transform(features_train)
    features_test = scaler.transform(features_test)
    features_val = scaler.transform(features_val)

    # Save the partitions to file
    outputs_path.mkdir(parents=True, exist_ok=True)
    
    features_train_path = outputs_path.joinpath("features_train.npz")
    features_test_path = outputs_path.joinpath("features_test.npz")
    features_val_path = outputs_path.joinpath("features_val.npz")
    
    targets_train_path = outputs_path.joinpath("targets_train.npz")
    targets_test_path = outputs_path.joinpath("targets_test.npz")
    targets_val_path = outputs_path.joinpath("targets_val.npz")

    np.savez(features_train_path, features_train)
    np.savez(features_test_path, features_test)
    np.savez(features_val_path, features_val)

    np.savez(targets_train_path, targets_train)
    np.savez(targets_test_path, targets_test)
    np.savez(targets_val_path, targets_val)

    # Save the StandardScaler too
    scaler_path = outputs_path.joinpath("scaler.joblib")
    jlb.dump(scaler, scaler_path)

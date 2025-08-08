import numpy as np
from math import log2
from pathlib import Path
from matplotlib import pyplot as plt


def plot_train_losses(losses_path: Path, outfile: Path) -> None:
    """
    """
    # Load the arrays
    npzfile = np.load(losses_path)
    train_losses = npzfile["train_losses"]
    val_losses = npzfile["val_losses"]

    # Plotting the training and validation losses
    epochs = train_losses.shape[0]

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.savefig(outfile)
    plt.clf()
    plt.close()


def plot_predictions(preds_path: Path, outfile: Path) -> None:
    """    
    """
    # Load the arrays
    npzfile = np.load(preds_path)
    test_preds = npzfile["predicted"]
    test_labels = npzfile["observed"]

    # Plot the losses
    plt.figure(figsize=(8, 8))
    plt.scatter(test_labels, test_preds, alpha=0.7)
    plt.xlabel('Observed (Actual) Values')
    plt.ylabel('Predicted Values')
    plt.savefig(outfile)
    plt.clf()
    plt.close()


def plot_histogram(dist_vals: np.ndarray, mixed_states: bool, plot_title: str, outfile: Path) -> None:
    """
    """
    # Make a dictionary of the observations. Keys are the computational basis
    num_obs = len(dist_vals)
    num_qubits = int(log2(num_obs))

    if mixed_states:
        obs_dict = {format(idx, f"0{num_qubits}b"): value for idx, value in enumerate(dist_vals)}

    else:
        obs_dict = {format(2**idx, f"0{num_qubits}b"): dist_vals[idx] for idx in range(num_qubits)}

    x_pos = list(range(len(obs_dict)))

    # Plot the observations
    plt.bar(x_pos, list(obs_dict.values()), color="mediumseagreen")
    plt.xticks(x_pos, list(obs_dict.keys()), rotation=45)
    plt.xlabel("State")
    plt.ylabel("Probability")
    plt.title(plot_title)
    plt.grid(axis='y', linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(outfile)
    plt.clf()
    plt.close()

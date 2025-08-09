import os
import torch
import hydra
from pathlib import Path
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from circuit import quantum_galton_board as qgb
from data.make_dataset import simulate_qgb, dataset_split
from models.model_run import train, test, dist_eval
from utils.misc import triangular_number, add_mixed_states
from utils.plot_utils import plot_train_losses, plot_predictions, plot_histogram

# Register new resolvers for the config files
OmegaConf.register_new_resolver("project_root", lambda: Path(__file__).resolve().parents[1])
OmegaConf.register_new_resolver("num_states", lambda lvl: 2**lvl)
OmegaConf.register_new_resolver("Rx_n", lambda lvl: triangular_number(lvl - 1))
OmegaConf.register_new_resolver("mean", lambda x, y: int((x + y)/2))


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    """
    # Generate a dataset from the circuit's outputs for random choices of parameters
    if cfg.run_mode == "generate":
        # Define some parameters based on configuration
        noise = True if cfg.circuit.name == "noisy" else False
        raw_data_basename = str(cfg.paths.raw_data_basename)
        processed_dir = Path(cfg.paths.processed_data_dir)
        
        # Run the simulations. Data will be saved to file
        data_filepath = simulate_qgb(levels=cfg.circuit.levels, 
                                     num_shots=cfg.circuit.num_shots, 
                                     sims_n=cfg.circuit.n_sims, 
                                     raw_basename=raw_data_basename,
                                     add_noise=noise)

        # Split the dataset. Data will be saved to file
        dataset_split(dataset_path=data_filepath, 
                      outputs_path=processed_dir,
                      train_size=cfg.dataset.train_size, 
                      test_size=cfg.dataset.test_size, 
                      val_size=cfg.dataset.val_size)

    # Train and test a neural network to predict the circuit's parameters
    elif cfg.run_mode == "train":
        # TODO: check if needed files from config exist

        # Instantiate the model, loss function and optimizer
        model_inst = instantiate(cfg.model)
        loss_fn_inst = instantiate(cfg.loss_fn)
        optimizer_inst = instantiate(cfg.optimizer, params=model_inst.parameters())

        # Read config parameters
        dataset_dir = Path(cfg.paths.processed_data_dir)
        models_dir = Path(cfg.paths.models_dir)
        max_epochs = int(cfg.train.num_epochs)
        batch_size = int(cfg.train.batch_size)

        # Train the model. Weights and losses are saved
        weights_path, losses_path = train(model=model_inst, 
                                          criterion=loss_fn_inst, 
                                          optimizer=optimizer_inst, 
                                          data_dir=dataset_dir, 
                                          num_epochs=max_epochs, 
                                          batch_size=batch_size, 
                                          train_outdir=models_dir)

        # Test the model. Predictions are saved
        # TODO: test if model instance is affected by train or if a copy is created
        preds_path = test(model=model_inst, 
                          criterion=loss_fn_inst, 
                          weights_path=weights_path, 
                          data_dir=dataset_dir,
                          test_outdir=models_dir,
                          batch_size=batch_size)

        # Make plots
        plots_dir = Path(cfg.paths.plots_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)

        losses_plot_path = plots_dir.joinpath("train_validate_losses.png")
        plot_train_losses(losses_path, outfile=losses_plot_path)

        pred_plot_path = plots_dir.joinpath("predicted_vs_observed.png")
        plot_predictions(preds_path, outfile=pred_plot_path)

    
    # Use a trained model to predict the biases for a given distribution
    elif cfg.run_mode == "sample_dist":
        # TODO: check if needed files from config exist
        
        # Load the model with it's weights
        model = instantiate(cfg.model)
        models_dir = Path(cfg.paths.models_dir)
        
        # TODO: make a new resolver to specify these paths since they're being used by multiple functions
        # Weights file path is reconstructed the same way it was created, 
        weights_path = models_dir.joinpath(f"{model.__class__.__name__}_weights.pt")
        model_weights = torch.load(weights_path, weights_only=True)
        model.load_state_dict(model_weights)

        # TODO: make a new resolver to specify this path as well
        # Load scaler used for the training data
        dataset_dir = Path(cfg.paths.processed_data_dir)
        scaler_path = dataset_dir.joinpath("scaler.joblib")

        # Sample from the target probability distribution
        target_dist = instantiate(cfg.distribution)

        # Add positions for mixed states to simulate the circuit's output
        target_dist = torch.tensor(add_mixed_states(target_dist), dtype=torch.float32)
    
        # Get the model's predicted circuit biases
        pred_biases = dist_eval(model=model, 
                                target_dist=target_dist, 
                                scaler_path=scaler_path)

        # Build a circuit with the predicted biases
        noise = True if cfg.circuit.name == "noisy" else False
        galton_circuit = qgb.build_galton_circuit(levels=int(cfg.circuit.levels), 
                                                  num_shots=int(cfg.circuit.num_shots), 
                                                  bias=pred_biases, 
                                                  add_noise=noise)
        circuit_dist = galton_circuit()

        # Plot the true target distribution
        plots_dir = Path(cfg.paths.plots_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)  # Should have been created during training already

        target_dist_plotfile = plots_dir.joinpath("target_dist.png")
        plot_histogram(dist_vals=target_dist.numpy(), 
                       mixed_states=noise, 
                       outfile=target_dist_plotfile, 
                       plot_title="Target distribution")

        # Plot the resulting distribution from the circuit
        circuit_dist_plotfile = plots_dir.joinpath("circuit_dist.png")
        plot_histogram(dist_vals=circuit_dist, 
                       mixed_states=noise, 
                       outfile=circuit_dist_plotfile,
                       plot_title="Circuit's output distribution")


if __name__ == "__main__":
    main()


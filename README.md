# Quantum Walks and Monte Carlo

This is the repository for the fina project of Womanium & Wiser Quantum Program 2025.

The code in this repository implements a quantum circuit for a fine-grained biased Quantum Galton Board (QGB) based on the paper: [Universal Statistical Simulator](https://arxiv.org/abs/2202.01735). A shallow Feed Forward Neural Network (FFNN) model is used to learn the parameters of the quantum circuit to produce any given probability distribution.

The scripts work on three stages: data generation, neural network training and target distribution prediction.

## Data generation

This runs multiple simulations of the circuit's output for randomly chosen parameters. These parameters are chosen at random on every simulation. The simulation's outputs are then stored under `data/raw` and the a train-validation-split is created and stored to be used during the neural network training.

## Neural network training

The data is loaded and the FFNN is trained. During the training, the QGB's output probabilities of the measured qubits are used as inputs and the circuit's parameters are used as labels. This means that, with enough data and training, the FFNN is capable of predicting the parameters needed to produce any given probability distribution.

## Target distribution prediction

The trained model is loaded and samples from a target distribution are then used to get predictions from the FFNN. These predictions are then used to run a circuit and the output probability distributions are plotted together with the target's probabilities for comparison.

# Installation

The Python environment with all the required packages can be installed using Pixi. To install Pixi, refer to their installation instruction on their [website](https://pixi.sh/latest/installation/). Clone this repository:

```
git clone https://github.com/elijiang125/GaltonBoard.git
```

And install the environment from within the repository's folder:

```
cd GaltonBoard
pixi install
```

After installation, the environment can be activated as follows:

```
pixi shell
```

# Running the scripts

The scripts use Hydra to leverage the configuration files in `src/config`. With the virtual environment activated, run the scripts with the configuration's default values as

```
python main.py
```

You can override the default configuration either by manually editing the configuration files, adding new files or through the command line. For example, to run 2000 simulations of a circuit with 4 levels and the default values for number of shots:

```
python main.py run_mode=generate circuit=ideal circuit.levels=4 circuit.n_sims=2000
```

If the configuration values are overriden for the data generation, the same values need to be used during training mode since the subfolder for the data is named based on the circuit's configuration. To train the FFNN with the model's defaut values using the data generated above run:

```
python main.py run_mode=train circuit=ideal circuit.levels=4 circuit.n_sims=2000
```

However, you could take advantage of Hydra's multi-run to generate and train on a given dataset as:

```
python main.py --multirun run_mode=generate,train circuit.levels=4 circuit.n_sims=2000
```

Trained models' weights are also stored on subfolders based on circuit's configuration, so to get predictions and plots for the exponential distribution using a model trained on the above configuration run

```
python main.py run_mode=sample_dist circuit.levels=4 circuit.n_sims=2000 distribution=expon
```

*NOTE*: It is not recommended to use the multi-run for the `sample_dist` mode together with the data generation and model training because Hydra will run all combinations of the different overriden parameters.

# Outputs

Besides the simulations data stored in `data/raw` and the splits in train, validation and test sets in `data/processed`, the script's outputs can be found on the `outputs` folder and consist of:

- Models' weights, loss values for training and validation, as well as predictions for the test dataset under `outputs/models`.
- Plots of the training and validations losses, predicted vs. observed on the test dataset and barplots for the target distribution and the circuit's output under `outputs/plots`

# Notebooks

We included notebooks under `notebooks/Results` showing the results for the deliverables for this project.

# Note on noisy circuits

Noise models are imported from Qiskit's GenericBackendV2 and 'qiskit.aer' device is used. This means that only circuits up to 5 levels are supported.

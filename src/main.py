from pathlib import Path
from data.make_dataset import simulate_qgb

simulate_qgb(levels=3, num_shots=1000, sims_n=5, results_path=Path("/home/jorpena/Documents/Quantum_Solvers/GaltonBoard/data/raw"))


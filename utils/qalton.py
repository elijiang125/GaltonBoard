import pennylane as qml
from pennylane import numpy as np


def reset_qubit() -> None:
    """
    Resets q0 to |0>.
    """
    # Mid-circuit measure
    m = qml.measure(0)
    qml.cond(m, qml.PauliX)(0)


def peg_gate(peg_wires: list) -> None:
    """
    Apply operators to simulate the results of a single peg.
    
    Takes a list with indices of exactly 4 wires where:
        - First wire is the control qubit
        - Second wire is on the left
        - Third wire is the middle qubit
        - Fourth wire is on the right
    """
    # 
    q0 = peg_wires[0]  # Control
    q1 = peg_wires[1]  # Left
    q2 = peg_wires[2]  # Middle
    q3 = peg_wires[3]  # Right
    
    # Apply the operators
    qml.CSWAP(wires=[q0, q1, q2])
    qml.CNOT(wires=[q2, q0])
    qml.CSWAP(wires=[q0, q2, q3])


def build_galton_circuit(levels: int, num_shots: int):
    """
    Simulate a Quantum Galton Board of specified levels.
    """
    num_wires = 2*levels
    dev = qml.device("default.qubit", wires=num_wires, shots=num_shots)

    @qml.qnode(dev)
    def circuit() -> np.ndarray:
        # Initial state
        qml.Hadamard(wires=[0])  # Induce superposition
        qml.PauliX(wires=[2])  # Ball's qubit

        # First peg
        peg_gate(peg_wires=[0, 1, 2, 3])

        return qml.probs(wires=[1, 3])

    return circuit



















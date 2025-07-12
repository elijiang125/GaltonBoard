import pennylane as qml

def reset_gate(idx: int, enable:bool = True) -> None:
    """
    Resets q_idx to |0>.
    Note: for hadamard quantum walk, cannot reset q0
    """
    if not enable:
        return

    # Mid-circuit measure
    m = qml.measure(idx)
    qml.cond(m, qml.PauliX)(idx)


def quantum_peg(peg_wires: list) -> None:
    """
    Apply operators to simulate the results of single peg.

    Takes a list with indices of exactly 4 wires.
    """
    # Control and input qubits
    q0 = peg_wires[0]  # Control
    q1 = peg_wires[1]  # Left
    q2 = peg_wires[2]  # Middle
    q3 = peg_wires[3]  # Right
    
    # Apply the operators
    qml.CSWAP(wires=[q0, q1, q2])
    qml.CNOT(wires=[q2, q0])
    qml.CSWAP(wires=[q0, q2, q3])

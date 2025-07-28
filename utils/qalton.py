import pennylane as qml
from pennylane import numpy as np
from math import asin, sqrt


def triangular_number(n):
    """
    Computes the triangular number T_n
    """
    return int(n*(n + 1)/2)


def reset_qubit(idx: int, enable:bool = True) -> None:
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


def level_pegs(qubits: list, phi_vals: list, coherence: bool) -> None:
    # 07/28 Update: Added coherence as a param
    """
    Applies all the quantum peg modules for a single level.
    Does this by taking triplets starting from the leftmost and moving to the right.
    Each triplet is used to represent the two possibilities for the ball (middle) to go through: left or right.
    """
    # Make sublists of input qubits' triplets
    q0 = qubits[0]
    qubits_triplets = [qubits[i:i + 3] for i in range(1, len(qubits) - 2, 2)]

    for tri_idx, triplet in enumerate(qubits_triplets):
        # Input qubits
        q1 = triplet[0]  # Left
        q2 = triplet[1]  # Middle - Ball
        q3 = triplet[2]  # Right

        # Apply the operators of a Quantum Peg
        quantum_peg(peg_wires=[q0, q1, q2, q3])

        # Return control qubit to original rotation
        if tri_idx + 1 < len(qubits_triplets):  # +1 is there because indices start at 0
            reset_qubit(0, enable=not coherence)  # Reset control qubit
            # 07/28 Update: enabled coherence
            if tri_idx < len(phi_vals): # if less than length of phi_vals -> RX
                qml.RX(phi_vals[tri_idx], wires=[q0])


def build_galton_circuit(levels: int, num_shots: int, bias: int | float | list = 0.5, coherence: bool = False, qiskit_noise_model=None, PL_noise = None, return_counts: bool=False):
    """
    Simulate a Quantum Galton Board of specified levels.
    """
    # 07/28 Update: Added qiskit_noise_model, PL_noise, and return_counts
    # Qiskit_noise_model & PL_noise  (PennyLane) are noise models. Not sure which one to use, so might as well experiment both
    # return_counts: use qml.counts for circuits w/ > 24 wires, qml.probs for anything else

    num_pegs = triangular_number(levels - 1)
    num_wires = 2*levels

    # 07/28 Update: Here, we must pick a device (ideal vs noisy)
    if qiskit_noise_model is not None:
        # Plan A: run on Qiskit Aer with its NoiseModel
        dev = qml.device(
            "qiskit.aer",
            wires=num_wires,
            shots=num_shots,
            noise_model=qiskit_noise_model,
        )
    else:
        # Path B: ideal OR default.mixed (noise added via transform)
        dev = qml.device(
            "default.mixed" if PL_noise is not None else "lightning.qubit",
            # Use PennyLane's if PL_noise has something, if not then use standard
            wires=num_wires,
            shots=num_shots,
        )
    qubits = list(range(num_wires))  # Local variable used in the inner function
    
    # Force the bias to a list to make it easier to work with single and multiple biases
    if isinstance(bias, float) or isinstance(bias, int):
        biases = [bias for i in range(num_pegs)]

    # Make sure that if multiple biases are provided, it matches the number needed for the levels specified
    elif len(bias) != num_pegs:
        print(f"{len(bias)} provided for {num_pegs} pegs")
        return

    # Compute angle(s) for the Rx gate
    phi_vals = [2*asin(sqrt(p)) for p in biases]

    @qml.qnode(dev)
    def circuit() -> np.ndarray:
        # Control and input qubits
        mid_idx = int(len(qubits)/2)
        q0 = qubits[0]  # Should always be 0, but for consistency let's keep it this way
        qb = qubits[mid_idx]  # Ball qubit

        # Initial state
        qml.RX(phi_vals[0], wires=[q0])  # Induce superposition
        qml.PauliX(wires=[qb])  # Start the ball in the middle

        # Let the ball fall through
        for lvl in range(2, levels + 1):  # +1 to keep the range inclusive
            # Specify the qubits involved on the current level
            side_wires = lvl - 1  # Number of wires needed to each side of the middle (ball) one
            left_range = mid_idx - side_wires
            right_range = mid_idx + side_wires + 1  # The +1 is there to make the slice inclusive on the right
            level_qubits = [q0] + qubits[left_range:right_range]  
            
            # Account for all possibilities in the current level
            Rx_needed = lvl - 2  # Number of Rx gates needed within the current level (# of spaces between pegs)
            Rx_used = triangular_number(Rx_needed) + 1  # Number of Rx gates used so far
            level_phi_vals = phi_vals[Rx_used:Rx_used + Rx_needed]
            level_pegs(level_qubits, level_phi_vals, coherence) # 07/28 Update: Added coherence param

            # Add leftover rotation for the final triplet and reset
            if lvl >= 3:
                # Draw a barrier for visualization
                qml.Barrier()

                # Start and end positions for range that gets us the triplets at the end that we need
                # For lvl = 3, we need 1 triplet at the end of the qubits list (left to right of the circuit)
                # For lvl = 4, we need 2 triplets at the end of the list, and so on...
                start_pos = len(level_qubits) - 2*lvl + 3  # len(level_qubits) - 1 - 2(lvl - 2)
                end_pos = len(level_qubits) - 2

                # Get the last lvl-2 level qubits' triplets
                for idx in range(start_pos, end_pos, 2):
                    # Slice the triplet
                    triplet = level_qubits[idx:idx + 3]

                    # Take the left(upper) and middle qubits on each selected triplet to apply CNOTs
                    q1 = triplet[0]  # Left
                    q2 = triplet[1]  # Middle

                    qml.CNOT(wires=[q2, q1])
                    reset_qubit(q2)
            
            # Reset the control qubit to |0> and apply Rx if there is a next level
            if lvl < levels:
                reset_qubit(0, enable = not coherence)  # Reset control qubit
                qml.RX(phi_vals[Rx_used + Rx_needed], wires=[q0])

        # 07/28 Update: IF we're doing noisy model, then we must do qml.counts to deepen n-levels
        # Note: qml.probs cannot do > 24 wires

        odd_wires = list(range(1, num_wires, 2))
        if return_counts:
            return qml.counts(wires=odd_wires)
        else:
            return qml.probs(wires=odd_wires)
        
    # 07/28 Update: so the PL noise model needs to be wrapped up as a QNode
    qnode = qml.Qnode(circuit, dev)
    if PL_noise is not None:
        qnode=qml.transforms.add_noise(qnode,PL_noise)

        
    return qnode


import pennylane as qml
from pennylane import numpy as np


def reset_control_qubit() -> None:
    """
    Resets q0 to |0>.
    """
    # Mid-circuit measure
    m = qml.measure(0)
    qml.cond(m, qml.PauliX)(0)


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

def peg_pair(control: int, left: int, mid: int) -> None:
    """
    A single CSWAP-CNOT pair
    """
    qml.CSWAP(wires=[control, left, mid])
    qml.CNOT(wires=[mid, control])

def level_pegs(qubits: list) -> None:
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

        # Return control qubit probability to 50% after every triplet exept the last one
        if tri_idx + 1 < len(qubits_triplets):  # +1 is there because indices start at 0
            qml.CNOT(wires=[q3, q0])


def build_galton_circuit(levels: int, num_shots: int):
    """
    Simulate a Quantum Galton Board of specified levels.
    """
    num_wires = 2*levels
    dev = qml.device("default.qubit", wires=num_wires, shots=num_shots)

    qubits = list(range(num_wires))  # Local variable used in the inner function

    @qml.qnode(dev)
    def circuit() -> np.ndarray:
        # Control and input qubits
        mid_idx = int(len(qubits)/2)
        q0 = qubits[0]  # Should always be 0, but for consistency let's keep it this way
        qb = qubits[mid_idx]  # Ball qubit
        h_count = 0 # Elizabeth: must keep track of the # of Hadamard gates implemented

        # Initial state
        qml.Hadamard(wires=[q0])  # Induce superposition
        h_count += 1 # Elizabeth: increment each H gate by 1
        qml.PauliX(wires=[qb])  # Start the ball in the middle

        quantum_peg([0, levels - 1, levels, levels + 1]) #Elizabeth: CSWAP + CNOT + CSWAP

        #Elizabeth: So must track indices for middle-section
        first_s = levels # control index for last CNOT of first section
        second_s = first_s + 1 # first target index for first CSWAP of the second section

        #--------------Okay so this is the middle section-------------

        if levels > 3:
            while h_count != levels - 2:
                h_count += 1
                reset_control_qubit()
                qml.Hadamard(wires=[0])
                print(f"We are now at the {h_count}-th Hadamard!")
            
                # so this is that yellow part where we need exactly two CSWAP-CNOT pairs
                peg_pair(0, first_s-2, first_s-1)
                peg_pair(0, first_s-1, first_s)
                print(f"Finished applying two pairs fo peg gates starting at wire {first_s}.") #debugging

                # okay now the green/red zone part this is going to suck
                firstCSWAP_index = second_s #cannot update this second_s, we need to use it to increement for above while loop

                while firstCSWAP_index > first_s:
                    qml.CSWAP(wires=[0, firstCSWAP_index, firstCSWAP_index + 1])
                    print(f"CSWAP(0, {firstCSWAP_index}, {firstCSWAP_index + 1})") #debugging
                    qml.CNOT(wires=[firstCSWAP_index, 0])
                    print(f"CNOT({firstCSWAP_index}, 0)") #debugging
                    firstCSWAP_index -= 1
                    
                #Final CSWAP
                qml.CSWAP(wires=[0, first_s, first_s + 1])
                
                # now we do the increments and decrements for the next middle layer
                first_s -= 1
                second_s += 1
        
        print(f"\n Left while loop because Hadamard layer is at {h_count}")
        # --------------Yay third section -------------------------------

        if levels > 2:
            reset_control_qubit()
            qml.Hadamard(wires=[0])

            for i in range(2 * (levels - 1) - 1):
                peg_pair(0, i+1, i+2)
            
            # last CSWAP
            qml.CSWAP(wires=[0, 2 * levels - 2, 2 * levels - 1])

        return qml.probs(wires=list(range(1, num_wires, 2)))
        # Let the ball fall through
        #for lvl in range(2, levels + 1):  # +1 to keep the range inclusive
            # Specify the qubits involved on the current level
            #side_wires = lvl - 1  # Number of wires needed to each side of the middle (ball) one
            #left_range = mid_idx - side_wires
            #right_range = mid_idx + side_wires + 1  # The +1 is there to make the slice inclusive on the right
            #level_qubits = [q0] + qubits[left_range:right_range]  
            
            # Account for all possibilities in the current level
            #level_pegs(level_qubits)

            # Reset the control qubit to |0> and apply Hadamard if there is a next level
            #if lvl < levels:
                #reset_control_qubit()
                #qml.Hadamard(wires=[q0])

    return circuit


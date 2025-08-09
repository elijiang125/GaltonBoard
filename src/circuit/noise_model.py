import pennylane as qml
from pennylane import numpy as np


def depolarization(p_flip, flip_wire):
    """
    Generalization of bit and phase-flips where also PauliZ can be applied with 
    probability p_flip.
    """
    if np.random.rand() < p_flip:
        gate_choice = np.random.choice([qml.PauliX, qml.PauliY, qml.PauliZ])
        gate_choice(wires=flip_wire)


def random_angle(p_rand, phi, prec):
    """
    Adds noise to an angle used as parameter for rotation with probability p_rand.
    """
    if np.random.rand() < p_rand:
        phi = phi + prec*np.random.rand()

    return phi


def noisy_rotation(phi, prec, rot_wire):
    """
    Performs a noisy rotation on the given wire with probability.
    """
    # Rotate with a random angle
    rand_angle = np.pi + prec*np.random.rand()
    qml.RX(rand_angle, wires=rot_wire)

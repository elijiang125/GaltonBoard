from pennylane import numpy as np
import pennylane as qml
from math import log2
import networkx as nx
from scipy.stats import entropy, expon, norm
from scipy.optimize import minimize
import sys
sys.path.append("src")

from circuit.quantum_galton_board import build_galton_circuit
# Some helper functions

def angle_to_bias(theta):
    return np.cos(theta / 2) ** 2

def extract_probs(probs):
    """
    Converts a 1D array of probabilities into a dictionary keyed by bitstrings.
    """
    num_qubits = int(np.log2(len(probs)))
    state_probs = {format(idx, f"0{num_qubits}b"): probs[idx] for idx in range(len(probs))}
    return state_probs

def build_galton_graph_parametrized(levels, angles):
    G = nx.DiGraph()
    node_id = lambda level, pos: f"{level}-{pos}"
    biases = angle_to_bias(np.array(angles))
    peg_index = 0

    for level in range(levels):
        for pos in range(level + 1):
            curr = node_id(level, pos)
            left = node_id(level + 1, pos)
            right = node_id(level + 1, pos + 1)

            b = biases[peg_index] if peg_index < len(biases) else 0.5

            G.add_edge(curr, left, weight=1 - b)
            G.add_edge(curr, right, weight=b)
            peg_index += 1

    return G

def simulate_distribution(G, levels):
    start = "0-0"
    dist = {node: 0.0 for node in G.nodes()}
    dist[start] = 1.0

    for level in range(levels):
        next_dist = {node: 0.0 for node in G.nodes()}
        for node in dist:
            for succ in G.successors(node):
                next_dist[succ] += dist[node] * G[node][succ]['weight']
        dist = next_dist

    final_nodes = [n for n in dist if n.startswith(f"{levels}-")]
    probs = np.array([dist[n] for n in final_nodes])
    return probs / np.sum(probs), final_nodes

def target_distribution(kind, size):
    if kind == "exponential":
        x_vals = np.linspace(0, 4, size)
        ref = expon.pdf(x_vals)
    elif kind == "normal":
        x_vals = np.linspace(-3, 3, size)
        ref = norm.pdf(x_vals)
    elif kind == "hadamard":
        ref = np.zeros(size)
        ref[size//2] = 1.0
    else:
        raise ValueError("Unknown distribution type")

    return ref / np.sum(ref)

def kl_divergence(p, q):
    p = np.clip(p, 1e-8, 1)
    q = np.clip(q, 1e-8, 1)
    return entropy(p, q)

# Cost Function

def variational_cost_fn(angles, levels, target_kind, dev=None):
    biases = angle_to_bias(np.array(angles))
    circuit = build_galton_circuit(levels=levels, num_shots=1000, bias=biases) if dev is None else build_galton_circuit(levels=levels, num_shots=1000, bias=biases, dev=dev)
    probs = circuit()
    target = target_distribution(target_kind, len(probs))
    return kl_divergence(probs, target)

# Optimize

def optimize_qaoa_params(levels, target_kind, dev=None, maxiter=200):
    from utils.misc import triangular_number
    num_pegs = triangular_number(levels - 1)
    init_angles = np.random.uniform(0, np.pi, size=num_pegs)

    result = minimize(
        variational_cost_fn,
        init_angles,
        args=(levels, target_kind, dev),
        method="Powell",
        options={"maxiter": maxiter, "disp": True}
    )

    final_angles = result.x
    biases = angle_to_bias(final_angles)
    circuit = build_galton_circuit(levels=levels, num_shots=1000, bias=biases) if dev is None else build_galton_circuit(levels=levels, num_shots=1000, bias=biases, dev=dev)
    final_probs = circuit()
    return final_probs, final_angles

# Main

def run_qaoa(levels, target_kind, layers=1, dev=None, maxiter=200, coherence=False, verbose=False):
    """
    Run QAOA-style optimization on a Galton Board.

    Args:
        levels (int): Number of levels in the Galton board.
        target_kind (str): One of 'exponential', 'normal', 'hadamard'.
        layers (int): Number of QAOA layers (repetitions of cost-mixer structure).
        dev: PennyLane device. If None, defaults to 'lightning.qubit'.
        maxiter (int): Max optimization steps.
        coherence (bool): Whether to use coherent evolution (no resets).
        verbose (bool): Print KL divergence and optimizer info.

    Returns:
        optimized_probs: Final output probability distribution.
        optimized_angles: Learned parameters.
        kl (float): Final KL divergence.
    """
    from utils.misc import triangular_number

    num_pegs = triangular_number(levels - 1)
    num_angles = num_pegs * layers

    # Setup device if not passed
    if dev is None:
        dev = qml.device("lightning.qubit", wires=2 * levels, shots=1000)

    # Cost function
    def cost_fn(angles):
        # Reshape if multiple layers
        if layers > 1:
            angles = np.array_split(angles, layers)
            probs_total = np.zeros(2 ** levels)
            for layer_angles in angles:
                bias = angle_to_bias(layer_angles)
                circuit = build_galton_circuit(levels, num_shots=1000, bias=bias, coherence=coherence)
                probs = circuit()
                probs_total += probs
            probs = probs_total / layers
        else:
            bias = angle_to_bias(angles)
            circuit = build_galton_circuit(levels, num_shots=1000, bias=bias, coherence=coherence)
            probs = circuit()

        target = target_distribution(target_kind, len(probs))
        return kl_divergence(probs, target)

    init_angles = np.random.uniform(0, np.pi, size=num_angles)

    result = minimize(cost_fn, init_angles, method="COBYLA", options={"maxiter": maxiter, "disp": verbose})
    final_angles = result.x

    if layers > 1:
        final_angles_split = np.array_split(final_angles, layers)
        probs_total = np.zeros(2 ** levels)
        for angles_layer in final_angles_split:
            bias = angle_to_bias(angles_layer)
            circuit = build_galton_circuit(levels, num_shots=1000, bias=bias, coherence=coherence)
            probs_total += circuit()
        final_probs = probs_total / layers
    else:
        bias = angle_to_bias(final_angles)
        circuit = build_galton_circuit(levels, num_shots=1000, bias=bias, coherence=coherence)
        final_probs = circuit()

    target = target_distribution(target_kind, len(final_probs))
    kl = kl_divergence(final_probs, target)

    if verbose:
        print(f"[QAOA-{layers} Layer] KL Divergence = {kl:.5f}")

    return final_probs, final_angles, kl
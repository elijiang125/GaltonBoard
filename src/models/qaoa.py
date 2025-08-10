from pennylane import numpy as np
import pennylane as qml
from pennylane import qcut
from math import log2
import networkx as nx
from scipy.stats import entropy, expon, norm
from scipy.optimize import minimize
from itertools import product
import sys
sys.path.append("src")
from typing import Optional

from circuit.quantum_galton_board import build_galton_circuit
# helpers

def angle_to_bias(theta):
    return np.cos(theta / 2) ** 2

def extract_probs(
    probs,
    *,
    measured_wires=None,
    num_bins=None,
    renormalize_onehot=True,
):
    """
    Turn a 1D probability vector from PennyLane into a dict keyed by bitstrings.
    If num_bins is given, compress to *one-hot bin labels* like '1000','0100',...

    Args
    ----
    probs : 1D array-like of shape (2**k,)
        Probabilities over k measured qubits (order is PL's default big-endian).
    measured_wires : list[int] | None
        Which qubits were measured. Only used for bookkeeping / debugging.
    num_bins : int | None
        If provided, return exactly `num_bins` keys that are one-hot strings.
        Any probability mass on non–one-hot strings is dropped (optionally
        renormalized to 1.0 over the one-hot support).
    renormalize_onehot : bool
        If True and num_bins is set, renormalize after dropping non–one-hot mass.

    Returns
    -------
    dict[str, float]
        If num_bins is None: all 2**k states.
        Else: exactly num_bins entries, e.g. {'1000': p0, '0100': p1, ...}.
    """
    import numpy as _np

    probs = _np.asarray(probs, dtype=float)
    k = int(_np.log2(probs.size))
    if probs.size != 2**k:
        raise ValueError("`probs` length must be a power of two.")

    # Full basis dictionary
    states = {format(i, f"0{k}b"): float(probs[i]) for i in range(2**k)}

    if num_bins is None:
        return states

    # Aggregate to one-hot bins
    bin_keys = [("0"*i) + "1" + ("0"*(num_bins-i-1)) for i in range(num_bins)]
    agg = {key: 0.0 for key in bin_keys}

    onehot_mass = 0.0
    for s, p in states.items():
        # keep only strings of Hamming weight 1 and length == num_bins
        if len(s) == num_bins and s.count("1") == 1:
            agg[s] += p
            onehot_mass += p

    if renormalize_onehot and onehot_mass > 0:
        for k_ in agg:
            agg[k_] /= onehot_mass

    return agg


def _expval_bitstring_with_cut(
    levels, bias, meas_wires, bitstring, cut_wires, dev, coherence
):   # NEW: allow passing PL noise
    """
    Returns Pr(meas_wires == bitstring) by building a QNode that:
      - queues the Galton board template on the given device
      - inserts WireCut at positions in cut_wires
      - returns expval(Projector([bitstring], wires=meas_wires))
      - applies qml.qcut.cut_circuit.transform to execute stitched fragments

    Notes:
      * Requires coherence=True (no mid-circuit resets).
      * Only 1 terminal measurement (an expval) → compatible with qml.qcut.
    """
    # make sure bitstring length matches meas_wires
    assert len(bitstring) == len(meas_wires), "bitstring length must match meas_wires"

    # 1) Build the full (coherent) QGB on *any* device – we only need its tape:
    @qml.qnode(dev)
    def qnode():
        # <<< THIS is the critical change: use the template so ops are queued >>>
        galton = build_galton_circuit(levels=levels, bias=bias, coherence=coherence)
        galton()                             # queues the Galton ops on this tape

        # place WireCut markers
        for w in cut_wires or []:
            qcut.WireCut(wires=w)

        # final scalar measurement
        return qml.expval(qml.Projector(bitstring, wires=meas_wires))

    # convert to a tape and cut
    tape = qnode.construct([], {})
    fragment_tapes, processing_fn = qcut.cut_circuit.transform(
        tape, device_wires=dev.wires
    )
    # execute each fragment and stitch
    results = [qml.execute([frag], dev, None)[0] for frag in fragment_tapes]
    return float(processing_fn(results))
def _probs_via_cut(levels, bias, meas_wires_subset, cut_wires, dev, coherence, fragment_noise_model=None):
    from itertools import product

    k = len(meas_wires_subset)
    probs = []
    for bits in product([0, 1], repeat=k):
        p = _expval_bitstring_with_cut(
            levels=levels,
            bias=bias,
            meas_wires=meas_wires_subset,
            bitstring=list(bits),
            cut_wires=cut_wires,
            dev=dev,
            coherence=coherence,
            fragment_noise_model=fragment_noise_model,   # NEW
        )
        probs.append(p)
    probs = np.array(probs, dtype=float)
    s = probs.sum()
    return probs / s if s > 0 else probs



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

# Cost Functions

def variational_cost_fn(angles, levels, target_kind, dev=None):
    biases = angle_to_bias(np.array(angles))
    circuit = build_galton_circuit(levels=levels, num_shots=1000, bias=biases) if dev is None else build_galton_circuit(levels=levels, num_shots=1000, bias=biases, dev=dev)
    probs = circuit()
    target = target_distribution(target_kind, len(probs))
    return kl_divergence(probs, target)

# Optimization

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

# MAIN
def run_qaoa(levels, target_kind, layers=1, dev=None, maxiter=200, coherence=False, verbose=False,
             cut=None, fragment_noise_model=None):
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
        cut: optional dict or None.
         When provided, enables circuit cutting on a small marginal.
         Expected keys:
           cut['wires']            → list of integers where to place qml.WireCut
           cut['measure_subset']   → list of wires to reconstruct as marginal (3–6 recommended)

    Returns:
        optimized_probs: Final output probability distribution.
        optimized_angles: Learned parameters.
        kl (float): Final KL divergence.
    """
    from utils.misc import triangular_number
    num_pegs = triangular_number(levels - 1)
    num_angles = num_pegs * layers

    if dev is None:
        dev = qml.device("lightning.qubit", wires=2 * levels, shots=1000)

    use_cutting = cut is not None
    cut_wires = (cut or {}).get("wires", [])
    meas_subset = (cut or {}).get("measure_subset", None)

    if use_cutting:
        if not coherence:
            raise ValueError("Circuit cutting requires coherence=True (no mid-circuit resets).")
        if meas_subset is None or len(meas_subset) == 0:
            # sensible default: the last 3 data wires
            meas_subset = list(range(2*levels - 3, 2*levels))
        # make sure all wires exist
        for w in cut_wires + meas_subset:
            if w < 0 or w >= 2*levels:
                raise ValueError(f"Wire {w} is out of bounds for 2*levels = {2*levels}.")

    # cost function uses either full probs (no cut) or marginal via cut
    def cost_fn(angles):
        bias = angle_to_bias(angles)
        if use_cutting:
            probs = _probs_via_cut(
                levels, bias, meas_subset, cut_wires, dev,
                coherence=True,
                fragment_noise_model=fragment_noise_model,   # NEW
            )
            target = target_distribution(target_kind, len(probs))
            return kl_divergence(probs, target)

    init_angles = np.random.uniform(0, np.pi, size=num_angles)
    result = minimize(cost_fn, init_angles, method="Powell",
                  options={"maxiter": 500, "disp": True})
    final_angles = result.x

    # Final eval
    if layers > 1:
        angles_split = np.array_split(final_angles, layers)
        if use_cutting:
            probs_total = 0
            for layer_angles in angles_split:
                bias = angle_to_bias(layer_angles)
                p = _probs_via_cut(levels, bias, meas_subset, cut_wires, dev, coherence=True)
                probs_total += p
            final_probs = probs_total / layers
        else:
            probs_total = 0
            for layer_angles in angles_split:
                bias = angle_to_bias(layer_angles)
                circuit = build_galton_circuit(levels, num_shots=1000, bias=bias, coherence=coherence)
                probs_total += circuit()
            final_probs = probs_total / layers
    else:
        bias = angle_to_bias(final_angles)
        if use_cutting:
            final_probs = _probs_via_cut(levels, bias, meas_subset, cut_wires, dev, coherence=True)
        else:
            circuit = build_galton_circuit(levels, num_shots=1000, bias=bias, coherence=coherence)
            final_probs = circuit()

    target = target_distribution(target_kind, len(final_probs))
    kl = kl_divergence(final_probs, target)

    if verbose:
        tag = f"QAOA-{layers} Layer"
        cut_tag = f" + CUT(marginal|wires={meas_subset})" if use_cutting else ""
        print(f"[{tag}{cut_tag}] KL Divergence = {kl:.5f}")

    # return distro + angles + meta
    return final_probs, final_angles, kl

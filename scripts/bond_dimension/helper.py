from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, Operator, Pauli
import numpy as np

from mqt.yaqs import simulator
from mqt.yaqs.core.libraries.circuit_library import create_ising_circuit, create_2d_ising_circuit, create_heisenberg_circuit, create_2d_heisenberg_circuit
from mqt.yaqs.core.data_structures.networks import MPS
from mqt.yaqs.core.data_structures.simulation_parameters import Observable, StrongSimParams
from mqt.yaqs.core.libraries.gate_library import Z

import copy
import numpy as np
import matplotlib.pyplot as plt
import time


from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, SparsePauliOp

from matplotlib.colors import LogNorm


def _mid_z_operator(num_qubits):
    """Helper to build a SparsePauliOp for Z on the middle qubit."""
    mid = num_qubits // 2
    label = ['I'] * num_qubits
    label[mid] = 'Z'
    return SparsePauliOp(''.join(label))

# def state_vector_simulator(circ):
#     # Run statevector simulator
#     sim = AerSimulator(method='statevector')
#     result = sim.run(circ).result()
#     sv = Statevector(result.get_statevector(circ))

#     # build Z_mid and compute expectation
#     z_mid = _mid_z_operator(circ.num_qubits)
#     exp_val = sv.expectation_value(z_mid).real

#     return sv, exp_val

from qiskit import QuantumCircuit


from qiskit import transpile
from qiskit_aer import AerSimulator

def tebd_simulator(circ, max_bond, threshold, initial_state=None):
    threshold = 1e-15
    circ2 = copy.deepcopy(circ)
    circ2.clear()
    if initial_state is not None:
        circ2.set_matrix_product_state(initial_state)
    circ2.append(circ, range(circ.num_qubits))
    circ2.save_matrix_product_state(label='final_mps')

    # 4. Build & transpile for the MPS backend
    sim = AerSimulator(
        method='matrix_product_state',
        matrix_product_state_max_bond_dimension=max_bond,
        matrix_product_state_truncation_threshold=threshold
    )
    tcirc = transpile(circ2, sim)

    # 5. Run & extract
    result = sim.run(tcirc).result()
    mps = result.data(0)['final_mps']
    bonds = [lam[0].shape[0] for lam in mps[0][1::]]
    return mps, bonds


def tdvp_simulator(circ, max_bond, threshold, pad, initial_state=None):
    if initial_state is None:
        initial_state = MPS(length=circ.num_qubits)
        initial_state.pad_bond_dimension(pad)
    measurements = [Observable(Z(), circ.num_qubits//2)]
    sim_params = StrongSimParams(measurements, num_traj=1, max_bond_dim=max_bond, threshold=threshold, get_state=True)
    simulator.run(initial_state, circ, sim_params, noise_model=None)
    mps = sim_params.output_state
    
    bonds = [tensor.shape[1] for tensor in mps.tensors[1::]]

    # sv = mps.to_vec()
    # exp_val = sim_params.observables[0].results[0]
    return mps, bonds


def generate_ising_observable_data(num_qubits, J, g, dt, pad, thresholds, max_bonds, timesteps_list):
    # Format: { simulator: [ (timesteps, threshold, max_bond, error), ... ] }
    results = {'TEBD': [], 'TDVP': []}
    for max_bond in max_bonds:
        print("Max Bond", max_bond)
        for threshold in thresholds:
            for i, timesteps in enumerate(timesteps_list):
                if i == 0:
                    delta_timesteps = timesteps
                    mps = None
                    mps_qiskit = None
                else:
                    delta_timesteps = timesteps - timesteps_list[i-1]
                circ = create_ising_circuit(num_qubits, J, g, dt, delta_timesteps)
                # Run both simulators
                mps, tdvp_bonds = tdvp_simulator(circ, max_bond, threshold, pad, initial_state=mps)
                mps_qiskit, tebd_bonds = tebd_simulator(circ, max_bond, threshold, initial_state=mps_qiskit)

                # Save the results
                results['TEBD'].append((timesteps, threshold, max_bond, tebd_bonds))
                results['TDVP'].append((timesteps, threshold, max_bond, tdvp_bonds))
    return results

def generate_periodic_ising_observable_data(num_qubits, J, g, dt, pad, thresholds, max_bonds, timesteps_list):
    # Format: { simulator: [ (timesteps, threshold, max_bond, error), ... ] }
    results = {'TEBD': [], 'TDVP': []}
    for max_bond in max_bonds:
        print("Max Bond", max_bond)
        for threshold in thresholds:
            print("Threshold", threshold)
            for i, timesteps in enumerate(timesteps_list):
                print("Timesteps", timesteps)
                if i == 0:
                    delta_timesteps = timesteps
                    mps = None
                    mps_qiskit = None
                else:
                    delta_timesteps = timesteps - timesteps_list[i-1]
                circ = create_ising_circuit(num_qubits, J, g, dt, delta_timesteps, periodic=True)
                # Run both simulators
                mps, tdvp_bonds = tdvp_simulator(circ, max_bond, threshold, pad, initial_state=mps)
                mps_qiskit, tebd_bonds = tebd_simulator(circ, max_bond, threshold, initial_state=mps_qiskit)

                # Save the results
                results['TEBD'].append((timesteps, threshold, max_bond, tebd_bonds))
                results['TDVP'].append((timesteps, threshold, max_bond, tdvp_bonds))
    return results

def generate_heisenberg_observable_data(num_qubits, J, h, dt, pad, thresholds, max_bonds, timesteps_list):
    # Format: { simulator: [ (timesteps, threshold, max_bond, error), ... ] }
    results = {'TEBD': [], 'TDVP': []}
    for max_bond in max_bonds:
        print("Max Bond", max_bond)
        for threshold in thresholds:
            print("Threshold", threshold)
            for i, timesteps in enumerate(timesteps_list):
                print("Timesteps", timesteps)
                if i == 0:
                    delta_timesteps = timesteps
                    mps = None
                    mps_qiskit = None
                else:
                    delta_timesteps = timesteps - timesteps_list[i-1]
                circ = create_heisenberg_circuit(num_qubits, J, J, J, h, dt, delta_timesteps)

                # Run both simulators
                mps, tdvp_bonds = tdvp_simulator(circ, max_bond, threshold, pad, initial_state=mps)
                mps_qiskit, tebd_bonds = tebd_simulator(circ, max_bond, threshold, initial_state=mps_qiskit)

                # Save the results
                results['TEBD'].append((timesteps, threshold, max_bond, tebd_bonds))
                results['TDVP'].append((timesteps, threshold, max_bond, tdvp_bonds))
    return results

def generate_2d_heisenberg_observable_data(num_rows, num_cols, J, h, dt, pad, thresholds, max_bonds, timesteps_list):
    # Format: { simulator: [ (timesteps, threshold, max_bond, error), ... ] }
    results = {'TEBD': [], 'TDVP': []}
    for max_bond in max_bonds:
        print("Max Bond", max_bond)
        for threshold in thresholds:
            for i, timesteps in enumerate(timesteps_list):
                print("Timesteps", timesteps)
                if i == 0:
                    delta_timesteps = timesteps
                    mps = None
                    mps_qiskit = None
                else:
                    delta_timesteps = timesteps - timesteps_list[i-1]
                circ = create_2d_heisenberg_circuit(num_rows, num_cols, J, J, J, h, dt, delta_timesteps)
                # Run both simulators
                mps, tdvp_bonds = tdvp_simulator(circ, max_bond, threshold, pad, initial_state=mps)
                mps_qiskit, tebd_bonds = tebd_simulator(circ, max_bond, threshold, initial_state=mps_qiskit)

                # Save the results
                results['TEBD'].append((timesteps, threshold, max_bond, tebd_bonds))
                results['TDVP'].append((timesteps, threshold, max_bond, tdvp_bonds))
    return results


def generate_2d_ising_observable_data(num_rows, num_cols, J, g, dt, pad, thresholds, max_bonds, timesteps_list):
    # Format: { simulator: [ (timesteps, threshold, max_bond, error), ... ] }
    results = {'TEBD': [], 'TDVP': []}
    for max_bond in max_bonds:
        print("Max Bond", max_bond)
        for threshold in thresholds:
            for i, timesteps in enumerate(timesteps_list):
                print("Timesteps", timesteps)
                if i == 0:
                    delta_timesteps = timesteps
                    mps = None
                    mps_qiskit = None
                else:
                    delta_timesteps = timesteps - timesteps_list[i-1]
                circ = create_2d_ising_circuit(num_rows, num_cols, J, g, dt, delta_timesteps)
                # Run both simulators
                mps, tdvp_bonds = tdvp_simulator(circ, max_bond, threshold, pad, initial_state=mps)
                mps_qiskit, tebd_bonds = tebd_simulator(circ, max_bond, threshold, initial_state=mps_qiskit)

                # Save the results
                results['TEBD'].append((timesteps, threshold, max_bond, tebd_bonds))
                results['TDVP'].append((timesteps, threshold, max_bond, tdvp_bonds))
    return results


# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import LogNorm, TwoSlopeNorm

# def plot_bond_heatmaps(results,
#                        bond_dim,
#                        threshold,
#                        method1='TEBD',
#                        method2='TDVP',
#                        cmap='viridis',
#                        diff_cmap='bwr',
#                        log_scale=False,
#                        figsize=(15,5)):
#     """
#     Plot three heatmaps of bond dimensions over circuit layers:
#       1) method1
#       2) method2
#       3) difference (method1 – method2) with a diverging colormap

#     Parameters
#     ----------
#     results : dict
#         Dict with two keys (e.g. 'TEBD', 'TDVP'), each mapping to a list of
#         tuples (depth, thr, bd, infid, err, bonds_list).
#     bond_dim : int
#         The maximum bond‐dim (bd) used in the run; filters results to only
#         entries with this bd.
#     threshold : float
#         The threshold (thr) used in the run; filters results to only
#         entries with this thr.
#     method1, method2 : str
#         Keys in `results` to compare.
#     cmap : str
#         Matplotlib colormap name for the first two plots.
#     diff_cmap : str
#         Diverging colormap name for the difference plot (default 'bwr').
#     log_scale : bool
#         If True, apply log‐normalization to the first two plots.
#     figsize : tuple
#         Figure size in inches (width, height).

#     Usage
#     -----
#     plot_bond_heatmaps(results,
#                        bond_dim=16,
#                        threshold=1e-6,
#                        method1='TEBD',
#                        method2='TDVP',
#                        cmap='magma',
#                        log_scale=True)
#     """
#     def extract_matrix(method):
#         # filter and sort by depth
#         filtered = [e for e in results[method] if e[1] == threshold and e[2] == bond_dim]
#         filtered.sort(key=lambda x: x[0])
#         return np.vstack([e[3] for e in filtered])

#     mat1 = extract_matrix(method1)
#     mat2 = extract_matrix(method2)
#     assert mat1.shape == mat2.shape, "Shape mismatch between methods"

#     # norms for first two
#     # compute the global vmin/vmax
#     total_min = min(mat1.min(), mat2.min())
#     total_max = max(mat1.max(), mat2.max())

#     if log_scale:
#         # one shared log‐norm for both
#         shared_norm = LogNorm(vmin=total_min, vmax=total_max)
#     else:
#         shared_norm = None

#     # use the same Norm in both heatmaps
#     norm1 = shared_norm
#     norm2 = shared_norm


#     # compute difference
#     diff = mat1 / mat2
#     # max_abs = np.max(np.abs(diff))
#     # diff_norm = TwoSlopeNorm(vcenter=0, vmin=-max_abs, vmax=max_abs)

#     # plot all three
#     fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
#     for ax, mat, title, norm in zip(
#         axes[:2],
#         (mat1, mat2),
#         (method1, method2),
#         (norm1, norm2)
#     ):
#         im = ax.imshow(
#             mat, aspect='auto', origin='lower',
#             interpolation='nearest', cmap=cmap, vmin=total_min, vmax=total_max,
#         )
#         ax.set_title(f"{title}")
#         ax.set_xlabel("Bond index")

#     # difference plot
#     im3 = axes[2].imshow(
#         diff, aspect='auto', origin='lower',
#         interpolation='nearest', cmap='coolwarm', vmin=1, vmax=1.5
#     )
#     axes[2].set_title(f"Compression ratio")
#     axes[2].set_xlabel("Bond index")
#     axes[0].set_ylabel("Trotter Steps")

#     # colorbars
#     cbar1 = fig.colorbar(im, ax=axes[:2].tolist(), shrink=0.8)
#     cbar1.set_label("Bond dimension")
#     cbar2 = fig.colorbar(im3, ax=axes[2], shrink=0.8)
#     cbar2.set_label("Compression ratio ($\\chi_{TEBD}/\\chi_{TDVP})")

#     plt.tight_layout()
#     plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def plot_bond_heatmaps(results,
                       bond_dim,
                       threshold,
                       method1='TEBD',
                       method2='TDVP',
                       cmap='viridis',
                       log_scale=False,
                       figsize=(12,5)):
    """
    Plot two heatmaps of bond dimensions over circuit layers:
      1) method1
      2) method2 with an inset showing total bond dimension over steps.

    Parameters
    ----------
    results : dict
        Dict with two keys (e.g. 'TEBD', 'TDVP'), each mapping to a list of
        tuples (depth, thr, bd, infid, err, bonds_list).
    bond_dim : int
        The maximum bond‐dim (bd) used in the run; filters results to only
        entries with this bd.
    threshold : float
        The threshold (thr) used in the run; filters results to only
        entries with this thr.
    method1, method2 : str
        Keys in `results` to compare.
    cmap : str
        Matplotlib colormap name for the heatmaps.
    log_scale : bool
        If True, apply log‐normalization to the heatmaps.
    figsize : tuple
        Figure size in inches (width, height).
    """
    def extract_matrix(method):
        filtered = [e for e in results[method] if e[1] == threshold and e[2] == bond_dim]
        filtered.sort(key=lambda x: x[0])
        return np.vstack([e[3] for e in filtered])

    mat1 = extract_matrix(method1)
    mat2 = extract_matrix(method2)
    assert mat1.shape == mat2.shape, "Shape mismatch between methods"

    # Determine shared color scaling
    vmin = min(mat1.min(), mat2.min())
    vmax = max(mat1.max(), mat2.max())
    norm = LogNorm(vmin=vmin, vmax=vmax) if log_scale else None

    # Create two panels
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    # Panel 1: method1
    im1 = axes[0].imshow(
        mat1, aspect='auto', origin='lower',
        interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax,
        norm=norm
    )
    axes[0].set_title(method1)
    axes[0].set_xlabel("Bond index")
    axes[0].set_ylabel("Trotter Steps")

    # Panel 2: method2
    im2 = axes[1].imshow(
        mat2, aspect='auto', origin='lower',
        interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax,
        norm=norm
    )
    axes[1].set_title(method2)
    axes[1].set_xlabel("Bond index")

    # Colorbar for panel 2 (next to it)
    cbar = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("Bond dimension")

    # Inset: total bond dimension over Trotter steps
    total1 = mat1.sum(axis=1)
    total2 = mat2.sum(axis=1)
    axins = inset_axes(axes[1], width="40%", height="30%", loc='upper right', borderpad=1)
    axins.plot(np.arange(len(total1)), total1, label=method1)
    axins.plot(np.arange(len(total2)), total2, label=method2)
    axins.set_title("Total χ")
    axins.set_xlabel("Step")
    axins.legend(fontsize='small', loc='lower left')

    plt.tight_layout()
    plt.show()

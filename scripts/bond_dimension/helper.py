import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from qiskit_aer import AerSimulator
from qiskit import transpile

from mqt.yaqs import simulator
from mqt.yaqs.core.libraries.circuit_library import create_ising_circuit, create_2d_ising_circuit, create_heisenberg_circuit, create_2d_heisenberg_circuit
from mqt.yaqs.core.data_structures.networks import MPS
from mqt.yaqs.core.data_structures.simulation_parameters import Observable, StrongSimParams
from mqt.yaqs.core.libraries.gate_library import Z


def tebd_simulator(circ, max_bond, threshold, initial_state=None):
    threshold = 1e-15
    circ2 = copy.deepcopy(circ)
    circ2.clear()
    if initial_state is not None:
        circ2.set_matrix_product_state(initial_state)
    circ2.append(circ, range(circ.num_qubits))
    circ2.save_matrix_product_state(label='final_mps')

    sim = AerSimulator(
        method='matrix_product_state',
        matrix_product_state_max_bond_dimension=max_bond,
        matrix_product_state_truncation_threshold=threshold
    )
    tcirc = transpile(circ2, sim)

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


def plot_bond_heatmaps(results,
                       bond_dim,
                       threshold,
                       method1='TEBD',
                       method2='TDVP',
                       cmap='plasma_r',
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
    total1 = np.sum(mat1**3, axis=1)
    total2 = np.sum(mat2**3, axis=1)
    axins = inset_axes(axes[1], width="30%", height="30%", loc='lower right', borderpad=4)
    print(total1/total2)
    axins.plot(np.arange(len(total1)), total1 / total2)
    # axins.plot(np.arange(len(total2)), total2, label=method2)
    axins.set_ylabel("Runtime Ratio")
    axins.set_xlabel("Step")
    # axins.set_ylim(0.8, 4)
    # axins.legend(fontsize='small', loc='upper left')

    plt.tight_layout()
    plt.show()

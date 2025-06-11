# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from __future__ import annotations

import copy
import operator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from qiskit import transpile
from qiskit_aer import AerSimulator

from mqt.yaqs import simulator
from mqt.yaqs.core.data_structures.networks import MPS
from mqt.yaqs.core.data_structures.simulation_parameters import Observable, StrongSimParams
from mqt.yaqs.core.libraries.circuit_library import (
    create_2d_heisenberg_circuit,
    create_2d_ising_circuit,
    create_heisenberg_circuit,
    create_ising_circuit,
)
from mqt.yaqs.core.libraries.gate_library import XX
from qiskit.quantum_info import SparsePauliOp, Statevector

def _mid_xx_operator(num_qubits):
    """Helper to build a SparsePauliOp for Z on the middle qubit."""
    mid = num_qubits // 2
    label = ["I"] * num_qubits
    label[mid] = "X"
    label[mid+1] = "X"
    return SparsePauliOp("".join(label))

def tebd_simulator(circ, initial_state=None):
    threshold = 1e-13
    circ2 = copy.deepcopy(circ)
    circ2.clear()
    if initial_state is not None:
        circ2.set_matrix_product_state(initial_state)
    circ2.append(circ, range(circ.num_qubits))
    circ2.save_matrix_product_state(label="final_mps")  
    op_xx = _mid_xx_operator(circ.num_qubits)
    mid = circ.num_qubits // 2
    circ2.save_expectation_value(op_xx, [*range(circ.num_qubits)], label="exp_xx")

    sim = AerSimulator(
        method="matrix_product_state",
        # matrix_product_state_max_bond_dimension=max_bond,
        matrix_product_state_truncation_threshold=threshold,
    )
    tcirc = transpile(circ2, sim)

    result = sim.run(tcirc).result()

    result = sim.run(tcirc).result()
    mps = result.data(0)["final_mps"]
    bonds = [lam[0].shape[0] for lam in mps[0][1::]]
    exp_xx = result.data(0)["exp_xx"]
    return mps, bonds, exp_xx


def tdvp_simulator(circ, min_bond, initial_state=None):
    if initial_state is None:
        initial_state = MPS(length=circ.num_qubits)

    measurements = [Observable(XX(), [circ.num_qubits // 2, circ.num_qubits//2+1])]
    sim_params = StrongSimParams(measurements, max_bond_dim=2**circ.num_qubits, min_bond_dim=min_bond, get_state=True)

    # circ_flipped = copy.deepcopy(circ).reverse_bits()
    simulator.run(initial_state, circ, sim_params, noise_model=None)
    mps = sim_params.output_state

    bonds = [tensor.shape[1] for tensor in mps.tensors[1::]]
    exp_val = sim_params.observables[0].results[0]
    return mps, bonds, exp_val


# def generate_ising_observable_data(num_qubits, J, g, dt, pad, thresholds, max_bonds, timesteps_list, min_bond_dim):
#     # Format: { simulator: [ (timesteps, threshold, max_bond, error), ... ] }
#     results = {"TEBD": [], "TDVP": []}
#     for max_bond in max_bonds:
#         for threshold in thresholds:
#             for i, timesteps in enumerate(timesteps_list):
#                 if i == 0:
#                     delta_timesteps = timesteps
#                     mps = None
#                     mps_qiskit = None
#                 else:
#                     delta_timesteps = timesteps - timesteps_list[i - 1]
#                 circ = create_ising_circuit(num_qubits, J, g, dt, delta_timesteps)
#                 # Run both simulators
#                 mps, tdvp_bonds = tdvp_simulator(circ, max_bond, threshold, min_bond_dim, initial_state=mps)
#                 mps_qiskit, tebd_bonds = tebd_simulator(circ, max_bond, threshold, initial_state=mps_qiskit)

#                 # Save the results
#                 results["TEBD"].append((timesteps, threshold, max_bond, tebd_bonds))
#                 results["TDVP"].append((timesteps, threshold, max_bond, tdvp_bonds))
#     return results


# def generate_periodic_ising_observable_data(num_qubits, J, g, dt, pad, thresholds, max_bonds, timesteps_list, min_bond_dim):
#     # Format: { simulator: [ (timesteps, threshold, max_bond, error), ... ] }
#     results = {"TEBD": [], "TDVP": []}
#     for max_bond in max_bonds:
#         for threshold in thresholds:
#             for i, timesteps in enumerate(timesteps_list):
#                 print("Timesteps", timesteps)
#                 if i == 0:
#                     delta_timesteps = timesteps
#                     mps = None
#                     mps_qiskit = None
#                 else:
#                     delta_timesteps = timesteps - timesteps_list[i - 1]
#                 circ = create_ising_circuit(num_qubits, J, g, dt, delta_timesteps, periodic=True)
#                 # Run both simulators
#                 mps, tdvp_bonds = tdvp_simulator(circ, max_bond, threshold, min_bond_dim, initial_state=mps)
#                 mps_qiskit, tebd_bonds = tebd_simulator(circ, max_bond, threshold, initial_state=mps_qiskit)

#                 # Save the results
#                 results["TEBD"].append((timesteps, threshold, max_bond, tebd_bonds))
#                 results["TDVP"].append((timesteps, threshold, max_bond, tdvp_bonds))
#     return results


# def generate_heisenberg_observable_data(num_qubits, J, h, dt, pad, thresholds, max_bonds, timesteps_list, min_bond_dim):
#     # Format: { simulator: [ (timesteps, threshold, max_bond, error), ... ] }
#     results = {"TEBD": [], "TDVP": []}
#     calculate_tebd = True
#     calculate_tdvp = True
#     for max_bond in max_bonds:
#         for threshold in thresholds:
#             for i, timesteps in enumerate(timesteps_list):
#                 print("Timesteps", timesteps)
#                 if i == 0:
#                     delta_timesteps = timesteps
#                     mps = None
#                     mps_qiskit = None
#                 else:
#                     delta_timesteps = timesteps - timesteps_list[i - 1]
#                 circ = create_heisenberg_circuit(num_qubits, J, J, J, h, dt, delta_timesteps)

#                 # Run both simulators
#                 if calculate_tdvp:
#                     mps, tdvp_bonds, tdvp_exp_val = tdvp_simulator(circ, max_bond, threshold, min_bond_dim, initial_state=mps)
#                 else:
#                     tdvp_bonds = [np.nan for _ in range(num_qubits-1)]
#                     tdvp_exp_val = np.nan
#                 if calculate_tebd:
#                     mps_qiskit, tebd_bonds, tebd_exp_val = tebd_simulator(circ, max_bond, threshold, initial_state=mps_qiskit)
#                 else:
#                     tebd_bonds = [np.nan for _ in range(num_qubits-1)]
#                     tebd_exp_val = np.nan 
#                 # Save the results
#                 results["TEBD"].append((timesteps, threshold, max_bond, tebd_bonds, tebd_exp_val))
#                 results["TDVP"].append((timesteps, threshold, max_bond, tdvp_bonds, tdvp_exp_val))
#                 if not calculate_tebd or max(tebd_bonds) > 256:
#                     calculate_tebd = False
#                 if not calculate_tdvp or max(tdvp_bonds) > 256:
#                     calculate_tdvp = False
#                 print("Difference", np.array(tebd_bonds) - np.array(tdvp_bonds))

#     return results

# def generate_periodic_heisenberg_observable_data(num_qubits, J, h, dt, pad, thresholds, max_bonds, timesteps_list, min_bond_dim):
#     # Format: { simulator: [ (timesteps, threshold, max_bond, error), ... ] }
#     results = {"TEBD": [], "TDVP": []}
#     calculate_tebd = True
#     calculate_tdvp = True
#     for max_bond in max_bonds:
#         for threshold in thresholds:
#             for i, timesteps in enumerate(timesteps_list):
#                 print("Timesteps", timesteps)
#                 if i == 0:
#                     delta_timesteps = timesteps
#                     mps = None
#                     mps_qiskit = None
#                 else:
#                     delta_timesteps = timesteps - timesteps_list[i - 1]
#                 circ = create_heisenberg_circuit(num_qubits, J, J, J, h, dt, delta_timesteps, periodic=True)

#                 # Run both simulators
#                 if calculate_tdvp:
#                     mps, tdvp_bonds, tdvp_exp_val = tdvp_simulator(circ, max_bond, threshold, min_bond_dim, initial_state=mps)
#                 else:
#                     tdvp_bonds = [np.nan for _ in range(num_qubits-1)]
#                     tdvp_exp_val = np.nan
#                 if calculate_tebd:
#                     mps_qiskit, tebd_bonds, tebd_exp_val = tebd_simulator(circ, max_bond, threshold, initial_state=mps_qiskit)
#                 else:
#                     tebd_bonds = [np.nan for _ in range(num_qubits-1)]
#                     tebd_exp_val = np.nan 
#                 # Save the results
#                 results["TEBD"].append((timesteps, threshold, max_bond, tebd_bonds, tebd_exp_val))
#                 results["TDVP"].append((timesteps, threshold, max_bond, tdvp_bonds, tdvp_exp_val))
#                 if not calculate_tebd or max(tebd_bonds) > 256:
#                     calculate_tebd = False
#                 if not calculate_tdvp or max(tdvp_bonds) > 256:
#                     calculate_tdvp = False
#                 print("Difference", np.array(tebd_bonds) - np.array(tdvp_bonds))
#     return results


# def generate_2d_heisenberg_observable_data(num_rows, num_cols, J, h, dt, pad, thresholds, max_bonds, timesteps_list, min_bond_dim):
#     # Format: { simulator: [ (timesteps, threshold, max_bond, error), ... ] }
#     results = {"TEBD": [], "TDVP": []}
#     for max_bond in max_bonds:
#         for threshold in thresholds:
#             for i, timesteps in enumerate(timesteps_list):
#                 if i == 0:
#                     delta_timesteps = timesteps
#                     mps = None
#                     mps_qiskit = None
#                 else:
#                     delta_timesteps = timesteps - timesteps_list[i - 1]
#                 circ = create_2d_heisenberg_circuit(num_rows, num_cols, J, J, J, h, dt, delta_timesteps)
#                 # Run both simulators
#                 mps, tdvp_bonds, tdvp_exp_val = tdvp_simulator(circ, max_bond, threshold, min_bond_dim, initial_state=mps)
#                 mps_qiskit, tebd_bonds, tebd_exp_val = tebd_simulator(circ, max_bond, threshold, initial_state=mps_qiskit)

#                 # Save the results
#                 results["TEBD"].append((timesteps, threshold, max_bond, tebd_bonds, tebd_exp_val))
#                 results["TDVP"].append((timesteps, threshold, max_bond, tdvp_bonds, tdvp_exp_val))
#     return results


# def generate_2d_ising_observable_data(num_rows, num_cols, J, g, dt, pad, thresholds, max_bonds, timesteps_list, min_bond_dim):
#     # Format: { simulator: [ (timesteps, threshold, max_bond, error), ... ] }
#     results = {"TEBD": [], "TDVP": []}
#     calculate_tebd = True
#     calculate_tdvp = True
#     for max_bond in max_bonds:
#         for threshold in thresholds:
#             for i, timesteps in enumerate(timesteps_list):
#                 print("Timesteps", timesteps)
#                 if i == 0:
#                     delta_timesteps = timesteps
#                     mps = None
#                     mps_qiskit = None
#                 else:
#                     delta_timesteps = timesteps - timesteps_list[i - 1]
#                 circ = create_2d_ising_circuit(num_rows, num_cols, J, g, dt, delta_timesteps)
#                 # Run both simulators
#                 if calculate_tdvp:
#                     mps, tdvp_bonds, tdvp_exp_val = tdvp_simulator(circ, max_bond, threshold, min_bond_dim, initial_state=mps)
#                 else:
#                     tdvp_bonds = [np.nan for _ in range(num_rows*num_cols-1)]
#                     tdvp_exp_val = np.nan
#                 if calculate_tebd:
#                     mps_qiskit, tebd_bonds, tebd_exp_val = tebd_simulator(circ, max_bond, threshold, initial_state=mps_qiskit)
#                 else:
#                     tebd_bonds = [np.nan for _ in range(num_rows*num_cols-1)]
#                     tebd_exp_val = np.nan 
#                 # Save the results
#                 results["TEBD"].append((timesteps, threshold, max_bond, tebd_bonds, tebd_exp_val))
#                 results["TDVP"].append((timesteps, threshold, max_bond, tdvp_bonds, tdvp_exp_val))
#                 if not calculate_tebd or max(tebd_bonds) > 256:
#                     calculate_tebd = False
#                 if not calculate_tdvp or max(tdvp_bonds) > 256:
#                     calculate_tdvp = False
#                 print("Difference", np.array(tebd_bonds) - np.array(tdvp_bonds))
#     return results

def generate_sim_data(
    make_circ, make_args,
    *,
    timesteps,
    min_bond_dim,
    periodic=False,
    break_on_exceed=False,
    bond_dim_limit=None
):
    """
    Generic driver computing TEBD/TDVP bond dims & expectation values
    for a fixed max_bond and threshold.

    make_circ(*make_args, nsteps, periodic=...) -> QuantumCircuit

    Returns:
      { "TEBD": [...], "TDVP": [...] }
    Each entry is a tuple:
      (timesteps, threshold, max_bond, bonds, exp_val)
    """
    results = {"TEBD": [], "TDVP": []}
    mps_tebd = None
    mps_tdvp = None

    calculate_tebd = calculate_tdvp = True

    for i, ts in enumerate(timesteps):
        # incremental step count
        delta_ts = ts if i == 0 else ts - timesteps[i-1]
        circ_step = make_circ(*make_args, delta_ts, periodic=periodic)

        # TDVP
        if calculate_tdvp:
            mps_tdvp, bonds_tdvp, exp_tdvp = tdvp_simulator(
                circ_step,
                min_bond=min_bond_dim,
                initial_state=mps_tdvp
            )
        else:
            length = len(bonds_tdvp) if 'bonds_tdvp' in locals() else 0
            bonds_tdvp = [np.nan] * length
            exp_tdvp = None

        # TEBD
        if calculate_tebd:
            mps_tebd, bonds_tebd, exp_tebd = tebd_simulator(
                circ_step,
                initial_state=mps_tebd
            )
        else:
            length = len(bonds_tebd) if 'bonds_tebd' in locals() else 0
            bonds_tebd = [np.nan] * length
            exp_tebd = None

        # record results
        results["TDVP"].append((ts, bonds_tdvp, exp_tdvp))
        results["TEBD"].append((ts, bonds_tebd, exp_tebd))

        # optional early stop if bond dims exceed limit
        if break_on_exceed and bond_dim_limit is not None:
            if calculate_tdvp and max(bonds_tdvp, default=0) > bond_dim_limit:
                calculate_tdvp = False
            if calculate_tebd and max(bonds_tebd, default=0) > bond_dim_limit:
                calculate_tebd = False

    return results

def plot_bond_heatmaps(
    results,
    method1="TEBD",
    method2="TDVP",
    cmap="viridis",
    log_scale=False,
    figsize=(7.0, 2.5)
) -> None:
    """
    Create a Nature Physics style 1×3 figure row with:
      (a) Observable expectation vs. Trotter steps for both TEBD and TDVP
      (b) Combined heatmap (top = TEBD, bottom = TDVP) with annotations
      (c) Memory/compression plot (placeholder)
    """
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'font.size': 8,
        'axes.titlesize': 9,
        'axes.labelsize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'lines.linewidth': 1.5,
        'lines.markersize': 4,
        'axes.titlepad': 2,
        'axes.labelpad': 2,
        'xtick.major.pad': 1,
        'ytick.major.pad': 1
    })

    def extract_obs(method):
        data = [(t, exp_val) for (t, bonds, exp_val) in results[method]]
        data.sort(key=lambda x: x[0])
        return (np.array(x) for x in zip(*data)) if data else (np.array([]), np.array([]))

    ts_tebd, obs_tebd = extract_obs(method1)
    ts_tdvp, obs_tdvp = extract_obs(method2)

    def extract_matrix(method):
        filtered = [e for e in results[method]]
        filtered.sort(key=operator.itemgetter(0))
        return np.vstack([e[1] for e in filtered]) if filtered else np.empty((0, 0))

    mat1 = extract_matrix(method1)
    mat2 = extract_matrix(method2)

    # Shared heatmap scale
    if mat1.size and mat2.size:
        vmin = np.nanmin([np.nanmin(mat1), np.nanmin(mat2)])
        vmax = np.nanmax([np.nanmax(mat1), np.nanmax(mat2)])
        norm = LogNorm(vmin=vmin, vmax=vmax) if log_scale else None
    else:
        norm = None
        vmin = vmax = None

    # ----- Figure setup -----
    fig, axes = plt.subplots(1, 3, figsize=figsize, gridspec_kw={'width_ratios': [1, 1, 1], 'wspace': 0.4})
    labels = ['(a)', '(b)', '(c)']

    # Panel (a): expectation vs steps
    ax = axes[0]
    if ts_tebd.size:
        ax.plot(ts_tebd, obs_tebd, marker='^', linestyle='-', label=method1)
    if ts_tdvp.size:
        ax.plot(ts_tdvp, obs_tdvp, marker='o', linestyle='--', label=method2)
    ax.set_xlabel('Trotter steps')
    ax.set_ylabel('Expectation value')
    if log_scale:
        ax.set_yscale('log')
    ax.legend(frameon=False, loc='upper left')
    ax.text(-0.15, 1.05, labels[0], transform=ax.transAxes,
            fontsize=9, fontweight='bold')

    # Panel (b): Combined heatmap
    ax = axes[1]
    mat1_time_x = np.array(mat1.T)
    mat2_time_x = np.array(mat2.T)
    combined = np.vstack([mat2_time_x, mat1_time_x]) if mat1.size and mat2.size else np.empty((0, 0))
    im = ax.imshow(combined, aspect='auto', origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, norm=norm)
    ax.axhline(y=mat1_time_x.shape[0] - 0.5, color='white', linewidth=1)
    ax.set_ylabel('Bond index')
    ax.set_xlabel('Trotter steps')
    ax.text(0.0125, 0.95, method1, transform=ax.transAxes,
            fontsize=8, verticalalignment='top', fontweight='bold', color='white')
    ax.text(0.0125, 0.4, method2, transform=ax.transAxes,
            fontsize=8, verticalalignment='bottom', fontweight='bold', color='white')

    # Shared colorbar
    cbar_ax = fig.add_axes([0.615, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Bond dim')

    # Panel (c): Memory/Compression placeholder
    ax = axes[2]
    tebd_total_bond = []
    tdvp_total_bond = []
    for i in range(mat1.shape[0]):
        tebd_total_bond.append(np.sum(mat1[i, :]))
        tdvp_total_bond.append(np.sum(mat2[i, :]))
    ax.plot(ts_tebd, tebd_total_bond)
    ax.plot(ts_tdvp, tdvp_total_bond)
    ax.set_xlabel('Trotter steps')
    ax.set_ylabel('Ratio')
    ax.text(-0.15, 1.05, labels[2], transform=ax.transAxes,
            fontsize=9, fontweight='bold')
    ax.text(0.01, 0.95, 'Memory', transform=ax.transAxes,
            fontsize=8, verticalalignment='top', fontweight='bold')
    ax.set_xticks([])
    # ax.set_yticks([])

    plt.tight_layout(rect=[0, 0, 0.6, 1])
    plt.show()

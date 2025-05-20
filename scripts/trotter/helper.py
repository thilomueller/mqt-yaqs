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
from qiskit import transpile
from qiskit.quantum_info import SparsePauliOp, Statevector
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


def _mid_z_operator(num_qubits):
    """Helper to build a SparsePauliOp for Z on the middle qubit."""
    mid = num_qubits // 2
    label = ["I"] * num_qubits
    label[mid] = "X"
    label[mid+1] = "X"
    return SparsePauliOp("".join(label))


def state_vector_simulator(circ):
    # Run statevector simulator
    sim = AerSimulator(method="statevector")
    tcirc = transpile(circ, sim)
    result = sim.run(tcirc).result()
    sv = Statevector(result.get_statevector(tcirc))

    # build Z_mid and compute expectation
    z_mid = _mid_z_operator(tcirc.num_qubits)
    exp_val = sv.expectation_value(z_mid).real

    return sv, exp_val


def tebd_simulator(circ, max_bond, threshold, initial_state):
    threshold = 1e-13
    # Save MPS snapshot into the circuit
    circ2 = copy.deepcopy(circ)
    circ2.clear()
    if initial_state is not None:
        circ2.set_matrix_product_state(initial_state)
    circ2.append(circ, range(circ.num_qubits))
    circ2.save_matrix_product_state(label="final_mps")
    sim = AerSimulator(
        method="matrix_product_state",
        matrix_product_state_max_bond_dimension=max_bond,
        matrix_product_state_truncation_threshold=threshold,
    )
    tcirc = transpile(circ2, sim)

    result = sim.run(tcirc).result()

    state_vector = result.get_statevector(tcirc)
    mps = result.data(0)["final_mps"]
    [lam[0].shape[0] for lam in mps[0][1::]]

    sv = Statevector(state_vector)
    z_mid = _mid_z_operator(circ.num_qubits)
    exp_val = sv.expectation_value(z_mid).real
    return mps, sv, exp_val


def tdvp_simulator(circ, max_bond, threshold, min_bond, initial_state=None):
    if initial_state is None:
        initial_state = MPS(length=circ.num_qubits)

    measurements = [Observable(XX(), [circ.num_qubits // 2, circ.num_qubits // 2 + 1])]
    sim_params = StrongSimParams(measurements, max_bond_dim=max_bond, min_bond_dim=min_bond, get_state=True)

    circ_flipped = copy.deepcopy(circ).reverse_bits()
    simulator.run(initial_state, circ_flipped, sim_params, noise_model=None)
    mps = sim_params.output_state

    sv = mps.to_vec()
    exp_val = sim_params.observables[0].results[0]
    return mps, sv, exp_val


def generate_ising_observable_data(num_qubits, J, g, dt, pad, thresholds, max_bonds, timesteps_list, min_bond_dim):
    # Format: { simulator: [ (timesteps, threshold, max_bond, error), ... ] }
    results = {"TEBD": [], "TDVP": []}
    for max_bond in max_bonds:
        for threshold in thresholds:
            for i, timesteps in enumerate(timesteps_list):
                if i == 0:
                    delta_timesteps = timesteps
                    mps = None
                    mps_qiskit = None
                else:
                    delta_timesteps = timesteps - timesteps_list[i - 1]
                circ = create_ising_circuit(num_qubits, J, g, dt, delta_timesteps)
                circ2 = copy.deepcopy(circ)
                circ2.save_statevector()
                circ_sv = create_ising_circuit(num_qubits, J, g, dt, timesteps)
                circ_sv.save_statevector()
                exact_sv, exact_exp_val = state_vector_simulator(circ_sv)

                # Run both simulators
                mps, tdvp_sv, tdvp_exp_val = tdvp_simulator(circ, max_bond, threshold, min_bond_dim, initial_state=mps)
                mps_qiskit, tebd_sv, tebd_exp_val = tebd_simulator(circ2, max_bond, threshold, initial_state=mps_qiskit)

                # Compute the absolute error relative to the exact solution
                # Second absolute is to stop numerical errors in squaring small floats
                tebd_infidelity = np.abs(1 - np.abs(np.vdot(exact_sv, tebd_sv)) ** 2)
                tdvp_infidelity = np.abs(1 - np.abs(np.vdot(exact_sv, tdvp_sv)) ** 2)
                tebd_error = np.abs(exact_exp_val - tebd_exp_val)
                tdvp_error = np.abs(exact_exp_val - tdvp_exp_val)

                # Save the results
                results["TEBD"].append((timesteps, threshold, max_bond, tebd_infidelity, tebd_error))
                results["TDVP"].append((timesteps, threshold, max_bond, tdvp_infidelity, tdvp_error))
    return results


def generate_periodic_ising_observable_data(num_qubits, J, g, dt, pad, thresholds, max_bonds, timesteps_list, min_bond_dim):
    # Format: { simulator: [ (timesteps, threshold, max_bond, error), ... ] }
    results = {"TEBD": [], "TDVP": []}
    for max_bond in max_bonds:
        for threshold in thresholds:
            for i, timesteps in enumerate(timesteps_list):
                if i == 0:
                    delta_timesteps = timesteps
                    mps = None
                    mps_qiskit = None
                else:
                    delta_timesteps = timesteps - timesteps_list[i - 1]
                circ = create_ising_circuit(num_qubits, J, g, dt, delta_timesteps, periodic=True)
                circ2 = copy.deepcopy(circ)
                circ2.save_statevector()
                circ_sv = create_ising_circuit(num_qubits, J, g, dt, timesteps, periodic=True)
                circ_sv.save_statevector()
                exact_sv, exact_exp_val = state_vector_simulator(circ_sv)

                # Run both simulators
                mps, tdvp_sv, tdvp_exp_val = tdvp_simulator(circ, max_bond, threshold, min_bond_dim, initial_state=mps)
                mps_qiskit, tebd_sv, tebd_exp_val = tebd_simulator(circ2, max_bond, threshold, initial_state=mps_qiskit)

                # Compute the absolute error relative to the exact solution
                # Second absolute is to stop numerical errors in squaring small floats
                tebd_infidelity = np.abs(1 - np.abs(np.vdot(exact_sv, tebd_sv)) ** 2)
                tdvp_infidelity = np.abs(1 - np.abs(np.vdot(exact_sv, tdvp_sv)) ** 2)
                tebd_error = np.abs(exact_exp_val - tebd_exp_val)
                tdvp_error = np.abs(exact_exp_val - tdvp_exp_val)

                # Save the results
                results["TEBD"].append((timesteps, threshold, max_bond, tebd_infidelity, tebd_error))
                results["TDVP"].append((timesteps, threshold, max_bond, tdvp_infidelity, tdvp_error))
    return results


def generate_heisenberg_observable_data(num_qubits, J, h, dt, pad, thresholds, max_bonds, timesteps_list, min_bond_dim):
    # Format: { simulator: [ (timesteps, threshold, max_bond, error), ... ] }
    results = {"TEBD": [], "TDVP": []}
    for max_bond in max_bonds:
        for threshold in thresholds:
            for i, timesteps in enumerate(timesteps_list):
                if i == 0:
                    delta_timesteps = timesteps
                    mps = None
                    mps_qiskit = None
                else:
                    delta_timesteps = timesteps - timesteps_list[i - 1]
                circ = create_heisenberg_circuit(num_qubits, J, J, J, h, dt, delta_timesteps)
                circ2 = copy.deepcopy(circ)
                circ2.save_statevector()
                circ_sv = create_heisenberg_circuit(num_qubits, J, J, J, h, dt, timesteps)
                circ_sv.save_statevector()
                exact_sv, exact_exp_val = state_vector_simulator(circ_sv)

                # Run both simulators
                mps, tdvp_sv, tdvp_exp_val = tdvp_simulator(circ, max_bond, threshold, min_bond_dim, initial_state=mps)
                mps_qiskit, tebd_sv, tebd_exp_val = tebd_simulator(circ2, max_bond, threshold, initial_state=mps_qiskit)

                # Compute the absolute error relative to the exact solution
                # Second absolute is to stop numerical errors in squaring small floats
                tebd_infidelity = np.abs(1 - np.abs(np.vdot(exact_sv, tebd_sv)) ** 2)
                tdvp_infidelity = np.abs(1 - np.abs(np.vdot(exact_sv, tdvp_sv)) ** 2)
                tebd_error = np.abs(exact_exp_val - tebd_exp_val)
                tdvp_error = np.abs(exact_exp_val - tdvp_exp_val)

                # Save the results
                results["TEBD"].append((timesteps, threshold, max_bond, tebd_infidelity, tebd_error))
                results["TDVP"].append((timesteps, threshold, max_bond, tdvp_infidelity, tdvp_error))
    return results


def generate_periodic_heisenberg_observable_data(num_qubits, J, h, dt, pad, thresholds, max_bonds, timesteps_list, min_bond_dim):
    # Format: { simulator: [ (timesteps, threshold, max_bond, error), ... ] }
    results = {"TEBD": [], "TDVP": []}
    for max_bond in max_bonds:
        for threshold in thresholds:
            for i, timesteps in enumerate(timesteps_list):
                if i == 0:
                    delta_timesteps = timesteps
                    mps = None
                    mps_qiskit = None
                else:
                    delta_timesteps = timesteps - timesteps_list[i - 1]
                circ = create_heisenberg_circuit(num_qubits, J, J, J, h, dt, delta_timesteps, periodic=True)
                circ2 = copy.deepcopy(circ)
                circ2.save_statevector()
                circ_sv = create_heisenberg_circuit(num_qubits, J, J, J, h, dt, timesteps, periodic=True)
                circ_sv.save_statevector()
                exact_sv, exact_exp_val = state_vector_simulator(circ_sv)

                # Run both simulators
                mps, tdvp_sv, tdvp_exp_val = tdvp_simulator(circ, max_bond, threshold, min_bond_dim, initial_state=mps)
                mps_qiskit, tebd_sv, tebd_exp_val = tebd_simulator(circ2, max_bond, threshold, initial_state=mps_qiskit)

                # Compute the absolute error relative to the exact solution
                # Second absolute is to stop numerical errors in squaring small floats
                tebd_infidelity = np.abs(1 - np.abs(np.vdot(exact_sv, tebd_sv)) ** 2)
                tdvp_infidelity = np.abs(1 - np.abs(np.vdot(exact_sv, tdvp_sv)) ** 2)
                tebd_error = np.abs(exact_exp_val - tebd_exp_val)
                tdvp_error = np.abs(exact_exp_val - tdvp_exp_val)

                # Save the results
                results["TEBD"].append((timesteps, threshold, max_bond, tebd_infidelity, tebd_error))
                results["TDVP"].append((timesteps, threshold, max_bond, tdvp_infidelity, tdvp_error))
    return results

# def generate_2d_heisenberg_observable_data(num_rows, num_cols, J, h, dt, pad, thresholds, max_bonds, timesteps_list, min_bond_dim):
#     # Format: { simulator: [ (timesteps, threshold, max_bond, error), ... ] }
#     results = {"TEBD": [], "TDVP": []}
#     for max_bond in max_bonds:
#         for threshold in thresholds:
#             for i, timesteps in enumerate(timesteps_list):
#                 if i == 0:
#                     delta_timesteps = timesteps
#                     mps = None
#                 else:
#                     delta_timesteps = timesteps - timesteps_list[i - 1]
#                 circ = create_2d_heisenberg_circuit(num_rows, num_cols, J, J, J, h, dt, delta_timesteps)
#                 circ2 = create_2d_heisenberg_circuit(num_rows, num_cols, J, J, J, h, dt, timesteps)
#                 circ2.save_statevector()
#                 exact_result = state_vector_simulator(circ2)

#                 # Run both simulators
#                 mps, tdvp_sv, tdvp_exp_val = tdvp_simulator(circ, max_bond, threshold, min_bond_dim, initial_state=mps)
#                 mps_qiskit, tebd_sv, tebd_exp_val = tebd_simulator(circ2, max_bond, threshold, initial_state=mps_qiskit)

#                 # Compute the absolute error relative to the exact solution
#                 # Second absolute is to stop numerical errors in squaring small floats
#                 tebd_infidelity = np.abs(1 - np.abs(np.vdot(exact_sv, tebd_sv)) ** 2)
#                 tdvp_infidelity = np.abs(1 - np.abs(np.vdot(exact_sv, tdvp_sv)) ** 2)
#                 tebd_error = np.abs(exact_exp_val - tebd_exp_val)
#                 tdvp_error = np.abs(exact_exp_val - tdvp_exp_val)

#                 # Save the results
#                 results["TEBD"].append((timesteps, threshold, max_bond, tebd_infidelity, tebd_error))
#                 results["TDVP"].append((timesteps, threshold, max_bond, tdvp_infidelity, tdvp_error))
#     return results


def generate_2d_ising_observable_data(num_rows, num_cols, J, g, dt, pad, thresholds, max_bonds, timesteps_list, min_bond_dim):
    # Format: { simulator: [ (timesteps, threshold, max_bond, error), ... ] }
    results = {"TEBD": [], "TDVP": []}
    for max_bond in max_bonds:
        for threshold in thresholds:
            for i, timesteps in enumerate(timesteps_list):
                if i == 0:
                    delta_timesteps = timesteps
                    mps = None
                    mps_qiskit = None
                else:
                    delta_timesteps = timesteps - timesteps_list[i - 1]
                circ = create_2d_ising_circuit(num_rows, num_cols, J, g, dt, delta_timesteps)
                circ2 = copy.deepcopy(circ)
                circ2.save_statevector()
                circ_sv = create_2d_ising_circuit(num_rows, num_cols, J, g, dt, timesteps)
                circ_sv.save_statevector()
                exact_sv, exact_exp_val = state_vector_simulator(circ_sv)

                # Run both simulators
                mps, tdvp_sv, tdvp_exp_val = tdvp_simulator(circ, max_bond, threshold, min_bond_dim, initial_state=mps)
                mps_qiskit, tebd_sv, tebd_exp_val = tebd_simulator(circ2, max_bond, threshold, initial_state=mps_qiskit)

                # Compute the absolute error relative to the exact solution
                # Second absolute is to stop numerical errors in squaring small floats
                tebd_infidelity = np.abs(1 - np.abs(np.vdot(exact_sv, tebd_sv)) ** 2)
                tdvp_infidelity = np.abs(1 - np.abs(np.vdot(exact_sv, tdvp_sv)) ** 2)
                tebd_error = np.abs(exact_exp_val - tebd_exp_val)
                tdvp_error = np.abs(exact_exp_val - tdvp_exp_val)

                # Save the results
                results["TEBD"].append((timesteps, threshold, max_bond, tebd_infidelity, tebd_error))
                results["TDVP"].append((timesteps, threshold, max_bond, tdvp_infidelity, tdvp_error))
    return results


def plot_error_vs_depth(results, bond_dims) -> None:
    # Create a 1xN figure (one subplot per threshold)
    _fig, axes = plt.subplots(2, 1, sharex=True)
    # Map bond dimensions to distinct colors
    color_map = {4: "red", 8: "green", 16: "blue", 32: "black", 64: "black", 256: "black"}
    # Use different markers for each simulator
    marker_map = {"TEBD": "o", "TDVP": "^"}

    axes[0].set_title("Infidelity")
    axes[1].set_title("Local Observable Error")
    for i, name in enumerate(["infidelity", "error"]):
        ax = axes[i]

        # Create an inset in the lower right corner
        ax_inset = ax.inset_axes([0.55, 0.05, 0.4, 0.4])
        ax_inset.set_title("TDVP Advantage", fontsize=8)
        ax_inset.tick_params(axis="both", labelsize=8)
        ax_inset.set_ylim(0.5e-1, 1e8)
        for bond_dim in bond_dims:
            # Extract data for TEBD and TDVP
            if name == "infidelity":
                tebd_data = [(depth, infid) for (depth, the, bd, infid, err) in results["TEBD"] if bd == bond_dim]
                tdvp_data = [(depth, infid) for (depth, the, bd, infid, err) in results["TDVP"] if bd == bond_dim]
            elif name == "error":
                tebd_data = [(depth, err) for (depth, the, bd, infid, err) in results["TEBD"] if bd == bond_dim]
                tdvp_data = [(depth, err) for (depth, the, bd, infid, err) in results["TDVP"] if bd == bond_dim]

            # Sort by circuit depth
            tebd_data.sort(key=operator.itemgetter(0))
            tdvp_data.sort(key=operator.itemgetter(0))

            # Unpack for plotting
            tebd_depths = [x[0] for x in tebd_data]
            tebd_errors = [x[1] for x in tebd_data]
            tdvp_depths = [x[0] for x in tdvp_data]
            tdvp_errors = [x[1] for x in tdvp_data]

            # Plot main curves on the primary axis
            ax.plot(
                tebd_depths,
                tebd_errors,
                label=f"TEBD, bond={bond_dim}" if i == 0 else "",
                color=color_map[bond_dim],
                marker=marker_map["TEBD"],
                linestyle="--",
            )
            ax.plot(
                tdvp_depths,
                tdvp_errors,
                label=f"TDVP, bond={bond_dim}" if i == 0 else "",
                color=color_map[bond_dim],
                marker=marker_map["TDVP"],
                linestyle="-",
            )

            ratio_errors = [te / td for td, te in zip(tdvp_errors, tebd_errors)]
            ax_inset.axhline(1, color="k", linestyle="--")  # reference line at ratio = 1
            # Plot the difference in the inset axis
            ax_inset.plot(
                tdvp_depths,
                ratio_errors,
                label=f"bond={bond_dim}",
                color=color_map[bond_dim],
                marker=marker_map["TDVP"],
                linestyle="-",
            )
            ax_inset.set_yscale("log")

        ax.set_yscale("log")
        ax.set_ylim(1e-16, 5e-1)

        ax.grid(True)

        # Add a horizontal line at 0 in the inset to mark no difference
        ax_inset.axhline(0, color="black", linestyle="--", linewidth=1)

    axes[1].set_ylabel("Circuit depth (Trotter steps)")
    axes[0].set_ylabel("Infidelity (log scale)")
    axes[1].set_ylabel("Local observable error (log scale)")

    axes[0].legend(loc="lower left", fontsize="small")
    axes[0].set_xlabel("Circuit depth (Trotter steps)")
    axes[0].set_xlabel(None)
    plt.tight_layout()
    plt.show()

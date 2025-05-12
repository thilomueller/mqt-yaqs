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

def state_vector_simulator(circ):
    simulator = AerSimulator(method='statevector')
    result = simulator.run(circ)
    state_vector = result.result().get_statevector(circ)

    sv = Statevector(state_vector)
    return sv


def tebd_simulator(circ, max_bond, threshold):
    # Set up the simulator with the MPS backend and the desired options.
    circ.save_matrix_product_state(label='mps')

    simulator = AerSimulator(method='matrix_product_state',
                                matrix_product_state_max_bond_dimension=max_bond,
                                matrix_product_state_truncation_threshold=threshold)
                               # mps_log_data=False)
    result = simulator.run(circ).result()
    state_vector = result.get_statevector(circ)
    mps = result.data(0)['mps'][0]

    max_bond = 1
    for tensor in mps:
        max_bond = max(max_bond, max(tensor[0].shape))
    print("TEBD Max Bond", max_bond)

    sv = Statevector(state_vector)
    return sv

# import quimb as qu
# import quimb.tensor as qtn
# import numpy as np

# def tebd_simulator_quimb(num_qubits, J, h, dt, timesteps, max_bond, threshold):
#     """
#     Simulate time evolution under a Heisenberg Hamiltonian using TEBD in quimb.
    
#     Parameters:
#       num_qubits (int): Number of spins in the chain.
#       J (float): Coupling constant (assumed uniform for simplicity).
#       h (float): External field (or anisotropy parameter, as needed).
#       dt (float): Time step for the Trotter evolution.
#       timesteps (int): Number of time steps.
#       max_bond (int): Maximum allowed bond dimension.
#       threshold (float): Truncation error threshold for SVD.
      
#     Returns:
#       state_vector (numpy.ndarray): Final state vector after evolution.
#     """
#     # Create the Heisenberg Hamiltonian.
#     # Here we use quimb's helper for a Heisenberg chain with periodic boundary conditions.
#     H = qtn.ham_1d_heis(L=num_qubits, j=J, bz=h, cyclic=False)

#     # Define the initial state as an MPS. For example, all spins up ('0' for spin-up).
#     psi0 = qtn.MPS_zero_state(num_qubits)
    
#     # Evolve the state using TEBD.
#     # The evolve method applies a Trotterized time evolution using the provided Hamiltonian.
#     # It returns the MPS that approximates the evolved state.
#     tebd = qtn.TEBD(psi0, H)
#     # tebd.split_opts['cutoff'] = threshold
#     # tebd.split_opts['max_bond'] = 64
#     tebd.update_to(T=dt*timesteps, dt=dt, order=2)
#     final_psi = tebd.pt
#     return final_psi

def tdvp_simulator(circ, max_bond, threshold, pad, initial_state=None):
    if initial_state is None:
        initial_state = MPS(length=circ.num_qubits)
        initial_state.pad_bond_dimension(pad)
    measurements = [Observable(Z(), circ.num_qubits//2)]
    sim_params = StrongSimParams(measurements, num_traj=1, max_bond_dim=max_bond, threshold=threshold, get_state=True, window_size=1)
    simulator.run(initial_state, circ, sim_params, noise_model=None)
    mps = sim_params.output_state
    print("TDVP Max Bond", mps.write_max_bond_dim())

    sv = mps.to_vec()
    return mps, sv


def generate_ising_infidelity_data(num_qubits, J, g, dt, pad, thresholds, max_bonds, timesteps_list):
    # Format: { simulator: [ (timesteps, threshold, max_bond, error), ... ] }
    results = {'TEBD': [], 'TDVP': []}
    for max_bond in max_bonds:
        print("Max Bond", max_bond)
        for threshold in thresholds:
            for i, timesteps in enumerate(timesteps_list):
                if i == 0:
                    delta_timesteps = timesteps
                    mps = None
                else:
                    delta_timesteps = timesteps - timesteps_list[i-1]
                circ = create_ising_circuit(num_qubits, J, g, dt, delta_timesteps)
                circ2 = create_ising_circuit(num_qubits, J, g, dt, timesteps)
                circ2.save_statevector()
                exact_result = state_vector_simulator(circ2)

                # Run both simulators
                mps, tdvp_result = tdvp_simulator(circ, max_bond, threshold, pad, initial_state=mps)
                tebd_result = tebd_simulator(circ2, max_bond, threshold)

                # Compute the absolute error relative to the exact solution
                # Second absolute is to stop numerical errors in squaring small floats
                tebd_error = np.abs(1-np.abs(np.vdot(exact_result, tebd_result))**2)
                tdvp_error = np.abs(1-np.abs(np.vdot(exact_result, tdvp_result))**2)
                # Save the results
                results['TEBD'].append((timesteps, threshold, max_bond, tebd_error))
                results['TDVP'].append((timesteps, threshold, max_bond, tdvp_error))
    return results

def generate_periodic_ising_infidelity_data(num_qubits, J, g, dt, pad, thresholds, max_bonds, timesteps_list):
    # Format: { simulator: [ (timesteps, threshold, max_bond, error), ... ] }
    results = {'TEBD': [], 'TDVP': []}
    for max_bond in max_bonds:
        print("Max Bond", max_bond)
        for threshold in thresholds:
            print("Threshold", threshold)
            for i, timesteps in enumerate(timesteps_list):
                if i == 0:
                    delta_timesteps = timesteps
                    mps = None
                else:
                    delta_timesteps = timesteps - timesteps_list[i-1]
                circ = create_ising_circuit(num_qubits, J, g, dt, delta_timesteps, periodic=True)
                circ2 = create_ising_circuit(num_qubits, J, g, dt, timesteps, periodic=True)
                circ2.save_statevector()
                exact_result = state_vector_simulator(circ2)

                # Run both simulators
                mps, tdvp_result = tdvp_simulator(circ, max_bond, threshold, pad, initial_state=mps)
                tebd_result = tebd_simulator(circ2, max_bond, threshold)

                # Compute the absolute error relative to the exact solution
                # Second absolute is to stop numerical errors in squaring small floats
                tebd_error = np.abs(1-np.abs(np.vdot(exact_result, tebd_result))**2)
                tdvp_error = np.abs(1-np.abs(np.vdot(exact_result, tdvp_result))**2)

                # Save the results
                results['TEBD'].append((timesteps, threshold, max_bond, tebd_error))
                results['TDVP'].append((timesteps, threshold, max_bond, tdvp_error))
    return results

def generate_heisenberg_infidelity_data(num_qubits, J, h, dt, pad, thresholds, max_bonds, timesteps_list):
    # Format: { simulator: [ (timesteps, threshold, max_bond, error), ... ] }
    results = {'TEBD': [], 'TDVP': []}
    for max_bond in max_bonds:
        print("Max Bond", max_bond)
        for threshold in thresholds:
            print("Threshold", threshold)
            for i, timesteps in enumerate(timesteps_list):
                if i == 0:
                    delta_timesteps = timesteps
                    mps = None
                else:
                    delta_timesteps = timesteps - timesteps_list[i-1]
                circ = create_heisenberg_circuit(num_qubits, J, J, J, h, dt, delta_timesteps)
                circ2 = create_heisenberg_circuit(num_qubits, J, J, J, h, dt, timesteps)
                circ2.save_statevector()
                exact_result = state_vector_simulator(circ2)

                # Run both simulators
                mps, tdvp_result = tdvp_simulator(circ, max_bond, threshold, pad, initial_state=mps)
                tebd_result = tebd_simulator(circ2, max_bond, threshold)

                # Compute the absolute error relative to the exact solution
                # Second absolute is to stop numerical errors in squaring small floats
                tebd_error = np.abs(1-np.abs(np.vdot(exact_result, tebd_result))**2)
                tdvp_error = np.abs(1-np.abs(np.vdot(exact_result, tdvp_result))**2)
                # Save the results
                results['TEBD'].append((timesteps, threshold, max_bond, tebd_error))
                results['TDVP'].append((timesteps, threshold, max_bond, tdvp_error))
    return results

def generate_2d_heisenberg_infidelity_data(num_rows, num_cols, J, h, dt, pad, thresholds, max_bonds, timesteps_list):
    # Format: { simulator: [ (timesteps, threshold, max_bond, error), ... ] }
    results = {'TEBD': [], 'TDVP': []}
    for max_bond in max_bonds:
        print("Max Bond", max_bond)
        for threshold in thresholds:
            for i, timesteps in enumerate(timesteps_list):
                if i == 0:
                    delta_timesteps = timesteps
                    mps = None
                else:
                    delta_timesteps = timesteps - timesteps_list[i-1]
                circ = create_2d_heisenberg_circuit(num_rows, num_cols, J, J, J, h, dt, delta_timesteps)
                circ2 = create_2d_heisenberg_circuit(num_rows, num_cols, J, J, J, h, dt, timesteps)
                circ2.save_statevector()
                exact_result = state_vector_simulator(circ2)

                # Run both simulators
                mps, tdvp_result = tdvp_simulator(circ, max_bond, threshold, pad, initial_state=mps)
                tebd_result = tebd_simulator(circ2, max_bond, threshold)

                # Compute the absolute error relative to the exact solution
                # Second absolute is to stop numerical errors in squaring small floats
                tebd_error = np.abs(1-np.abs(np.vdot(exact_result, tebd_result))**2)
                tdvp_error = np.abs(1-np.abs(np.vdot(exact_result, tdvp_result))**2)
                # Save the results
                results['TEBD'].append((timesteps, threshold, max_bond, tebd_error))
                results['TDVP'].append((timesteps, threshold, max_bond, tdvp_error))
    return results


def generate_2d_ising_infidelity_data(num_rows, num_cols, J, g, dt, pad, thresholds, max_bonds, timesteps_list):
    # Format: { simulator: [ (timesteps, threshold, max_bond, error), ... ] }
    results = {'TEBD': [], 'TDVP': []}
    for max_bond in max_bonds:
        print("Max Bond", max_bond)
        for threshold in thresholds:
            for i, timesteps in enumerate(timesteps_list):
                if i == 0:
                    delta_timesteps = timesteps
                    mps = None
                else:
                    delta_timesteps = timesteps - timesteps_list[i-1]
                circ = create_2d_ising_circuit(num_rows, num_cols, J, g, dt, delta_timesteps)
                circ2 = create_2d_ising_circuit(num_rows, num_cols, J, g, dt, timesteps)
                circ2.save_statevector()
                exact_result = state_vector_simulator(circ2)

                # Run both simulators
                mps, tdvp_result = tdvp_simulator(circ, max_bond, threshold, pad, initial_state=mps)
                tebd_result = tebd_simulator(circ2, max_bond, threshold)

                # Compute the absolute error relative to the exact solution
                # Second absolute is to stop numerical errors in squaring small floats
                tebd_error = np.abs(1-np.abs(np.vdot(exact_result, tebd_result))**2)
                tdvp_error = np.abs(1-np.abs(np.vdot(exact_result, tdvp_result))**2)
                # Save the results
                results['TEBD'].append((timesteps, threshold, max_bond, tebd_error))
                results['TDVP'].append((timesteps, threshold, max_bond, tdvp_error))
    return results


def plot_error_vs_depth(results, thresholds, bond_dims):
    # Create a 1xN figure (one subplot per threshold)
    fig, axes = plt.subplots(1, len(thresholds), figsize=(15, 5), sharey=True)
    # Map bond dimensions to distinct colors
    color_map = {4: 'red', 8: 'green', 16: 'blue', 32: 'black', 64: 'black'}
    # Use different markers for each simulator
    marker_map = {'TEBD': 'o', 'TDVP': '^'}

    for i, threshold in enumerate(thresholds):
        if len(thresholds) == 1:
            ax = axes
        else:
            ax = axes[i]
        ax.set_title(f"Threshold = {threshold}")

        # Create an inset in the lower right corner
        ax_inset = ax.inset_axes([0.55, 0.05, 0.4, 0.4])
        ax_inset.set_title("TDVP Advantage", fontsize=8)
        ax_inset.tick_params(axis='both', labelsize=8)
        ax_inset.set_ylim(0.5e-1, 1e8)
        for bond_dim in bond_dims:
            # Extract data for TEBD and TDVP
            tebd_data = [
                (depth, err)
                for (depth, thr, bd, err) in results['TEBD']
                if thr == threshold and bd == bond_dim
            ]
            tdvp_data = [
                (depth, err)
                for (depth, thr, bd, err) in results['TDVP']
                if thr == threshold and bd == bond_dim
            ]

            # Sort by circuit depth
            tebd_data.sort(key=lambda x: x[0])
            tdvp_data.sort(key=lambda x: x[0])

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
                marker=marker_map['TEBD'],
                linestyle='--'
            )
            ax.plot(
                tdvp_depths,
                tdvp_errors,
                label=f"TDVP, bond={bond_dim}" if i == 0 else "",
                color=color_map[bond_dim],
                marker=marker_map['TDVP'],
                linestyle='-'
            )

            ratio_errors = [te / td for td, te in zip(tdvp_errors, tebd_errors)]
            ax_inset.axhline(1, color='k', linestyle='--')  # reference line at ratio = 1
            # Plot the difference in the inset axis
            ax_inset.plot(
                tdvp_depths,
                ratio_errors,
                label=f"bond={bond_dim}",
                color=color_map[bond_dim],
                marker=marker_map['TDVP'],
                linestyle='-'
            )
            ax_inset.set_yscale('log')


        ax.set_xlabel("Circuit depth (Trotter steps)")
        ax.set_yscale("log")
        ax.set_ylim(1e-16, 5e-1)

        if i == 0:
            ax.set_ylabel("Fidelity error (log scale)")
        ax.grid(True)

        # Add a horizontal line at 0 in the inset to mark no difference
        ax_inset.axhline(0, color='black', linestyle='--', linewidth=1)

    # Place a combined legend in the left-most subplot
    if len(thresholds) == 1:
        ax.legend(loc="lower left", fontsize="small")
    else:
        axes[0].legend(loc="lower left", fontsize="small")
    plt.suptitle("Simulator Error vs. Circuit Depth\n(Varying SVD Threshold and Bond Dimension)")
    plt.tight_layout()
    plt.show()

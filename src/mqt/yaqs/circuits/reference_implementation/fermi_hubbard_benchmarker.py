# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Benchmarker for the Fermi-Hubbard circuits.

This module provides functions for benchmarking the speed and accuracy
of the TDVP simulation of the Fermi-Hubbard circuits compared to the
reference implementations.
"""

# ignore non-lowercase argument names for physics notation
# ruff: noqa: N803

from __future__ import annotations

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from mqt.yaqs.core.data_structures.networks import MPS
from mqt.yaqs.core.data_structures.simulation_parameters import Observable, StrongSimParams
from mqt.yaqs.core.libraries.circuit_library import create_2d_fermi_hubbard_circuit
from mqt.yaqs.circuits.reference_implementation.fermi_hubbard_reference import create_fermi_hubbard_model_qutip, site_index

from mqt.yaqs import simulator


def benchmarker() -> None:
    # Define the model
    t = 1.0         # kinetic hopping
    mu = 0.5        # chemical potential
    u = 4.0         # onsite interaction
    Lx, Ly = 2, 2   # lattice dimensions
    L = Lx * Ly     # total lattice sites

    num_sites = Lx * Ly
    num_qubits = 2 * num_sites
    num_trotter_steps = 1
    timesteps = 1
    dt = 0.01
    total_time = dt * timesteps
    print("Total time: ", total_time)

    ##################################
    # Reference simulation
    ##################################
    state_list = sum(([qt.basis(2, 0), qt.basis(2, 0)] if x >= L/2 else [qt.basis(2, 1), qt.basis(2, 1)] for x in range(L)), [])    # brick wall state
    initial_state = qt.tensor(state_list)

    H = create_fermi_hubbard_model_qutip(Lx, Ly, u, -t, mu)

    tlist = np.linspace(0, total_time, timesteps)
    result = qt.mesolve(H, initial_state, tlist, [], [])
    occupations_qutip = np.zeros((num_qubits, timesteps), dtype='complex')

    for x in range(Lx):
        for y in range(Ly):
            i_up = site_index(x, y, '↑', Lx)
            i_down = site_index(x, y, '↓', Lx)
            n_up = qt.fcreate(n_sites=2*L, site=i_up) * qt.fdestroy(n_sites=2*L, site=i_up)
            n_down = qt.fcreate(n_sites=2*L, site=i_down) * qt.fdestroy(n_sites=2*L, site=i_down)
            occupations_qutip[i_up, :] = qt.expect(n_up, result.states)
            occupations_qutip[i_down, :] = qt.expect(n_down, result.states)

    ##################################
    # YAQS simulation
    ##################################

    # Define the simulation parameters
    N = 1
    window_size = 0
    measurements = [Observable('z', site) for site in range(num_qubits)]

    #max_bond_dim_list = range(2, 2**num_qubits//2+1, 2)
    #threshold_list = [10**-x for x in range(1, 13)]

    max_bond_dim_list = range(2, 2**3//2+1, 2)
    threshold_list = [10**-x for x in range(1, 5)]

    heatmap = np.empty((len(max_bond_dim_list), len(threshold_list)))

    for i, max_bond_dim in enumerate(max_bond_dim_list):
        for j, threshold in enumerate(threshold_list):
            # Define the initial state
            state = MPS(num_qubits, state='wall', pad=32)

            # Run the simulation
            occupations_yaqs = np.zeros((num_qubits, timesteps), dtype='complex')

            for timestep in range(timesteps):
                print("Timestep: " + str(timestep))
                circuit = create_2d_fermi_hubbard_circuit(Lx=Lx, Ly=Ly, u=u, t=t, mu=mu,
                                                        num_trotter_steps=num_trotter_steps,
                                                        dt=dt, timesteps=1)

                sim_params = StrongSimParams(measurements, N, max_bond_dim, threshold, window_size, get_state=True)

                simulator.run(state, circuit, sim_params, None)

                for observable in sim_params.observables:
                    index = observable.site
                    occupations_yaqs[index, timestep] = 0.5 * (1 - observable.results.item())

                state = MPS(num_qubits, sim_params.output_state.tensors, pad=32)

            # Calculate error
            error = np.linalg.norm(occupations_yaqs[:,-1][::-1] - occupations_qutip[:,-1], 2)
            print("max bond dim: ", max_bond_dim, "\t threshold: ", threshold)
            print(error)
            heatmap[i, j] = error

    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(heatmap.T, aspect="auto", norm=LogNorm())
    title = r"$\Delta t = $" + str(dt) + ", Timesteps = " + str(timesteps) + ", Total simulation time T = " + str(total_time)
    ax.set_title(title)
    ax.set_xlabel("max bond dim")
    ax.set_ylabel("threshold")
    formatted_threshold_vals = [f"$10^{{{int(np.log10(t))}}}$" for t in threshold_list]
    tick_positions = np.arange(0, len(max_bond_dim_list), 10)
    tick_labels = [str(max_bond_dim_list[i]) for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(range(len(threshold_list)))
    ax.set_yticklabels(formatted_threshold_vals)
    fig.subplots_adjust(top=0.95, right=0.88)
    cbar_ax = fig.add_axes(rect=(0.9, 0.11, 0.025, 0.8))
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.set_title("$\\langle Z \\rangle$")

    plt.show()


if __name__ == "__main__":
    benchmarker()
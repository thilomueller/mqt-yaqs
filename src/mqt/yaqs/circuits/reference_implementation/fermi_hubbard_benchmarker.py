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
import matplotlib.pyplot as plt

from mqt.yaqs.core.data_structures.networks import MPS
from mqt.yaqs.core.data_structures.simulation_parameters import Observable, StrongSimParams
from mqt.yaqs.core.libraries.circuit_library import create_2d_fermi_hubbard_circuit

from mqt.yaqs import simulator


def benchmarker() -> None:
    # Define the model
    t = 1.0         # kinetic hopping
    mu = 0.5        # chemical potential
    u = 4.0         # onsite interaction
    Lx, Ly = 2, 2   # lattice dimensions

    num_sites = Lx * Ly
    num_qubits = 2 * num_sites
    num_trotter_steps = 1
    timesteps = 1000
    dt = 0.01
    total_time = dt * timesteps
    print("Total time: ", total_time)

    # Define the initial state
    state = MPS(num_qubits, state='wall', pad=32)

    # Define the simulation parameters
    N = 1
    max_bond_dim = 16
    threshold = 1e-6
    window_size = 0
    measurements = [Observable('z', site) for site in range(num_qubits)]

    # Run the simulation
    occupations = np.zeros((num_qubits, timesteps), dtype='complex')

    for timestep in range(timesteps):
        print("Timestep: " + str(timestep))
        circuit = create_2d_fermi_hubbard_circuit(Lx=Lx, Ly=Ly, u=u, t=t, mu=mu,
                                                num_trotter_steps=num_trotter_steps,
                                                dt=dt, timesteps=1)

        sim_params = StrongSimParams(measurements, N, max_bond_dim, threshold, window_size, get_state=True)

        simulator.run(state, circuit, sim_params, None)

        for observable in sim_params.observables:
            index = observable.site
            occupations[index, timestep] = 0.5 * (1 - observable.results.item())

        state = MPS(num_qubits, sim_params.output_state.tensors, pad=32)

    plt.figure(figsize=(10, 5))
    for i in range(num_qubits):
        site = i // 2
        spin = "↑" if i % 2 == 0 else '↓'
        plt.plot(occupations[i, :], label=f"Site {site} " + spin)

    plt.xlabel("Time")
    plt.ylabel("Occupation")
    plt.title("2D Hubbard Model: Time Evolution of Site Occupations (mqt-yaqs)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    benchmarker()
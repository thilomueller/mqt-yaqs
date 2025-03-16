# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Library of useful quantum circuits.

This module provides functions for creating quantum circuits that simulate
the dynamics of the Ising and Heisenberg models. The functions create_ising_circuit
and create_Heisenberg_circuit construct Qiskit QuantumCircuit objects based on specified
parameters such as the number of qubits, interaction strengths, time steps, and total simulation time.
These circuits are used to simulate the evolution of quantum many-body systems under the
respective Hamiltonians.
"""

# ignore non-lowercase argument names for physics notation
# ruff: noqa: N803

from __future__ import annotations

from qiskit.circuit import QuantumCircuit, QuantumRegister


def create_ising_circuit(
    L: int, J: float, g: float, dt: float, timesteps: int, *, periodic: bool = False
) -> QuantumCircuit:
    """Ising Trotter circuit with optional periodic boundary conditions.

    Create a quantum circuit for simulating the Ising model. When periodic is True,
    a long-range rzz gate is added between the last and first qubits in each timestep.

    Args:
        L (int): Number of qubits in the circuit.
        J (float): Coupling constant for the ZZ interaction.
        g (float): Transverse field strength.
        dt (float): Time step for the simulation.
        timesteps (int): Number of time steps to simulate.
        periodic (bool, optional): If True, add a long-range gate between qubits 0 and L-1.
                                   Defaults to False.

    Returns:
        QuantumCircuit: A quantum circuit representing the Ising model evolution.
    """
    # Angle on X rotation
    alpha = -2 * dt * g
    # Angle on ZZ rotation
    beta = -2 * dt * J

    circ = QuantumCircuit(L)
    for _ in range(timesteps):
        # Apply RX rotations on all qubits.
        for site in range(L):
            circ.rx(theta=alpha, qubit=site)

        # Even-odd nearest-neighbor interactions.
        for site in range(L // 2):
            circ.rzz(beta, qubit1=2 * site, qubit2=2 * site + 1)

        # Odd-even nearest-neighbor interactions.
        for site in range(1, L // 2):
            circ.rzz(beta, qubit1=2 * site - 1, qubit2=2 * site)

        # For odd L > 1, handle the last pair.
        if L % 2 != 0 and L != 1:
            circ.rzz(beta, qubit1=L - 2, qubit2=L - 1)

        # If periodic, add an additional long-range gate between qubit L-1 and qubit 0.
        if periodic and L > 1:
            circ.rzz(beta, qubit1=0, qubit2=L - 1)

    return circ


def create_2d_ising_circuit(
    num_rows: int, num_cols: int, J: float, g: float, dt: float, timesteps: int
) -> QuantumCircuit:
    """2D Ising Trotter circuit on a rectangular grid using a snaking MPS ordering.

    Args:
        num_rows (int): Number of rows in the qubit grid.
        num_cols (int): Number of columns in the qubit grid.
        J (float): Coupling constant for the ZZ interaction.
        g (float): Transverse field strength.
        dt (float): Time step for the simulation.
        timesteps (int): Number of Trotter steps.

    Returns:
        QuantumCircuit: A quantum circuit representing the 2D Ising model evolution with MPS-friendly ordering.
    """
    total_qubits = num_rows * num_cols
    circ = QuantumCircuit(total_qubits)

    # Define a helper function to compute the snaking index.
    def site_index(row: int, col: int) -> int:
        # For even rows, map left-to-right; for odd rows, map right-to-left.
        if row % 2 == 0:
            return row * num_cols + col
        return row * num_cols + (num_cols - 1 - col)

    # Single-qubit rotation and ZZ interaction angles.
    alpha = -2 * dt * g
    beta = -2 * dt * J

    for _ in range(timesteps):
        # Apply RX rotations to all qubits according to the snaking order.
        for row in range(num_rows):
            for col in range(num_cols):
                q = site_index(row, col)
                circ.rx(alpha, q)

        # Horizontal interactions: within each row, apply rzz gates between adjacent qubits.
        for row in range(num_rows):
            # Even bonds in the row.
            for col in range(0, num_cols - 1, 2):
                q1 = site_index(row, col)
                q2 = site_index(row, col + 1)
                circ.rzz(beta, q1, q2)
            # Odd bonds in the row.
            for col in range(1, num_cols - 1, 2):
                q1 = site_index(row, col)
                q2 = site_index(row, col + 1)
                circ.rzz(beta, q1, q2)

        # Vertical interactions: between adjacent rows.
        for col in range(num_cols):
            # Even bonds vertically.
            for row in range(0, num_rows - 1, 2):
                q1 = site_index(row, col)
                q2 = site_index(row + 1, col)
                circ.rzz(beta, q1, q2)
            # Odd bonds vertically.
            for row in range(1, num_rows - 1, 2):
                q1 = site_index(row, col)
                q2 = site_index(row + 1, col)
                circ.rzz(beta, q1, q2)

    return circ


def create_heisenberg_circuit(
    L: int, Jx: float, Jy: float, Jz: float, h: float, dt: float, timesteps: int
) -> QuantumCircuit:
    """Heisenberg Trotter circuit.

    Create a quantum circuit for simulating the Heisenberg model.

    Args:
        L (int): Number of qubits (sites) in the circuit.
        Jx (float): Coupling constant for the XX interaction.
        Jy (float): Coupling constant for the YY interaction.
        Jz (float): Coupling constant for the ZZ interaction.
        h (float): Magnetic field strength.
        dt (float): Time step for the simulation.
        timesteps (int): Number of time steps to simulate.

    Returns:
        QuantumCircuit: A quantum circuit representing the Heisenberg model evolution.
    """
    theta_xx = -2 * dt * Jx
    theta_yy = -2 * dt * Jy
    theta_zz = -2 * dt * Jz
    theta_z = -2 * dt * h

    circ = QuantumCircuit(L)
    for _ in range(timesteps):
        # Z application
        for site in range(L):
            circ.rz(phi=theta_z, qubit=site)

        # ZZ application
        for site in range(L // 2):
            circ.rzz(theta=theta_zz, qubit1=2 * site, qubit2=2 * site + 1)

        for site in range(1, L // 2):
            circ.rzz(theta=theta_zz, qubit1=2 * site - 1, qubit2=2 * site)

        if L % 2 != 0 and L != 1:
            circ.rzz(theta=theta_zz, qubit1=L - 2, qubit2=L - 1)

        # XX application
        for site in range(L // 2):
            circ.rxx(theta=theta_xx, qubit1=2 * site, qubit2=2 * site + 1)

        for site in range(1, L // 2):
            circ.rxx(theta=theta_xx, qubit1=2 * site - 1, qubit2=2 * site)

        if L % 2 != 0 and L != 1:
            circ.rxx(theta=theta_xx, qubit1=L - 2, qubit2=L - 1)

        # YY application
        for site in range(L // 2):
            circ.ryy(theta=theta_yy, qubit1=2 * site, qubit2=2 * site + 1)

        for site in range(1, L // 2):
            circ.ryy(theta=theta_yy, qubit1=2 * site - 1, qubit2=2 * site)

        if L % 2 != 0 and L != 1:
            circ.ryy(theta=theta_yy, qubit1=L - 2, qubit2=L - 1)

    return circ


def create_2d_heisenberg_circuit(
    num_rows: int, num_cols: int, Jx: float, Jy: float, Jz: float, h: float, dt: float, timesteps: int
) -> QuantumCircuit:
    """2D Heisenberg Trotter circuit on a rectangular grid using a snaking MPS ordering.

    Args:
        num_rows (int): Number of rows in the qubit grid.
        num_cols (int): Number of columns in the qubit grid.
        Jx (float): Coupling constant for the XX interaction.
        Jy (float): Coupling constant for the YY interaction.
        Jz (float): Coupling constant for the ZZ interaction.
        h (float): Single-qubit Z-field strength.
        dt (float): Time step for the simulation.
        timesteps (int): Number of Trotter steps.

    Returns:
        QuantumCircuit: A quantum circuit representing the 2D Heisenberg model evolution
                       with MPS-friendly ordering.
    """
    total_qubits = num_rows * num_cols
    circ = QuantumCircuit(total_qubits)

    # Define a helper function to compute the snaking index.
    def site_index(row: int, col: int) -> int:
        # For even rows, map left-to-right; for odd rows, map right-to-left.
        if row % 2 == 0:
            return row * num_cols + col
        return row * num_cols + (num_cols - 1 - col)

    # Define the Trotter angles
    theta_xx = -2.0 * dt * Jx
    theta_yy = -2.0 * dt * Jy
    theta_zz = -2.0 * dt * Jz
    theta_z = -2.0 * dt * h

    for _ in range(timesteps):
        # (1) Apply single-qubit Z rotations to all qubits
        for row in range(num_rows):
            for col in range(num_cols):
                q = site_index(row, col)
                circ.rz(theta_z, q)

        # (2) ZZ interactions
        # Horizontal even bonds
        for row in range(num_rows):
            for col in range(0, num_cols - 1, 2):
                q1 = site_index(row, col)
                q2 = site_index(row, col + 1)
                circ.rzz(theta_zz, q1, q2)
        # Horizontal odd bonds
        for row in range(num_rows):
            for col in range(1, num_cols - 1, 2):
                q1 = site_index(row, col)
                q2 = site_index(row, col + 1)
                circ.rzz(theta_zz, q1, q2)
        # Vertical even bonds
        for col in range(num_cols):
            for row in range(0, num_rows - 1, 2):
                q1 = site_index(row, col)
                q2 = site_index(row + 1, col)
                circ.rzz(theta_zz, q1, q2)
        # Vertical odd bonds
        for col in range(num_cols):
            for row in range(1, num_rows - 1, 2):
                q1 = site_index(row, col)
                q2 = site_index(row + 1, col)
                circ.rzz(theta_zz, q1, q2)

        # (3) XX interactions
        # Horizontal even bonds
        for row in range(num_rows):
            for col in range(0, num_cols - 1, 2):
                q1 = site_index(row, col)
                q2 = site_index(row, col + 1)
                circ.rxx(theta_xx, q1, q2)
        # Horizontal odd bonds
        for row in range(num_rows):
            for col in range(1, num_cols - 1, 2):
                q1 = site_index(row, col)
                q2 = site_index(row, col + 1)
                circ.rxx(theta_xx, q1, q2)
        # Vertical even bonds
        for col in range(num_cols):
            for row in range(0, num_rows - 1, 2):
                q1 = site_index(row, col)
                q2 = site_index(row + 1, col)
                circ.rxx(theta_xx, q1, q2)
        # Vertical odd bonds
        for col in range(num_cols):
            for row in range(1, num_rows - 1, 2):
                q1 = site_index(row, col)
                q2 = site_index(row + 1, col)
                circ.rxx(theta_xx, q1, q2)

        # (4) YY interactions
        # Horizontal even bonds
        for row in range(num_rows):
            for col in range(0, num_cols - 1, 2):
                q1 = site_index(row, col)
                q2 = site_index(row, col + 1)
                circ.ryy(theta_yy, q1, q2)
        # Horizontal odd bonds
        for row in range(num_rows):
            for col in range(1, num_cols - 1, 2):
                q1 = site_index(row, col)
                q2 = site_index(row, col + 1)
                circ.ryy(theta_yy, q1, q2)
        # Vertical even bonds
        for col in range(num_cols):
            for row in range(0, num_rows - 1, 2):
                q1 = site_index(row, col)
                q2 = site_index(row + 1, col)
                circ.ryy(theta_yy, q1, q2)
        # Vertical odd bonds
        for col in range(num_cols):
            for row in range(1, num_rows - 1, 2):
                q1 = site_index(row, col)
                q2 = site_index(row + 1, col)
                circ.ryy(theta_yy, q1, q2)

    return circ


def create_1D_Fermi_Hubbard_circuit(model, dt, timesteps):
    'H = -1/2 mu (I-Z) + 1/4 u (I-Z) (I-Z) - 1/2 t (XX + YY)'

    assert model['name'] == '1D_Fermi_Hubbard'

    mu = model['mu']
    u = model['u']
    t = model['t']
    n = model['num_trotter_steps']

    spin_up = QuantumRegister(model['L'], '↑')
    spin_down = QuantumRegister(model['L'], '↓')
    circ = QuantumCircuit(spin_up, spin_down)

    def H_1():
        """Add the time evolution of the chemical potential term"""
        theta = mu*dt/(2*n)
        for j in range(model['L']):
            circ.p(theta=theta, qubit=spin_up[j])
            circ.p(theta=theta, qubit=spin_down[j])

    def H_2():
        """Add the time evolution of the onsite interaction term"""
        theta = -u*dt/(2*n)
        for j in range(model['L']):
            circ.cp(theta=theta, control_qubit=spin_up[j], target_qubit=spin_down[j])

    def H_3():
        """Add the time evolution of the kinetic hopping term"""
        theta = -dt*t/n
        for j in range(model['L']-1):
            if j % 2 == 0:
                circ.rxx(theta=theta, qubit1=spin_up[j+1], qubit2=spin_up[j])
                circ.ryy(theta=theta, qubit1=spin_up[j+1], qubit2=spin_up[j])
                circ.rxx(theta=theta, qubit1=spin_down[j+1], qubit2=spin_down[j])
                circ.ryy(theta=theta, qubit1=spin_down[j+1], qubit2=spin_down[j])
        for j in range(model['L']-1):
            if j % 2 != 0:
                circ.rxx(theta=theta, qubit1=spin_up[j+1], qubit2=spin_up[j])
                circ.ryy(theta=theta, qubit1=spin_up[j+1], qubit2=spin_up[j])
                circ.rxx(theta=theta, qubit1=spin_down[j+1], qubit2=spin_down[j])
                circ.ryy(theta=theta, qubit1=spin_down[j+1], qubit2=spin_down[j])

    for _ in range(n*timesteps):
        H_1()
        H_2()
        H_3()
        H_2()
        H_1()
    
    return circ


def lookup_qiskit_ordering(particle, spin) -> int:
    """
    Looks up the Qiskit mapping from a 2D lattice to a 1D qubit-line
    
    Parameters:
        particle (int): The index of the particle in the physical lattice.
        spin (int): 0 for spin up, 1 for spin down.
    
    Returns:
        int: The index in the 1D qubit-line.
    """
    if spin == '↑':
        spin = 0
    elif spin == '↓':
        spin = 1

    if spin not in (0, 1):
        raise ValueError("spin must be 0 or 1")
    
    return 2*particle + spin


def add_long_range_interaction(circ, i, j, outer_op, alpha):
    """
    Add a long range interaction that is decomposed into two-qubit gates.
    outer_op=X: X_i ⊗ Z_{i+1} ⊗ ... ⊗ Z_{j-1} ⊗ X_j
    outer_op=Y: Y_i ⊗ Z_{i+1} ⊗ ... ⊗ Y_{j-1} ⊗ X_j
    """
    import numpy as np
    
    if i >= j:
        raise Exception("Assumption i < j violated.")
    
    phi = 1*alpha
    circ.rz(phi=phi, qubit=j)

    for k in range(i, j):
        # prepend the CNOT gate
        aux_circ = QuantumCircuit(circ.num_qubits)
        aux_circ.cx(control_qubit=k, target_qubit=j)
        circ.compose(aux_circ, front=True, inplace=True)
        # append the CNOT gate
        circ.cx(control_qubit=k, target_qubit=j)
    if outer_op == 'x' or outer_op == 'X':
        theta = np.pi/2
        # prepend the Ry gates
        aux_circ = QuantumCircuit(circ.num_qubits)
        aux_circ.ry(theta=theta, qubit=i)
        aux_circ.ry(theta=theta, qubit=j)
        circ.compose(aux_circ, front=True, inplace=True)
        # append the same Ry gates with negative phase
        circ.ry(theta=-theta, qubit=i)
        circ.ry(theta=-theta, qubit=j)
    elif outer_op == 'y' or outer_op == 'Y':
        theta = np.pi/2
        # prepend the Rx gates
        aux_circ = QuantumCircuit(circ.num_qubits)
        aux_circ.rx(theta=theta, qubit=i)
        aux_circ.rx(theta=theta, qubit=j)
        circ.compose(aux_circ, front=True, inplace=True)
        # append the same Rx gates with negative phase
        circ.rx(theta=-theta, qubit=i)
        circ.rx(theta=-theta, qubit=j)
    else:
        raise Exception("Only Pauli X or Y matrices are supported as outer operator.")
    

def add_hopping_term(circ, i, j, alpha):
    """
    Adds a hopping operator of the form
    exp(-i*(X_i ⊗ Z_{i+1} ⊗ ... ⊗ Z_{j-1} ⊗ X_j + Y_i ⊗ Z_{i+1} ⊗ ... ⊗ Z_{j-1} ⊗ Y_j))
    to the circuit.
    """
    XX = QuantumCircuit(circ.num_qubits)
    YY = QuantumCircuit(circ.num_qubits)
    add_long_range_interaction(XX, i, j, 'X', alpha)
    add_long_range_interaction(YY, i, j, 'Y', alpha)
    circ.compose(XX, inplace=True)
    circ.compose(YY, inplace=True)


def create_2D_Fermi_Hubbard_circuit(model, dt, timesteps):
    'H = -1/2 mu (I-Z) + 1/4 u (I-Z) (I-Z) - 1/2 t (XX + YY)'

    assert model['name'] == '2D_Fermi_Hubbard'

    mu = model['mu']
    u = model['u']
    t = model['t']
    n = model['num_trotter_steps']
    L = model['Lx'] * model['Ly']
    N = 2*L

    circ = QuantumCircuit(N)

    def H_1():
        """Add the time evolution of the chemical potential term"""
        theta = -mu*dt/(2*n)
        for j in range(L):
            q_up = lookup_qiskit_ordering(j, '↑')
            q_down = lookup_qiskit_ordering(j, '↓')
            circ.p(theta=theta, qubit=q_up)
            circ.p(theta=theta, qubit=q_down)

    def H_2():
        """Add the time evolution of the onsite interaction term"""
        theta = -u*dt/(2*n)
        for j in range(L):
            q_up = lookup_qiskit_ordering(j, '↑')
            q_down = lookup_qiskit_ordering(j, '↓')
            circ.cp(theta=theta, control_qubit=q_up, target_qubit=q_down)

    def H_3():
        """Add the time evolution of the kinetic hopping term"""
        alpha = t*dt/n

        def horizontal_odd():
            for y in range(model['Ly']):
                for x in range(model['Lx']-1):
                    if x % 2 == 0:
                        p1 = y*model['Lx'] + x
                        p2 = p1 + 1
                        #print("particle (" + str(p1) + ", " + str(p2) + ")")
                        q1_up = lookup_qiskit_ordering(p1, '↑')
                        q2_up = lookup_qiskit_ordering(p2, '↑')
                        q1_down = lookup_qiskit_ordering(p1, '↓')
                        q2_down = lookup_qiskit_ordering(p2, '↓')
                        add_hopping_term(circ, q1_up, q2_up, alpha)
                        add_hopping_term(circ, q1_down, q2_down, alpha)
                        #print("qubit ↑ (" + str(q1_up) + ", " + str(q2_up) + ")")
                        #print("qubit ↓ (" + str(q1_down) + ", " + str(q2_down) + ")")
                        #print("--")

        def horizontal_even():
            for y in range(model['Ly']):
                for x in range(model['Lx']-1):
                    if x % 2 != 0:
                        p1 = y*model['Lx'] + x
                        p2 = p1 + 1
                        #print("particle (" + str(p1) + ", " + str(p2) + ")")
                        q1_up = lookup_qiskit_ordering(p1, '↑')
                        q2_up = lookup_qiskit_ordering(p2, '↑')
                        q1_down = lookup_qiskit_ordering(p1, '↓')
                        q2_down = lookup_qiskit_ordering(p2, '↓')
                        add_hopping_term(circ, q1_up, q2_up, alpha)
                        add_hopping_term(circ, q1_down, q2_down, alpha)
                        #print("qubit ↑ (" + str(q1_up) + ", " + str(q2_up) + ")")
                        #print("qubit ↓ (" + str(q1_down) + ", " + str(q2_down) + ")")
                        #print("--")

        def vertical_odd():
            for y in range(model['Ly']-1):
                if y % 2 == 0:
                    for x in range(model['Lx']):
                        p1 = y*model['Lx'] + x
                        p2 = p1 + model['Lx']
                        #print("particle (" + str(p1) + ", " + str(p2) + ")")
                        q1_up = lookup_qiskit_ordering(p1, '↑')
                        q2_up = lookup_qiskit_ordering(p2, '↑')
                        q1_down = lookup_qiskit_ordering(p1, '↓')
                        q2_down = lookup_qiskit_ordering(p2, '↓')
                        add_hopping_term(circ, q1_up, q2_up, alpha)
                        add_hopping_term(circ, q1_down, q2_down, alpha)
                        #print("qubit ↑ (" + str(q1_up) + ", " + str(q2_up) + ")")
                        #print("qubit ↓ (" + str(q1_down) + ", " + str(q2_down) + ")")
                        #print("--")  

        def vertical_even():
            for y in range(model['Ly']-1):
                if y % 2 != 0:
                    for x in range(model['Lx']):
                        p1 = y*model['Lx'] + x
                        p2 = p1 + model['Lx']
                        #print("particle (" + str(p1) + ", " + str(p2) + ")")
                        q1_up = lookup_qiskit_ordering(p1, '↑')
                        q2_up = lookup_qiskit_ordering(p2, '↑')
                        q1_down = lookup_qiskit_ordering(p1, '↓')
                        q2_down = lookup_qiskit_ordering(p2, '↓')
                        add_hopping_term(circ, q1_up, q2_up, alpha)
                        add_hopping_term(circ, q1_down, q2_down, alpha)
                        #print("qubit ↑ (" + str(q1_up) + ", " + str(q2_up) + ")")
                        #print("qubit ↓ (" + str(q1_down) + ", " + str(q2_down) + ")")
                        #print("--")
        
        #for _ in range(n):
        horizontal_odd()
        horizontal_even()
        vertical_odd()
        vertical_even()

    for _ in range(timesteps):
        for _ in range(n):
            H_1()
            H_2()
            H_3()
            H_2()
            H_1()
    
    return circ
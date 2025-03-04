import copy
import numpy as np
import pytest

from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag

from yaqs.core.data_structures.networks import MPS

from yaqs.circuits.CircuitTJM import (
    process_layer,
    apply_single_qubit_gate,
    construct_generator_MPO,
    apply_window,
    apply_two_qubit_gate,
    CircuitTJM,
)

def test_process_layer():
    # Create a QuantumCircuit with 9 qubits and 9 classical bits.
    qc = QuantumCircuit(9, 9)
    qc.measure(0, 0)
    qc.barrier(1)
    qc.x(qc.qubits[2])
    qc.cx(4, 5)
    qc.cx(7, 8)
    
    # Convert the circuit to a DAG.
    dag = circuit_to_dag(qc)
    
    # Call process_layer on the DAG.
    single, even, odd = process_layer(dag)
    
    # After processing, the measurement and barrier nodes should have been removed.
    for node in dag.op_nodes():
        assert node.op.name not in ['measure', 'barrier'], \
            f"Unexpected node {node.op.name} in the DAG op nodes."

    # Verify that the single-qubit gate is in the single-qubit group.
    single_names = [node.op.name.lower() for node in single]
    assert any("x" in name for name in single_names), "X gate not found in single group."
    
    # Verify the grouping of two-qubit gates.
    # For each node in the even group, the lower qubit index should be even.
    for node in even:
        q0 = node.qargs[0]._index
        q1 = node.qargs[1]._index
        assert min(q0, q1) % 2 == 0, f"Node with qubits {q0, q1} not in even group."
    
    # For each node in the odd group, the lower qubit index should be odd.
    for node in odd:
        q0 = node.qargs[0]._index
        q1 = node.qargs[1]._index
        assert min(q0, q1) % 2 == 1, f"Node with qubits {q0, q1} not in odd group."


def test_apply_single_qubit_gate():
    from yaqs.core.libraries.gate_library import GateLibrary
    mps = MPS(length=1)
    tensor = mps.tensors[0]

    qc = QuantumCircuit(1)
    qc.x(0)

    dag = circuit_to_dag(qc)
    node = dag.front_layer()[0]
    
    apply_single_qubit_gate(mps, node)

    gate_tensor = getattr(GateLibrary, 'x').tensor
    expected = np.einsum('ab,bcd->acd', gate_tensor, tensor)
    np.testing.assert_allclose(mps.tensors[0], expected)

def test_construct_generator_MPO():
    from yaqs.core.libraries.gate_library import GateLibrary
    gate = getattr(GateLibrary, 'cx')()
    gate.set_sites(1, 3)
    length = 5
    mpo, first_site, last_site = construct_generator_MPO(gate, length)
    for tensor in mpo.tensors:
        print(tensor)
    # The first and last sites should be 1 and 3.
    assert first_site == 1
    assert last_site == 3
    # Check the MPO tensors at the designated sites.
    np.testing.assert_allclose(np.squeeze(np.transpose(mpo.tensors[1], (2, 3, 0, 1))), np.complex128(gate.generator[0]))
    np.testing.assert_allclose(np.squeeze(np.transpose(mpo.tensors[3], (2, 3, 0, 1))), np.complex128(gate.generator[1]))
    # All other tensors should be the identity.
    for i in range(length):
        if i not in [1, 3]:
            np.testing.assert_allclose(np.squeeze(np.transpose(mpo.tensors[i], (2, 3, 0, 1))), np.eye(2, dtype=complex))

def test_apply_window():
    # Create dummy MPS and MPO objects with 5 tensors.
    length = 5
    tensors = [np.full((2, 1, 1), i, dtype=complex) for i in range(5)]
    mps = MPS(length, tensors)

    from yaqs.core.libraries.gate_library import GateLibrary
    gate = getattr(GateLibrary, 'cx')()
    gate.set_sites(1, 2)
    mpo, first_site, last_site = construct_generator_MPO(gate, length)

    from yaqs.core.data_structures.simulation_parameters import Observable, StrongSimParams
    N = 1
    max_bond_dim = 4
    threshold = 1e-12
    window_size = 1
    measurements = [Observable('z', 0)]
    sim_params = StrongSimParams(measurements, N, max_bond_dim, threshold, window_size)

    short_state, short_mpo, window = apply_window(mps, mpo, first_site, last_site, sim_params)
    
    assert window == [0, 3]
    # The short state should have tensors corresponding to indices 1 through 4.
    assert short_state.length == 4
    assert short_mpo.length == 4

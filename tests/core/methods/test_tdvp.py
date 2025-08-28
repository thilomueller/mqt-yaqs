# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for Time-Dependent Variational Principle (TDVP) methods in YAQS.

This module contains unit tests for verifying various components of TDVP-based methods,
including:

- Tensor decomposition (splitting and merging) routines for Matrix Product States (MPS)
   and Matrix Product Operators (MPO).
- Environment updates (left and right) and projection routines essential to efficient TDVP implementations.
- Single-site and two-site TDVP algorithms for evolving quantum states under given Hamiltonians.

The tests ensure that:
- Tensor reshaping and decomposition operations maintain numerical accuracy.
- Environment tensors are updated with correct shapes and dimensions.
- TDVP routines yield canonical MPS states with the correct orthogonality center.

These tests confirm the correctness and stability of TDVP-based simulations within the YAQS framework.
"""

# ignore non-lowercase variable names for physics notation
# ruff: noqa: N806

from __future__ import annotations

from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pytest
from scipy.linalg import expm

from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
from mqt.yaqs.core.libraries.gate_library import X, Z
from mqt.yaqs.core.methods.tdvp import (
    global_dynamic_tdvp,
    merge_mpo_tensors,
    merge_mps_tensors,
    project_bond,
    project_site,
    single_site_tdvp,
    split_mps_tensor,
    two_site_tdvp,
    update_bond,
    update_left_environment,
    update_right_environment,
    update_site,
)

rng = np.random.default_rng()


def test_split_mps_tensor_left_right_sqrt() -> None:
    """Test splitting of an MPS tensor using different singular value distribution options.

    This test creates a random tensor A with shape (4, 3, 5), corresponding to (d0*d1, D0, D2)
    with d0 = d1 = 2, D0 = 3, and D2 = 5. For each SVD distribution option ("left", "right", "sqrt"),
    the function split_mps_tensor is called to decompose A into two tensors A0 and A1. The test then
    reconstructs A from A0 and A1 by undoing the transpose on A1 and contracting over the singular value index.
    The reconstructed tensor is compared to the original A.
    """
    A = rng.random(size=(4, 3, 5))
    # Placeholder
    measurements = [Observable(Z(), site) for site in range(1)]
    sim_params = AnalogSimParams(
        measurements,
        elapsed_time=0.2,
        dt=0.1,
        sample_timesteps=True,
        num_traj=1,
        max_bond_dim=100,
        trunc_mode="relative",
        threshold=0,
        order=1,
        show_progress=False,
    )
    physical_dimensions = [A.shape[0] // 2, A.shape[0] // 2]
    for distr in ["left", "right", "sqrt"]:
        A0, A1 = split_mps_tensor(
            A, svd_distribution=distr, sim_params=sim_params, physical_dimensions=physical_dimensions, dynamic=False
        )
        # A0 should have shape (2, 3, r) and A1 should have shape (2, r, 5), where r is the effective rank.
        assert A0.ndim == 3
        assert A1.ndim == 3
        r = A0.shape[2]
        assert A1.shape[1] == r
        # Reconstruct A: first undo the transpose on A1 so that its shape becomes (r, 2, 5)
        A1_recon = A1.transpose((1, 0, 2))
        # Contract A0 (indices: d0, D0, r) with A1_recon (indices: r, d1, D2) over the rank index r.
        # This yields a tensor of shape (d0, d1, D0, D2). Then, we transpose to (d0*d1, D0, D2)
        A_recon = np.tensordot(A0, A1_recon, axes=(2, 0))  # shape (2, 3, 2, 5)
        A_recon = A_recon.transpose((0, 2, 1, 3)).reshape(4, 3, 5)
        np.testing.assert_allclose(A, A_recon, atol=1e-6)


def test_split_mps_tensor_invalid_shape() -> None:
    """Test that split_mps_tensor raises a ValueError when the input tensor's first dimension is not divisible by 2.

    This test creates a tensor A with shape (3, 3, 5) and expects the function to raise an error.
    """
    A = rng.random(size=(3, 3, 5))
    # Placeholder
    measurements = [Observable(Z(), site) for site in range(1)]
    sim_params = AnalogSimParams(
        measurements,
        elapsed_time=0.2,
        dt=0.1,
        sample_timesteps=True,
        num_traj=1,
        max_bond_dim=100,
        threshold=1e-8,
        order=1,
        show_progress=False,
    )
    physical_dimensions = [3, 3]
    with pytest.raises(
        ValueError, match=r"The first dimension of the tensor must be a combination of the given physical dimensions."
    ):
        split_mps_tensor(
            A, svd_distribution="left", sim_params=sim_params, physical_dimensions=physical_dimensions, dynamic=False
        )


def test_merge_mps_tensors() -> None:
    """Test the merge_mps_tensors function.

    This test creates two tensors A0 and A1 with shapes (2, 3, 4) and (5, 4, 7), respectively.
    It then merges them via merge_mps_tensors. The expected shape is (10, 3, 7) because
    the contraction is performed over the third axis of A0 and the second axis of A1.
    """
    A0 = rng.random(size=(2, 3, 4))
    A1 = rng.random(size=(5, 4, 7))
    merged = merge_mps_tensors(A0, A1)
    assert merged.shape == (10, 3, 7)


def test_merge_mpo_tensors() -> None:
    """Test the merge_mpo_tensors function.

    This test creates two 4D arrays A0 and A1 with shapes (2, 3, 4, 5) and (7, 8, 5, 9), respectively.
    After merging via merge_mpo_tensors, the expected shape is (14, 24, 4, 9).
    """
    A0 = rng.random(size=(2, 3, 4, 5))
    A1 = rng.random(size=(7, 8, 5, 9))
    merged = merge_mpo_tensors(A0, A1)
    assert merged.shape == (14, 24, 4, 9)


def test_update_right_environment() -> None:
    """Test the update_right_environment function.

    This test creates dummy arrays A, B, W, and R with compatible shapes for the contraction
    operations defined in update_right_environment. It then verifies that the resulting tensor
    has the expected shape (3, 8, 9).
    """
    A = rng.random(size=(2, 3, 4))
    R = rng.random(size=(4, 5, 6))
    W = rng.random(size=(7, 2, 8, 5))
    B = rng.random(size=(7, 9, 6))
    Rnext = update_right_environment(A, B, W, R)
    assert Rnext.shape == (3, 8, 9)


def test_update_left_environment() -> None:
    """Test the update_left_environment function.

    This test constructs dummy arrays A, B, W, and L with compatible shapes for the contraction.
    It then verifies that the output is a 3D tensor.
    """
    A = rng.random(size=(3, 4, 10))
    B = rng.random(size=(7, 6, 8))
    L_arr = rng.random(size=(4, 5, 6))
    W = rng.random(size=(7, 3, 5, 9))
    Rnext = update_left_environment(A, B, W, L_arr)
    assert Rnext.ndim == 3


def test_project_site() -> None:
    """Test the project_site function.

    This test creates dummy tensors A, R, W, and L with appropriate shapes and checks that
    the output of project_site is a 3D tensor.
    """
    A = rng.random(size=(2, 3, 4))
    R = rng.random(size=(4, 5, 6))
    W = rng.random(size=(7, 2, 8, 5))
    L_arr = rng.random(size=(3, 8, 9))
    out = project_site(L_arr, R, W, A)
    assert out.ndim == 3


def test_project_bond() -> None:
    """Test the project_bond function.

    This test creates a bond tensor C and dummy tensors L and R with compatible shapes,
    and verifies that the output has the expected shape (6, 5).
    """
    C = rng.random(size=(2, 3))
    R = rng.random(size=(3, 4, 5))
    L_arr = rng.random(size=(2, 4, 6))
    out = project_bond(L_arr, R, C)
    assert out.shape == (6, 5)


def test_update_site() -> None:
    """Test the update_site function.

    This test creates a dummy MPS tensor A (shape (2,2,4)), along with tensors L, R, and W,
    and applies update_site with a small time step and a fixed number of Lanczos iterations.
    The output should have the same shape as the input tensor A.
    """
    A = rng.random(size=(2, 2, 4))
    R = rng.random(size=(4, 1, 4))
    W = rng.random(size=(2, 2, 1, 1))
    L_arr = rng.random(size=(2, 1, 2))
    dt = 0.05
    lanczos_iterations = 10
    out = update_site(L_arr, R, W, A, dt, lanczos_iterations)
    assert out.shape == A.shape, f"Expected shape {A.shape}, got {out.shape}"


def test_update_bond() -> None:
    """Test the update_bond function.

    This test creates a square bond tensor C and compatible dummy tensors R and L.
    It applies update_bond and checks that the output shape matches that of C.
    """
    C = rng.random(size=(2, 2))
    R = rng.random(size=(2, 2, 2))
    L_arr = rng.random(size=(2, 2, 2))
    dt = 0.05
    lanczos_iterations = 10
    out = update_bond(L_arr, R, C, dt, lanczos_iterations)
    assert out.shape == C.shape, f"Expected shape {C.shape}, got {out.shape}"


def test_single_site_tdvp() -> None:
    """Test the single_site_TDVP function.

    This test initializes an Ising MPO and an MPS of length 5 (initialized to 'zeros'),
    along with AnalogSimParams configured for a single trajectory update.
    It runs single_site_TDVP and verifies that the MPS remains of the same length, all tensors are numpy arrays,
    and the MPS is left in a canonical form with the orthogonality center at site 0.
    """
    L = 5
    J = 1
    g = 0.5
    H = MPO()
    H.init_ising(L, J, g)
    state = MPS(L, state="zeros")
    measurements = [Observable(Z(), site) for site in range(L)]
    sim_params = AnalogSimParams(
        measurements,
        elapsed_time=0.2,
        dt=0.1,
        sample_timesteps=True,
        num_traj=1,
        max_bond_dim=4,
        threshold=1e-6,
        order=1,
        show_progress=False,
    )
    single_site_tdvp(state, H, sim_params, numiter_lanczos=5)
    assert state.length == L
    for tensor in state.tensors:
        assert isinstance(tensor, np.ndarray)
    canonical_site = state.check_canonical_form()[0]
    assert canonical_site == 0, (
        f"MPS should be site-canonical at site 0 after single-site TDVP, but got canonical site: {canonical_site}"
    )


def test_two_site_tdvp() -> None:
    """Test the two_site_TDVP function.

    This test initializes an Ising MPO and an MPS of length 5, sets up AnalogSimParams,
    and runs two_site_TDVP. It checks that the MPS retains the correct number of tensors,
    that all tensors remain numpy arrays, and that the MPS is in canonical form with the orthogonality center at site 0.
    """
    L = 5
    J = 1
    g = 0.5
    H = MPO()
    H.init_ising(L, J, g)
    state = MPS(L, state="zeros")
    ref_mps = deepcopy(state)
    measurements = [Observable(Z(), site) for site in range(L)]
    sim_params = AnalogSimParams(
        measurements,
        elapsed_time=0.2,
        dt=0.1,
        sample_timesteps=True,
        num_traj=1,
        max_bond_dim=16,
        threshold=1e-12,
        order=1,
        show_progress=False,
    )
    two_site_tdvp(state, H, sim_params, numiter_lanczos=25)
    assert state.length == L
    for tensor in state.tensors:
        assert isinstance(tensor, np.ndarray)
    canonical_site = state.check_canonical_form()[0]
    assert canonical_site == 0, (
        f"MPS should be site-canonical at site 0 after two-site TDVP, but got canonical site: {canonical_site}"
    )
    # Check against exact evolution
    state_vec = ref_mps.to_vec()
    H_mat = H.to_matrix()
    U = expm(-1j * sim_params.dt * H_mat)
    state_vec = U @ state_vec
    found = state.to_vec()
    assert np.allclose(state_vec, found)


def test_dynamic_tdvp_one_site() -> None:
    """Test dynamic TDVP, single site.

    Test that dynamic_TDVP calls single_site_TDVP exactly once when the current maximum bond dimension
    exceeds sim_params.max_bond_dim.

    In this test, sim_params.max_bond_dim is set to 0 so that the current maximum bond dimension of the MPS,
    computed by state.get_max_bond(), is greater than 0. Therefore, the else branch of dynamic_TDVP should be
    taken, and single_site_tdvp should be called exactly once.
    """
    # Define the system Hamiltonian.
    L = 5
    J = 1
    g = 0.5
    H = MPO()
    H.init_ising(L, J, g)

    # Define the initial state.
    state = MPS(L, state="zeros")

    # Define simulation parameters with max_bond_dim set to 0.
    elapsed_time = 0.2
    dt = 0.1
    sample_timesteps = False
    num_traj = 1
    max_bond_dim = 0  # Force condition for single_site_TDVP.
    min_bond_dim = 2
    threshold = 1e-6
    order = 1
    measurements = [Observable(X(), site) for site in range(L)]
    sim_params = AnalogSimParams(
        measurements,
        elapsed_time,
        dt,
        num_traj,
        max_bond_dim,
        min_bond_dim,
        threshold,
        order,
        sample_timesteps=sample_timesteps,
        show_progress=False,
    )

    with patch("mqt.yaqs.core.methods.tdvp.single_site_tdvp") as mock_single_site:
        global_dynamic_tdvp(state, H, sim_params)
        mock_single_site.assert_called_once_with(state, H, sim_params)


def test_dynamic_tdvp_two_site() -> None:
    """Test dynamic TDVP, two site.

    Test that dynamic_TDVP calls two_site_TDVP exactly once when the current maximum bond dimension
    is less than or equal to sim_params.max_bond_dim.

    In this test, sim_params.max_bond_dim is set to 2, so if the current maximum bond dimension is ≤ 2,
    the if branch of dynamic_TDVP is executed and two_site_TDVP is called exactly once.
    """
    # Define the system Hamiltonian.
    L = 5
    J = 1
    g = 0.5
    H = MPO()
    H.init_ising(L, J, g)

    # Define the initial state.
    state = MPS(L, state="zeros")

    # Define simulation parameters with max_bond_dim set to 2.
    elapsed_time = 0.2
    dt = 0.1
    sample_timesteps = False
    num_traj = 1
    max_bond_dim = 8  # Force condition for two_site_tdvp.
    min_bond_dim = 2
    threshold = 1e-6
    order = 1
    measurements = [Observable(X(), site) for site in range(L)]
    sim_params = AnalogSimParams(
        measurements,
        elapsed_time,
        dt,
        num_traj,
        max_bond_dim,
        min_bond_dim,
        threshold,
        order,
        sample_timesteps=sample_timesteps,
        show_progress=False,
    )

    with patch("mqt.yaqs.core.methods.tdvp.two_site_tdvp") as mock_two_site:
        global_dynamic_tdvp(state, H, sim_params)
        mock_two_site.assert_called_once_with(state, H, sim_params, dynamic=True)


def _rand_unitary_like(m: int, n: int, *, seed: int) -> np.ndarray:
    """Return an m x n semi-unitary (orthonormal columns) for deterministic tests."""
    rng_local = np.random.default_rng(seed)
    A = rng_local.normal(size=(m, n)) + 1j * rng_local.normal(size=(m, n))
    # QR gives orthonormal columns
    Q, _ = np.linalg.qr(A)
    return Q[:, :n]


def _theta_from_singulars(s: np.ndarray, m: int, n: int, *, seed: int) -> np.ndarray:
    """Construct θ = U diag(s) V† with shape (m, n), using deterministic U,V.

    Returns:
        Re-constructed matrix
    """
    r = min(len(s), m, n)
    U = _rand_unitary_like(m, r, seed=seed)
    V = _rand_unitary_like(n, r, seed=seed + 1)
    S = np.diag(s[:r])
    return U @ S @ V.conj().T


def _as_input_tensor(theta: np.ndarray, d0: int, d1: int, d2: int, d3: int) -> np.ndarray:
    """Map (d0*D0, d1*D2) matrix to split_mps_tensor input shape (d0*d1, d2, d3).

    Returns:
        Reshaped input tensor
    """
    # theta: (d0*D0) x (d1*D2)
    t = theta.reshape(d0, d2, d1, d3).transpose(0, 2, 1, 3)  # (d0, d1, d2, d3)
    return t.reshape(d0 * d1, d2, d3)


@pytest.mark.parametrize(("svs", "threshold", "expected_keep"), [
    # Tail power (sum of discarded s^2) crosses threshold exactly at the boundary.
    (np.array([1.0, 0.5, 0.1, 0.01]), 1e-4, 3),                     # discard 0.01 -> 1e-4
    (np.array([1.0, 0.5, 0.01, 0.001]), 1e-4, 2),                   # 1e-4 + 1e-6 >= 1e-4
    (np.array([1.0, 0.2, 0.2, 0.2]), 0.2**2 + 0.2**2 + 0.2**2, 1),  # keep only the largest
])
def test_split_truncation_discarded_weight_kept_count(svs: np.NDArray[np.float64], threshold: float, expected_keep: int) -> None:
    """discarded_weight: keep count matches tail-power threshold; shapes consistent."""
    d0, d1, D0, D2 = 2, 2, 3, 3
    theta = _theta_from_singulars(svs, d0 * D0, d1 * D2, seed=11)
    A_in = _as_input_tensor(theta, d0, d1, D0, D2)

    # Minimal params; set trunc_mode after construction to avoid signature mismatch.
    measurements = [Observable(Z(), 0)]
    sim_params = AnalogSimParams(
        measurements,
        elapsed_time=0.2,
        dt=0.1,
        sample_timesteps=True,
        num_traj=1,
        max_bond_dim=100,
        threshold=threshold,
        order=1,
        show_progress=False,
        min_bond_dim=1,
    )
    sim_params.trunc_mode = "discarded_weight"

    A0, A1 = split_mps_tensor(
        A_in, svd_distribution="sqrt", sim_params=sim_params, physical_dimensions=[d0, d1], dynamic=True
    )
    keep = A0.shape[2]
    assert keep == expected_keep
    assert A1.shape[1] == keep

    # Verify tail power condition that triggered this selection.
    tail = svs[expected_keep:] if expected_keep < len(svs) else np.array([], dtype=svs.dtype)
    assert np.sum(tail**2) >= threshold or expected_keep == len(svs)


@pytest.mark.parametrize(("svs", "rel_thr", "expected_keep"), [
    # Keep all s_i strictly greater than rel_thr * s_max
    (np.array([1.0, 0.6, 0.4, 0.1]), 0.5, 2),   # keep 1.0, 0.6
    (np.array([1.0, 0.99, 0.98]), 0.95, 3),  # keep all
    (np.array([1.0, 0.49, 0.3]), 0.5, 1),  # keep only 1.0
])
def test_split_truncation_relative_kept_count(svs: np.NDArray[np.float64], rel_thr: float, expected_keep: int) -> None:
    """relative: keep count matches s_i/s_max > threshold; shapes consistent."""
    d0, d1, D0, D2 = 2, 3, 2, 3
    theta = _theta_from_singulars(svs, d0 * D0, d1 * D2, seed=12)
    A_in = _as_input_tensor(theta, d0, d1, D0, D2)

    measurements = [Observable(Z(), 0)]
    sim_params = AnalogSimParams(
        measurements,
        elapsed_time=0.2,
        dt=0.1,
        sample_timesteps=True,
        num_traj=1,
        max_bond_dim=100,
        threshold=rel_thr,   # interpreted as relative threshold in this mode
        order=1,
        show_progress=False,
        min_bond_dim=1,
    )
    sim_params.trunc_mode = "relative"

    A0, A1 = split_mps_tensor(
        A_in, svd_distribution="sqrt", sim_params=sim_params, physical_dimensions=[d0, d1], dynamic=True
    )
    keep = A0.shape[2]
    assert keep == expected_keep
    assert A1.shape[1] == keep

    # Sanity around threshold boundary (strict >)
    smax = float(np.max(svs))
    if expected_keep > 0:
        assert np.all(svs[:expected_keep] / smax > rel_thr)
    if expected_keep < len(svs):
        assert not (svs[expected_keep] / smax > rel_thr)


def test_split_truncation_min_max_bond_enforced() -> None:
    """min_bond_dim/max_bond_dim are respected in both modes."""
    svs = np.array([1.0, 0.9, 0.8, 0.7])
    d0, d1, D0, D2 = 2, 2, 3, 3
    theta = _theta_from_singulars(svs, d0 * D0, d1 * D2, seed=13)
    A_in = _as_input_tensor(theta, d0, d1, D0, D2)

    measurements = [Observable(Z(), 0)]

    # relative would keep >2, cap at max_bond_dim=2
    sim_rel = AnalogSimParams(
        measurements,
        elapsed_time=0.2,
        dt=0.1,
        sample_timesteps=True,
        num_traj=1,
        max_bond_dim=2,
        threshold=0.5,
        order=1,
        show_progress=False,
        min_bond_dim=2,
    )
    sim_rel.trunc_mode = "relative"
    A0, A1 = split_mps_tensor(A_in, "sqrt", sim_rel, [d0, d1], dynamic=True)
    assert A0.shape[2] == 2
    assert A1.shape[1] == 2

    # discarded_weight would keep 1 for high threshold; min_bond_dim=2 lifts it to 2
    sim_dw = AnalogSimParams(
        measurements,
        elapsed_time=0.2,
        dt=0.1,
        sample_timesteps=True,
        num_traj=1,
        max_bond_dim=10,
        threshold=0.49,  # tail-power trigger
        order=1,
        show_progress=False,
        min_bond_dim=2,
    )
    sim_dw.trunc_mode = "discarded_weight"
    A0, A1 = split_mps_tensor(A_in, "sqrt", sim_dw, [d0, d1], dynamic=True)
    assert A0.shape[2] == 2
    assert A1.shape[1] == 2


@pytest.mark.parametrize("distr", ["left", "right", "sqrt"])
def test_split_truncation_distribution_reconstructs_optimal_rank(distr: str) -> None:
    """All SVD distribution choices reconstruct the optimal rank-k approximation."""
    svs = np.array([1.0, 0.7, 0.3, 0.1])
    d0, d1, D0, D2 = 2, 2, 3, 3
    theta = _theta_from_singulars(svs, d0 * D0, d1 * D2, seed=14)
    A_in = _as_input_tensor(theta, d0, d1, D0, D2)

    # Use a very permissive relative threshold so we keep k=4 (full rank) here;
    # the identity should still hold for any k produced.
    measurements = [Observable(Z(), 0)]
    sim_params = AnalogSimParams(
        measurements,
        elapsed_time=0.2,
        dt=0.1,
        sample_timesteps=True,
        num_traj=1,
        max_bond_dim=10,
        threshold=0.05,
        order=1,
        show_progress=False,
        min_bond_dim=1,
    )
    sim_params.trunc_mode = "relative"

    A0, A1 = split_mps_tensor(A_in, distr, sim_params, [d0, d1], dynamic=True)
    k = A0.shape[2]

    L = A0.reshape(d0 * D0, k)
    R = A1.transpose(1, 0, 2).reshape(k, d1 * D2)
    theta_recon = L @ R

    # Compare with best rank-k SVD approximation of the original theta
    u, s, v = np.linalg.svd(theta, full_matrices=False)
    theta_opt_k = (u[:, :k] * s[:k]) @ v[:k, :]
    np.testing.assert_allclose(theta_recon, theta_opt_k, atol=1e-10, rtol=1e-8)

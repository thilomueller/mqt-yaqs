import numpy as np
from scipy.linalg import eigh_tridiagonal, expm

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from yaqs.data_structures.MPS import MPS
    from yaqs.data_structures.MPO import MPO

# from .mps import MPS, merge_mps_tensor_pair, split_mps_tensor
# from .mpo import MPO, merge_mpo_tensor_pair
# from .operation import (
#         contraction_operator_step_right,
#         contraction_operator_step_left,
#         compute_right_operator_blocks,
#         apply_local_hamiltonian,
#         apply_local_bond_contraction)
# from .krylov import expm_krylov
# from .qnumber import qnumber_flatten, is_qsparse
# from .bond_ops import qr

__all__ = ['integrate_local_singlesite', 'integrate_local_twosite']


def single_site_TDVP(H: 'MPO', state: 'MPS', dt, numsteps: int, numiter_lanczos: int = 25):
    """
    Symmetric single-site TDVP integration.
    `psi` is overwritten in-place with the time-evolved state.

    Args:
        H: Hamiltonian as MPO
        psi: initial state as MPS
        dt: time step; for real-time evolution, use purely imaginary dt
        numsteps: number of time steps
        numiter_lanczos: number of Lanczos iterations for each site-local step

    Returns:
        float: norm of initial psi

    Reference:
        J. Haegeman, C. Lubich, I. Oseledets, B. Vandereycken, F. Verstraete
        Unifying time evolution and optimization with matrix product states
        Phys. Rev. B 94, 165116 (2016) (arXiv:1408.5056)
    """

    # number of sites
    L = H.length
    assert L == state.length

    # right-normalize input matrix product state
    state.normalize()

    # left and right operator blocks
    # initialize leftmost block by 1x1x1 identity
    BR = _compute_right_operator_blocks(state, H)
    BL = [None for _ in range(L)]
    BL[0] = np.array([[[1]]], dtype=BR[0].dtype)

    # # consistency check
    # for i in range(len(BR)):
    #     assert is_qsparse(BR[i], [psi.qD[i+1], H.qD[i+1], -psi.qD[i+1]]), \
    #         'sparsity pattern of operator blocks must match quantum numbers'

    for n in range(numsteps):
        # sweep from left to right
        for i in range(L - 1):
            # evolve psi.A[i] forward in time by half a time step
            state.tensors[i] = _local_hamiltonian_step(BL[i], BR[i], H.tensors[i], state.tensors[i], 0.5*dt, numiter_lanczos)
            # left-orthonormalize current psi.A[i]
            s = state.tensors[i].shape
            Q, C = np.linalg.qr(state.tensors[i].reshape((s[0]*s[1], s[2])))

            # (Q, C, psi.qD[i+1]) = qr(psi.A[i].reshape((s[0]*s[1], s[2])),
            #                          qnumber_flatten([psi.qd, psi.qD[i]]), psi.qD[i+1])
            state.tensors[i] = Q.reshape((s[0], s[1], Q.shape[1]))

            # update the left blocks
            BL[i+1] = _contraction_operator_step_left(state.tensors[i], state.tensors[i], H.tensors[i], BL[i])

            # evolve C backward in time by half a time step
            C = _local_bond_step(BL[i+1], BR[i], C, -0.5*dt, numiter_lanczos)
            # update psi.A[i+1] tensor: multiply with C from left
            state.tensors[i+1] = np.einsum(state.tensors[i+1], (0, 3, 2), C, (1, 3), (0, 1, 2), optimize=True)

        # evolve psi.A[L-1] forward in time by a full time step
        i = L - 1
        state.tensors[i] = _local_hamiltonian_step(BL[i], BR[i], H.tensors[i], state.tensors[i], dt, numiter_lanczos)

        # sweep from right to left
        for i in reversed(range(1, L)):
            # right-orthonormalize current psi.A[i]
            # flip left and right virtual bond dimensions
            state.tensors[i] = state.tensors[i].transpose((0, 2, 1))
            # perform QR decomposition
            s = state.tensors[i].shape
            Q, C = np.linalg.qr(state.tensors[i].reshape((s[0]*s[1], s[2])))
            # (Q, C, qbond) = qr(psi.A[i].reshape((s[0]*s[1], s[2])),
            #                    qnumber_flatten([psi.qd, -psi.qD[i+1]]), -psi.qD[i])
            # psi.qD[i] = -qbond
            # replace psi.A[i] by reshaped Q matrix and undo flip of left and right virtual bond dimensions
            state.tensors[i] = Q.reshape((s[0], s[1], Q.shape[1])).transpose((0, 2, 1))
            # update the right blocks
            BR[i-1] = _contraction_operator_step_right(state.tensors[i], state.tensors[i], H.tensors[i], BR[i])
            # evolve C backward in time by half a time step
            C = np.transpose(C)
            C = _local_bond_step(BL[i], BR[i-1], C, -0.5*dt, numiter_lanczos)
            # update psi.A[i-1] tensor: multiply with C from right
            state.tensors[i-1] = np.einsum(state.tensors[i-1], (0, 1, 3), C, (3, 2), (0, 1, 2), optimize=True)
            # evolve psi.A[i-1] forward in time by half a time step
            state.tensors[i-1] = _local_hamiltonian_step(BL[i-1], BR[i-1], H.tensors[i-1], state.tensors[i-1], 0.5*dt, numiter_lanczos)


def _contraction_operator_step_right(A: np.ndarray, B: np.ndarray, W: np.ndarray, R: np.ndarray):
    r"""
    Contraction step from right to left, with a matrix product operator
    sandwiched in between.

    To-be contracted tensor network::

          _____           ______
         /     \         /
      ---|1 B*2|---   ---|2
         \__0__/         |
            |            |
                         |
          __|__          |
         /  0  \         |
      ---|2 W 3|---   ---|1   R
         \__1__/         |
            |            |
                         |
          __|__          |
         /  0  \         |
      ---|1 A 2|---   ---|0
         \_____/         \______
    """
    assert A.ndim == 3
    assert B.ndim == 3
    assert W.ndim == 4
    assert R.ndim == 3
    # multiply with A tensor
    T = np.tensordot(A, R, 1)
    # multiply with W tensor
    T = np.tensordot(W, T, axes=((1, 3), (0, 2)))
    # interchange levels 0 <-> 2 in T
    T = T.transpose((2, 1, 0, 3))
    # multiply with conjugated B tensor
    Rnext = np.tensordot(T, B.conj(), axes=((2, 3), (0, 2)))
    return Rnext


def _contraction_operator_step_left(A: np.ndarray, B: np.ndarray, W: np.ndarray, L: np.ndarray):
    r"""
    Contraction step from left to right, with a matrix product operator
    sandwiched in between.

    To-be contracted tensor network::

     ______           _____
           \         /     \
          2|---   ---|1 B*2|---
           |         \__0__/
           |            |
           |
           |          __|__
           |         /  0  \
      L   1|---   ---|2 W 3|---
           |         \__1__/
           |            |
           |
           |          __|__
           |         /  0  \
          0|---   ---|1 A 2|---
     ______/         \_____/
    """
    assert A.ndim == 3
    assert B.ndim == 3
    assert W.ndim == 4
    assert L.ndim == 3
    # multiply with conjugated B tensor
    T = np.tensordot(L, B.conj(), axes=(2, 1))
    # multiply with W tensor
    T = np.tensordot(W, T, axes=((0, 2), (2, 1)))
    # multiply with A tensor
    Lnext = np.tensordot(A, T, axes=((0, 1), (0, 2)))
    return Lnext


def _compute_right_operator_blocks(state: 'MPS', H: 'MPO'):
    """
    Compute all partial contractions from the right.
    """
    L = state.length
    assert L == H.length

    BR = [None for _ in range(L)]

    # initialize rightmost dummy block
    BR[L-1] = np.array([[[1]]], dtype=complex)

    for i in reversed(range(L-1)):
        BR[i] = _contraction_operator_step_right(state.tensors[i+1], state.tensors[i+1], H.tensors[i+1], BR[i+1])
    return BR


def _lanczos_iteration(Afunc, vstart, numiter):
    """
    Perform a "matrix free" Lanczos iteration.

    Args:
        Afunc:      "matrix free" linear transformation of a given vector
        vstart:     starting vector for iteration
        numiter:    number of iterations (should be much smaller than dimension of vstart)

    Returns:
        tuple: tuple containing
          - alpha:      diagonal real entries of Hessenberg matrix
          - beta:       off-diagonal real entries of Hessenberg matrix
          - V:          `len(vstart) x numiter` matrix containing the orthonormal Lanczos vectors
    """
    # normalize starting vector
    nrmv = np.linalg.norm(vstart)
    assert nrmv > 0
    vstart = vstart / nrmv

    alpha = np.zeros(numiter)
    beta  = np.zeros(numiter-1)

    V = np.zeros((numiter, len(vstart)), dtype=complex)
    V[0] = vstart

    for j in range(numiter-1):
        w = Afunc(V[j])
        alpha[j] = np.vdot(w, V[j]).real
        w -= alpha[j]*V[j] + (beta[j-1]*V[j-1] if j > 0 else 0)
        beta[j] = np.linalg.norm(w)
        if beta[j] < 100*len(vstart)*np.finfo(float).eps:
            # warnings.warn(
            #     f'beta[{j}] ~= 0 encountered during Lanczos iteration.',
            #     RuntimeWarning)
            # premature end of iteration
            numiter = j + 1
            return (alpha[:numiter], beta[:numiter-1], V[:numiter, :].T)
        V[j+1] = w / beta[j]

    # complete final iteration
    j = numiter-1
    w = Afunc(V[j])
    alpha[j] = np.vdot(w, V[j]).real
    return (alpha, beta, V.T)


def _arnoldi_iteration(Afunc, vstart, numiter):
    """
    Perform a "matrix free" Arnoldi iteration.

    Args:
        Afunc:      "matrix free" linear transformation of a given vector
        vstart:     starting vector for iteration
        numiter:    number of iterations (should be much smaller than dimension of vstart)

    Returns:
        tuple: tuple containing
          - H:      `numiter x numiter` upper Hessenberg matrix
          - V:      `len(vstart) x numiter` matrix containing the orthonormal Arnoldi vectors
    """
    # normalize starting vector
    nrmv = np.linalg.norm(vstart)
    assert nrmv > 0
    vstart = vstart / nrmv

    H = np.zeros((numiter, numiter), dtype=complex)
    V = np.zeros((numiter, len(vstart)), dtype=complex)
    V[0] = vstart

    for j in range(numiter-1):
        w = Afunc(V[j])
        # subtract the projections on previous vectors
        for k in range(j+1):
            H[k, j] = np.vdot(V[k], w)
            w -= H[k, j]*V[k]
        H[j+1, j] = np.linalg.norm(w)
        # if H[j+1, j] < 100*len(vstart)*np.finfo(float).eps:
        #     warnings.warn(
        #         f'H[{j+1}, {j}] ~= 0 encountered during Arnoldi iteration.',
        #         RuntimeWarning)
        #     # premature end of iteration
        #     numiter = j + 1
        #     return H[:numiter, :numiter], V[:numiter, :].T
        V[j+1] = w / H[j+1, j]

    # complete final iteration
    j = numiter-1
    w = Afunc(V[j])
    for k in range(j+1):
        H[k, j] = np.vdot(V[k], w)
        w -= H[k, j]*V[k]

    return H, V.T

def _expm_krylov(Afunc, v, dt, numiter, hermitian=False):
    """
    Compute Krylov subspace approximation of the matrix exponential
    applied to input vector: `expm(dt*A)*v`.

    Reference:
        M. Hochbruck and C. Lubich
        On Krylov subspace approximations to the matrix exponential operator
        SIAM J. Numer. Anal. 34, 1911 (1997)
    """
    if hermitian:
        alpha, beta, V = _lanczos_iteration(Afunc, v, numiter)
        # diagonalize Hessenberg matrix
        w_hess, u_hess = eigh_tridiagonal(alpha, beta)
        return V @ (u_hess @ (np.linalg.norm(v) * np.exp(-1j*dt*w_hess) * u_hess[0]))
    else:
        H, V = _arnoldi_iteration(Afunc, v, numiter)
        return V @ (np.linalg.norm(v) * expm(-1j*dt*H)[:, 0])


def _apply_local_hamiltonian(L: np.ndarray, R: np.ndarray, W: np.ndarray, A: np.ndarray):
    r"""
    Apply a local Hamiltonian operator.

    To-be contracted tensor network (the indices at the open legs
    show the ordering for the output tensor)::

     ______                           ______
           \                         /
          2|---1                 2---|2
           |                         |
           |                         |
           |            0            |
           |          __|__          |
           |         /  0  \         |
      L   1|---   ---|2 W 3|---   ---|1   R
           |         \__1__/         |
           |            |            |
           |                         |
           |          __|__          |
           |         /  0  \         |
          0|---   ---|1 A 2|---   ---|0
     ______/         \_____/         \______
    """
    assert L.ndim == 3
    assert R.ndim == 3
    assert W.ndim == 4
    assert A.ndim == 3
    # multiply A with R tensor and store result in T
    T = np.tensordot(A, R, 1)
    # multiply T with W tensor
    T = np.tensordot(W, T, axes=((1, 3), (0, 2)))
    # multiply T with L tensor
    T = np.tensordot(T, L, axes=((2, 1), (0, 1)))
    # interchange levels 1 <-> 2 in T
    T = T.transpose((0, 2, 1))
    return T


def _apply_local_bond_contraction(L, R, C):
    r"""
    Apply "zero-site" bond contraction.

    To-be contracted tensor network::

     ______                           ______
           \                         /
          2|---                   ---|2
           |                         |
           |                         |
           |                         |
           |                         |
           |                         |
      L   1|-----------   -----------|1   R
           |                         |
           |                         |
           |                         |
           |          _____          |
           |         /     \         |
          0|---   ---|0 C 1|---   ---|0
     ______/         \_____/         \______
    """
    assert L.ndim == 3
    assert R.ndim == 3
    assert C.ndim == 2
    # multiply C with R tensor and store result in T
    T = np.tensordot(C, R, 1)
    # multiply L with T tensor
    T = np.tensordot(L, T, axes=((0, 1), (0, 1)))
    return T


def _local_hamiltonian_step(L, R, W, A, dt, numiter: int):
    """
    Local time step effected by Hamiltonian, based on a Lanczos iteration.
    """
    return _expm_krylov(
        lambda x: _apply_local_hamiltonian(L, R, W, x.reshape(A.shape)).reshape(-1),
            A.reshape(-1), -dt, numiter, hermitian=True).reshape(A.shape)


def _local_bond_step(L, R, C, dt, numiter: int):
    """
    Local "zero-site" bond step, based on a Lanczos iteration.
    """
    return _expm_krylov(
        lambda x: _apply_local_bond_contraction(L, R, x.reshape(C.shape)).reshape(-1),
            C.reshape(-1), -dt, numiter, hermitian=True).reshape(C.shape)
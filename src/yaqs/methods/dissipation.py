import numpy as np
import opt_einsum as oe
from scipy.linalg import expm

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from yaqs.data_structures.MPS import MPS
    from yaqs.data_structures.Noise_Model import NoiseModel


# TODO: Assumes noise is same at all sites
#       Could be sped-up by pre-calculating exponential somewhere else
#       Likely not a problem since it's only exponentiating small matrices
def apply_dissipation(state: 'MPS', noise_model: 'NoiseModel', dt: float):
    A = np.zeros(noise_model.jump_operators[0].shape)
    for i, jump_operator in enumerate(noise_model.jump_operators):
        A += noise_model.strengths[i]*np.conj(jump_operator).T @ jump_operator
    dissipative_operator = expm(-1/2*dt*A)

    for i, tensor in enumerate(state.tensors):
        state.tensors[i] = oe.contract('ab, bcd->acd', dissipative_operator, tensor)

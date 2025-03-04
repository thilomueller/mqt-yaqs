from __future__ import annotations
import numpy as np
import opt_einsum as oe
from scipy.linalg import expm


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from yaqs.core.data_structures.networks import MPS
    from yaqs.core.data_structures.noise_model import NoiseModel

# TODO: Assumes noise is same at all sites
#       Could be sped-up by pre-calculating exponential somewhere else
#       Likely not a problem since it's only exponentiating small matrices
def apply_dissipation(state: MPS, noise_model: NoiseModel, dt: float):
    """
    Apply dissipation to the system state using the given noise model and time step.

    This function modifies the state tensors by applying a dissipative operator that is calculated
    from the noise model's jump operators and strengths.

    Args:
        state (MPS): The Matrix Product State (MPS) representing the current state of the system.
        noise_model (NoiseModel): The noise model containing jump operators and their corresponding strengths.
        dt (float): The time step for the evolution.

    Returns:
        None
    """
    if noise_model and any(gamma != 0 for gamma in noise_model.strengths):
        # Calculate the dissipation operator A
        A = sum(noise_model.strengths[i] * np.conj(jump_operator).T @ jump_operator
                for i, jump_operator in enumerate(noise_model.jump_operators))
        
        # Compute the dissipative operator by exponentiating the matrix A
        dissipative_operator = expm(-0.5 * dt * A)

    # Apply the dissipative operator to each tensor in the MPS
    for i in reversed(range(state.length)):
        if noise_model and any(gamma != 0 for gamma in noise_model.strengths):
            state.tensors[i] = oe.contract('ab, bcd->acd', dissipative_operator, state.tensors[i])
        # Prepares state for probability calculation, results in mixed canonical form at site 0
        # Shifting it during the sweep is faster than setting it at the end
        if i != 0:
            state.shift_orthogonality_center_left(current_orthogonality_center=i)

import copy
import numpy as np
import opt_einsum as oe

from yaqs.operations.operations import scalar_product

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from yaqs.data_structures.MPS import MPS
    from yaqs.data_structures.noise_model import NoiseModel
    from yaqs.data_structures.simulation_parameters import SimulationParams

# TODO: Can be reduced to a single tensor contraction
def calculate_stochastic_factor(state: 'MPS') -> float:
    return 1 - scalar_product(state, state)

# TODO: Can be reduced to a single tensor contraction
def create_probability_distribution(state: 'MPS', noise_model: 'NoiseModel', dt: float) -> dict:
        # Ordered as [Jump 0 Site 0, Jump 1 Site 0, Jump 0 Site 1, Jump 1 Site 1...]
        jump_dict = {'jumps': [], 'strengths': [], 'sites': [], 'probabilities': []}
        dp_m_list = []
        for i, tensor in enumerate(state.tensors):
            for j, jump_operator in enumerate(noise_model.jump_operators):
                jumped_state = copy.deepcopy(state)
                jumped_state.tensors[i] = oe.contract('ab, bcd->acd', jump_operator, tensor)

                dp_m = dt*noise_model.strengths[j]*scalar_product(jumped_state, jumped_state)
                dp_m_list.append(dp_m.real)
                jump_dict['jumps'].append(jump_operator)
                jump_dict['strengths'].append(noise_model.strengths[j])
                jump_dict['sites'].append(i)
        print(np.sum(dp_m_list))
        jump_dict['probabilities'] = dp_m_list/np.sum(dp_m_list).astype(float)
        return jump_dict


def stochastic_process(previous_state: 'MPS', state: 'MPS', noise_model: 'NoiseModel', dt: float) -> 'MPS':
    dp = calculate_stochastic_factor(state)
    if np.random.rand() >= dp:
        # No jump
        state.normalize()
        return state
    else:
        # Jump
        jump_dict = create_probability_distribution(state, noise_model, dt)
        choices = [*range(len(jump_dict['probabilities']))]
        choice = np.random.choice(choices, p=jump_dict['probabilities'])
        jump_operator = jump_dict['jumps'][choice]
        previous_state.tensors[jump_dict['sites'][choice]] = oe.contract('ab, bcd->acd', jump_operator, previous_state.tensors[jump_dict['sites'][choice]])
        previous_state.normalize(form='B')
        return previous_state

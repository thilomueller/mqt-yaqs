import copy
import numpy as np

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from yaqs.data_structures.MPO import MPO
    from yaqs.data_structures.MPS import MPS
    from yaqs.data_structures.noise_model import NoiseModel
    from yaqs.data_structures.simulation_parameters import SimulationParams

from yaqs.methods.dissipation import apply_dissipation
from yaqs.methods.dynamic_TDVP import dynamic_TDVP
from yaqs.methods.stochastic_process import stochastic_process


# TODO: Paralellized
import concurrent.futures
import multiprocessing


def TJM(initial_state: 'MPS', H: 'MPO', noise_model: 'NoiseModel', sim_params: 'SimulationParams', full_data=False, multi_threaded=True) -> np.ndarray:
    all_trajectories_exp_values = []
    # TODO: Extend to multiple measurements
    observables = list(sim_params.measurements.keys())
    sites = list(sim_params.measurements.values())

    times = np.arange(0, sim_params.T + sim_params.dt, sim_params.dt)

    def run_trajectory(i):
        print(f"Trajectory {i}")

        # Create deep copies of the shared inputs to avoid race conditions
        trajectory_state = copy.deepcopy(initial_state)

        single_trajectory_exp_values = [trajectory_state.measure_observable(observables[0], sites[0])]

        phi = initialize(trajectory_state, noise_model, sim_params)
        single_trajectory_exp_values.append(sample(phi, H, noise_model, sim_params))

        for _ in times[2:]:
            # print(f"Time {t}")
            phi = step_through(phi, H, noise_model, sim_params)
            single_trajectory_exp_values.append(sample(phi, H, noise_model, sim_params))

        return single_trajectory_exp_values

    if multi_threaded:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            all_trajectories_exp_values = list(executor.map(run_trajectory, range(sim_params.N)))
    else:
        for i in range(sim_params.N):
            single_trajectory_exp_values = run_trajectory(i)
            all_trajectories_exp_values.append(single_trajectory_exp_values)

    all_trajectories_exp_values = np.array(all_trajectories_exp_values)

    if full_data:
        return times, all_trajectories_exp_values
    else:
        return times, np.mean(all_trajectories_exp_values, axis=0)

# def TJM(initial_state: 'MPS', H: 'MPO', noise_model: 'NoiseModel', sim_params: 'SimulationParams', full_data=False) -> np.ndarray:
#     all_trajectories_exp_values = []
#     # TODO: Extend to multiple measurements
#     observables = list(sim_params.measurements.keys())
#     sites = list(sim_params.measurements.values())

#     times = np.arange(0, sim_params.T + sim_params.dt, sim_params.dt)
#     for i in range(sim_params.N):
#         print(f"Trajectory {i}")
#         single_trajectory_exp_values = [initial_state.measure_observable(observables[0], sites[0])]

#         phi = initialize(copy.deepcopy(initial_state), noise_model, sim_params)
#         single_trajectory_exp_values.append(sample(phi, H, noise_model, sim_params))

#         for _, t in enumerate(times[2:]):
#             print(f"Time {t}")
#             phi = step_through(phi, H, noise_model, sim_params)
#             single_trajectory_exp_values.append(sample(phi, H, noise_model, sim_params))

#         all_trajectories_exp_values.append(single_trajectory_exp_values)

#     all_trajectories_exp_values = np.array(all_trajectories_exp_values)

#     if full_data:
#         return times, all_trajectories_exp_values
#     else:
#         return times, np.mean(all_trajectories_exp_values, axis=0)


def initialize(state: 'MPS', noise_model: 'NoiseModel', sim_params: 'SimulationParams') -> 'MPS': 
    previous_state = copy.deepcopy(state)
    apply_dissipation(state, noise_model, sim_params.dt/2)
    return stochastic_process(previous_state, state, noise_model, sim_params.dt)


def step_through(state: 'MPS', H: 'MPO', noise_model: 'NoiseModel', sim_params: 'SimulationParams') -> 'MPS': 
    previous_state = copy.deepcopy(state)
    dynamic_TDVP(state, H, sim_params.dt, sim_params.max_bond_dim)
    apply_dissipation(state, noise_model, sim_params.dt)
    return stochastic_process(previous_state, state, noise_model, sim_params.dt)


def sample(phi: 'MPS', H: 'MPO', noise_model: 'NoiseModel', sim_params: 'SimulationParams') -> 'MPS':
    psi = copy.deepcopy(phi)
    dynamic_TDVP(psi, H, sim_params.dt, sim_params.max_bond_dim)
    apply_dissipation(psi, noise_model, sim_params.dt/2)
    psi.normalize('B')
    return psi.measure_observable(list(sim_params.measurements.keys())[0], list(sim_params.measurements.values())[0])

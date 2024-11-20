import matplotlib.pyplot as plt

from yaqs.data_structures.MPO import MPO
from yaqs.data_structures.MPS import MPS
from yaqs.data_structures.noise_model import NoiseModel
from yaqs.data_structures.simulation_parameters import SimulationParams
from yaqs.methods.TJM import TJM


# Define the system Hamiltonian
L = 10
d = 2
J = 1
g = 0.1
H_0 = MPO()
H_0.init_Ising(L, d, J, g)

# Define the initial state
state = MPS(L, state='zeros')

# Define the noise model
noise_model = NoiseModel(['excitation', 'relaxation', 'dephasing'], [0.1, 0.1, 0.1])

# Define the simulation parameters
dt = 0.1
timesteps = 40
max_bond_dim = 8
N = 50
sim_params = SimulationParams({'x': L//2}, timesteps*dt, dt, max_bond_dim, N)

if __name__ == "__main__":
    # Start trajectory execution
    print("Starting TJM Trajectories")
    all_exp_vals = []

    # Run TJM using multiple processes
    times, exp_values = TJM(state, H_0, noise_model, sim_params, full_data=False, multi_core=True)

    plt.plot(times, exp_values, label=f'TJM L={L}')
    plt.xlabel('t')
    plt.ylabel('$\\langle X^{[L/2]} \\rangle$')
    plt.legend()
    plt.show()

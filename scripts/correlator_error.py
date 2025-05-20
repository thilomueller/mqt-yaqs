import pickle

from trotter.correlator_utils import generate_heisenberg_error_data, generate_periodic_heisenberg_error_data, generate_2d_ising_error_data


def run_correlator_test():
    # General Heisenberg parameters
    J = 1
    h = 1
    dt = 0.1
    num_qubits = 16

    min_bonds = [2, 4, 8, 16]
    timesteps_list = [*range(1, 21)]

    # 1D Heisenberg model
    print("1D Heisenberg")
    results = generate_heisenberg_error_data(num_qubits, J, h, dt, min_bonds, timesteps_list)
    filename = f"results/correlator_error/heisenberg_error.pickle"
    with open(filename, 'wb') as f:
        pickle.dump({
            'results': results,
        }, f)

    # 1D Perioidic Heisenberg model
    print("1D Periodic")
    results = generate_periodic_heisenberg_error_data(num_qubits, J, h, dt, min_bonds, timesteps_list)
    filename = f"results/correlator_error/periodic_heisenberg_error.pickle"
    with open(filename, 'wb') as f:
        pickle.dump({
            'results': results,
        }, f)

    # 2D Ising model
    J = 1
    g = 1
    dt = 0.1
    num_rows = 4
    num_cols = 4

    print("2D Ising")
    results = generate_2d_ising_error_data(num_rows, num_cols, J, g, dt, min_bonds, timesteps_list)
    filename = f"results/correlator_error/2d_ising_error.pickle"
    with open(filename, 'wb') as f:
        pickle.dump({
            'results': results,
        }, f)

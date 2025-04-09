from mqt.yaqs.core.data_structures.networks import MPO
import numpy as np

H_terms = [
    (1.0, ['Z', 'Z', 'I', 'I']),
    (0.5, ['X', 'I', 'X', 'I']),
    (-0.2, ['I', 'Y', 'Y', 'I']),
]

mpo = MPO()
H = mpo.build_full_hamiltonian(H_terms, L=4)
mpo.init_from_sum_op(terms=H_terms, L=4)

error = np.linalg.norm(mpo.to_matrix() - H, 2)
print("Reconstruction error: " + str(error))
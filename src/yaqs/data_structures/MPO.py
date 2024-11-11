import numpy as np

from yaqs.library.tensor_library import TensorLibrary

# Convention (sigma, chi_l, sigma', chi_l+1)
class IsingMPO:
    def __init__(self, length: int, physical_dimension: int, J: int, g: int):
        zero = np.zeros((physical_dimension, physical_dimension))
        identity = np.identity(physical_dimension)
        X = getattr(TensorLibrary, "x")().matrix
        Z = getattr(TensorLibrary, "z")().matrix

        left_bound = np.array([identity, -J*Z, -g*X])
        left_bound = np.expand_dims(left_bound, 0)
        inner = np.array([np.array([identity, -J*Z, -g*X]),
                        np.array([zero, zero, Z]),
                        np.array([zero, zero, identity])])

        right_bound = np.array([[-g*X],
                                [Z],
                                [identity]])

        self.tensors = [left_bound] + [inner]*(length-2) + [right_bound]
        for i, tensor in enumerate(self.tensors):
            # left, right, sigma, sigma'
            self.tensors[i] = np.transpose(tensor, (2, 0, 3, 1))

class customMPO:
    def __init__(self, length:int, left_bound: np.ndarray, inner: np.ndarray, right_bound: np.ndarray):
        self.tensors = [left_bound] + [inner]*(length-2) + [right_bound]
# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tensor Network Data Structures.

This module implements classes for representing quantum states and operators using tensor networks.
It defines the Matrix Product State (MPS) and Matrix Product Operator (MPO) classes, along with various
methods for network normalization, canonicalization, measurement, and validity checks. These classes and
utilities are essential for simulating quantum many-body systems using tensor network techniques.
"""

from __future__ import annotations

import concurrent.futures
import copy
import multiprocessing
from typing import TYPE_CHECKING

import numpy as np
import opt_einsum as oe
from tqdm import tqdm

from ..libraries.observables_library import ObservablesLibrary
from ..methods.decompositions import right_qr, truncated_right_svd

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .simulation_parameters import Observable


class MPS:
    """Matrix Product State (MPS) class for representing quantum states.

    This class forms the basis of the MPS used in YAQS simulations.
    The index order is (sigma, chi_l-1, chi_l).

    Attributes:
    length (int): The number of sites in the MPS.
    tensors (list[NDArray[np.complex128]]): List of rank-3 tensors representing the MPS.
    physical_dimensions (list[int]): List of physical dimensions for each site.
    flipped (bool): Indicates if the network has been flipped.

    Methods:
    __init__(length: int, tensors: list[NDArray[np.complex128]] | None = None,
                physical_dimensions: list[int] | None = None, state: str = "zeros") -> None:
        Initializes the MPS with given length, tensors, physical dimensions, and initial state.
    pad_bond_dimension():
        Pads bond dimension with zeros
    write_max_bond_dim() -> int:
        Returns the maximum bond dimension in the MPS.
    flip_network() -> None:
        Flips the bond dimensions in the network to allow operations from right to left.
    shift_orthogonality_center_right(current_orthogonality_center: int) -> None:
        Left and right normalizes the MPS around a selected site, shifting the orthogonality center to the right.
    shift_orthogonality_center_left(current_orthogonality_center: int) -> None:
        Left and right normalizes the MPS around a selected site, shifting the orthogonality center to the left.
    set_canonical_form(orthogonality_center: int) -> None:
        Left and right normalizes the MPS around a selected site to set it in canonical form.
    normalize(form: str = "B") -> None:
        Normalizes the MPS in the specified form.
    measure(observable: Observable) -> np.float64:
        Measures the expectation value of an observable at a specified site.
    norm(site: int | None = None) -> np.float64:
        Computes the norm of the MPS, optionally at a specified site.
    write_tensor_shapes() -> None:
        Writes the shapes of the tensors in the MPS.
    check_if_valid_mps() -> None:
        Checks if the MPS is valid by verifying bond dimensions.
    check_canonical_form() -> list[int]:
        Checks the canonical form of the MPS and returns the orthogonality center(s).
    """

    def __init__(
        self,
        length: int,
        tensors: list[NDArray[np.complex128]] | None = None,
        physical_dimensions: list[int] | None = None,
        state: str = "zeros",
        pad: int | None = None,
    ) -> None:
        """Initializes a Matrix Product State (MPS).

        Parameters
        ----------
        length : int
            Number of sites (qubits) in the MPS.
        tensors : list[NDArray[np.complex128]], optional
            Predefined tensors representing the MPS. Must match `length` if provided.
            If None, tensors are initialized according to `state`.
        physical_dimensions : list[int], optional
            Physical dimension for each site. Defaults to qubit systems (dimension 2) if None.
        state : str, optional
            Initial state configuration. Valid options include:
            - "zeros": Initializes all qubits to |0⟩.
            - "ones": Initializes all qubits to |1⟩.
            - "x+": Initializes each qubit to (|0⟩ + |1⟩)/√2.
            - "x-": Initializes each qubit to (|0⟩ - |1⟩)/√2.
            - "y+": Initializes each qubit to (|0⟩ + i|1⟩)/√2.
            - "y-": Initializes each qubit to (|0⟩ - i|1⟩)/√2.
            - "Neel": Alternating pattern |0101...⟩.
            - "wall": Domain wall at given site |000111>
            - "random": Initializes each qubit randomly.
            Default is "zeros".

        Raises:
        ------
        AssertionError
            If `tensors` is provided and its length does not match `length`.
        ValueError
            If the provided `state` parameter does not match any valid initialization string.
        """  # noqa: DOC501
        self.flipped = False
        if tensors is not None:
            assert len(tensors) == length
            self.tensors = tensors
        else:
            self.tensors = []
        self.length = length
        if physical_dimensions is None:
            # Default case is the qubit (2-level) case
            self.physical_dimensions = []
            for _ in range(self.length):
                self.physical_dimensions.append(2)
        else:
            self.physical_dimensions = physical_dimensions
        assert len(self.physical_dimensions) == length

        # Create d-level |0> state
        if not tensors:
            for i, d in enumerate(self.physical_dimensions):
                vector = np.zeros(d, dtype=complex)
                if state == "zeros":
                    # |0>
                    vector[0] = 1
                elif state == "ones":
                    # |1>
                    vector[1] = 1
                elif state == "x+":
                    # |+> = (|0> + |1>)/sqrt(2)
                    vector[0] = 1 / np.sqrt(2)
                    vector[1] = 1 / np.sqrt(2)
                elif state == "x-":
                    # |-> = (|0> - |1>)/sqrt(2)
                    vector[0] = 1 / np.sqrt(2)
                    vector[1] = -1 / np.sqrt(2)
                elif state == "y+":
                    # |+i> = (|0> + i|1>)/sqrt(2)
                    vector[0] = 1 / np.sqrt(2)
                    vector[1] = 1j / np.sqrt(2)
                elif state == "y-":
                    # |-i> = (|0> - i|1>)/sqrt(2)
                    vector[0] = 1 / np.sqrt(2)
                    vector[1] = -1j / np.sqrt(2)
                elif state == "Neel":
                    # |010101...>
                    if i % 2:
                        vector[0] = 1
                    else:
                        vector[1] = 1
                elif state == "wall":
                    # |000111>
                    if i < length // 2:
                        vector[0] = 1
                    else:
                        vector[1] = 1
                elif state == "random":
                    rng = np.random.default_rng()
                    vector[0] = rng.random()
                    vector[1] = 1 - vector[0]
                else:
                    msg = "Invalid state string"
                    raise ValueError(msg)

                tensor = np.expand_dims(vector, axis=(0, 1))

                tensor = np.transpose(tensor, (2, 0, 1))
                self.tensors.append(tensor)

            if state == "random":
                self.normalize()
        if pad is not None:
            self.pad_bond_dimension(pad)

    def pad_bond_dimension(self, target_dim: int) -> None:
        """Pad bond dimension.

        Pads the bond dimensions of each tensor in the MPS so that the internal bond
        dimensions are at least target_dim. For the first tensor the left bond dimension
        remains 1, and for the last tensor the right bond dimension remains 1.

        Parameters
        ----------
        target_dim : int
            The desired bond dimension for the internal bonds.

        Raises:
        ------
        ValueError: If target_dim is smaller than any existing bond dimension.
        """
        for i, tensor in enumerate(self.tensors):
            # Tensor shape is (physical_dim, chi_left, chi_right)
            phys, chi_left, chi_right = tensor.shape

            # Determine desired bond dimensions
            new_left = chi_left if i == 0 else target_dim
            new_right = chi_right if i == self.length - 1 else target_dim

            # Check if the target dimensions are valid
            if chi_left > new_left or chi_right > new_right:
                msg = "Target bond dim must be at least as large as the current bond dim."
                raise ValueError(msg)

            # Create a new tensor with zeros and copy the original data into the appropriate block
            new_tensor = np.zeros((phys, new_left, new_right), dtype=tensor.dtype)
            new_tensor[:, :chi_left, :chi_right] = tensor

            self.tensors[i] = new_tensor

    def write_max_bond_dim(self) -> int:
        """Write max bond dim.

        Calculate and return the maximum bond dimension of the tensors in the network.
        This method iterates over all tensors in the network and determines the maximum
        bond dimension by comparing the first and third dimensions of each tensor's shape.
        The global maximum bond dimension is then returned.

        Returns:
            int: The maximum bond dimension found among all tensors in the network.
        """
        global_max = 0
        for tensor in self.tensors:
            local_max = max(tensor.shape[0], tensor.shape[2])
            global_max = max(global_max, local_max)

        return global_max

    def flip_network(self) -> None:
        """Flip MPS.

        Flips the bond dimensions in the network so that we can do operations
        from right to left rather than coding it twice.

        """
        new_tensors = []
        for tensor in self.tensors:
            new_tensor = np.transpose(tensor, (0, 2, 1))
            new_tensors.append(new_tensor)

        new_tensors.reverse()
        self.tensors = new_tensors
        self.flipped = not self.flipped

    def almost_equal(self, other: MPS) -> bool:
        """Checks if the tensors of this MPS are almost equal to the other MPS.

        Args:
            other (MPS): The other MPS to compare with.

        Returns:
            bool: True if all tensors of this tensor are almost equal to the
                other MPS, False otherwise.
        """
        if self.length != other.length:
            return False
        for i in range(self.length):
            if self.tensors[i].shape != other.tensors[i].shape:
                return False
            if not np.allclose(self.tensors[i], other.tensors[i]):
                return False
        return True

    def shift_orthogonality_center_right(self, current_orthogonality_center: int, decomposition: str = "QR") -> None:
        """Shifts orthogonality center right.

        This function performs a QR decomposition to shift the known current center to the right and move
        the canonical form. This is essential for maintaining efficient tensor network algorithms.

        Args:
            current_orthogonality_center (int): current center
            decomposition: Decides between QR or SVD decomposition. QR is faster, SVD allows bond dimension to reduce
                           Default is QR.
        """
        tensor = self.tensors[current_orthogonality_center]
        if decomposition == "QR":
            site_tensor, bond_tensor = right_qr(tensor)
        elif decomposition == "SVD":
            site_tensor, s_vec, v_mat = truncated_right_svd(tensor, threshold=1e-12, max_bond_dim=None)
            bond_tensor = np.diag(s_vec) @ v_mat
        self.tensors[current_orthogonality_center] = site_tensor

        # If normalizing, we just throw away the R
        if current_orthogonality_center + 1 < self.length:
            self.tensors[current_orthogonality_center + 1] = oe.contract(
                "ij, ajc->aic", bond_tensor, self.tensors[current_orthogonality_center + 1]
            )

    def shift_orthogonality_center_left(self, current_orthogonality_center: int) -> None:
        """Shifts orthogonality center left.

        This function flips the network, performs a right shift, then flips the network again.

        Args:
            current_orthogonality_center (int): current center
        """
        self.flip_network()
        self.shift_orthogonality_center_right(self.length - current_orthogonality_center - 1)
        self.flip_network()

    def set_canonical_form(self, orthogonality_center: int, decomposition: str = "QR") -> None:
        """Sets canonical form of MPS.

        Left and right normalizes an MPS around a selected site.
        NOTE: Slow method compared to shifting based on known form and should be avoided.

        Args:
            orthogonality_center (int): site of matrix MPS around which we normalize
            decomposition: Type of decomposition. Default QR.
        """

        def sweep_decomposition(orthogonality_center: int, decomposition: str = "QR") -> None:
            for site, _ in enumerate(self.tensors):
                if site == orthogonality_center:
                    break
                self.shift_orthogonality_center_right(site, decomposition)

        sweep_decomposition(orthogonality_center, decomposition)
        self.flip_network()
        flipped_orthogonality_center = self.length - 1 - orthogonality_center
        sweep_decomposition(flipped_orthogonality_center, decomposition)
        self.flip_network()

    def normalize(self, form: str = "B", decomposition: str = "QR") -> None:
        """Normalize MPS.

        Normalize the network to a specified form.
        This method normalizes the network to the specified form. By default, it normalizes
        to form "B" (right canonical).
        The normalization process involves flipping the network, setting the canonical form with the
        orthogonality center at the last position, and shifting the orthogonality center to the rightmost position.

        NOTE: Slow method compared to shifting based on known form and should be avoided.

        Args:
            form (str): The form to normalize the network to. Default is "B".
            decomposition: Decides between QR or SVD decomposition. QR is faster, SVD allows bond dimension to reduce
                           Default is QR.
        """
        if form == "B":
            self.flip_network()

        self.set_canonical_form(orthogonality_center=self.length - 1, decomposition=decomposition)
        self.shift_orthogonality_center_right(self.length - 1)

        if form == "B":
            self.flip_network()

    def truncate(self, threshold: float = 1e-12, max_bond_dim: int | None = None) -> None:
        """Truncates the MPS in place.

        Args:
            state: The MPS.
            sim_params: The truncation parameters.
            threshold: The truncation threshold. Default
            max_bond_dim: The maximum bond dimension allowed. Default None.

        """
        if self.length != 1:
            for i, tensor in enumerate(self.tensors[:-1]):
                _, _, v_mat = truncated_right_svd(tensor, threshold, max_bond_dim)
                # Pull v into left leg of next tensor.
                new_next = np.tensordot(v_mat, self.tensors[i + 1], axes=(1, 1))
                new_next = new_next.transpose(1, 0, 2)
                self.tensors[i + 1] = new_next
                # Pull v^dag into current tensor.
                self.tensors[i] = np.tensordot(
                    self.tensors[i],
                    v_mat.conj(),  # No transpose, put correct axes instead
                    axes=(2, 1),
                )

    def scalar_product(self, other: MPS, site: int | None = None) -> np.complex128:
        """Compute the scalar (inner) product between two Matrix Product States (MPS).

        The function contracts the corresponding tensors of two MPS objects. If no specific site is
        provided, the contraction is performed sequentially over all sites to yield the overall inner
        product. When a site is specified, only the tensors at that site are contracted.

        Args:
            other (MPS): The second Matrix Product State.
            site (int | None): Optional site index at which to compute the contraction. If None, the
                contraction is performed over all sites.

        Returns:
            np.complex128: The resulting scalar product as a complex number.
        """
        a_copy = copy.deepcopy(self)
        b_copy = copy.deepcopy(other)
        for i, tensor in enumerate(a_copy.tensors):
            a_copy.tensors[i] = np.conj(tensor)

        result = np.array(np.inf)
        if site is None:
            for idx in range(self.length):
                tensor = oe.contract("abc, ade->bdce", a_copy.tensors[idx], b_copy.tensors[idx])
                result = tensor if idx == 0 else oe.contract("abcd, cdef->abef", result, tensor)
        else:
            result = oe.contract("ijk, ijk", a_copy.tensors[site], b_copy.tensors[site])
        return np.complex128(np.squeeze(result))

    def local_expval(self, operator: NDArray[np.complex128], site: int) -> np.complex128:
        """Compute the local expectation value of an operator on an MPS.

        The function applies the given operator to the tensor at the specified site of a deep copy of the
        input MPS, then computes the scalar product between the original and the modified state at that site.
        This effectively calculates the expectation value of the operator at the specified site.

        Args:
            operator (NDArray[np.complex128]): The local operator (matrix) to be applied.
            site (int): The index of the site at which to evaluate the expectation value.

        Returns:
            np.complex128: The computed expectation value (typically, its real part is of interest).

        Notes:
            A deep copy of the state is used to prevent modifications to the original MPS.
        """
        temp_state = copy.deepcopy(self)
        temp_state.tensors[site] = oe.contract("ab, bcd->acd", operator, temp_state.tensors[site])
        return self.scalar_product(temp_state, site)

    def measure_expectation_value(self, observable: Observable) -> np.float64:
        """Measurement of expectation value.

        Measure the expectation value of a given observable.

        Parameters:
        observable (Observable): The observable to measure. It must have a 'site' attribute indicating
        the site to measure and a 'name' attribute corresponding to a gate in the GateLibrary.

        Returns:
        np.float64: The real part of the expectation value of the observable.
        """
        assert observable.site in range(self.length), "State is shorter than selected site for expectation value."
        # Copying done to stop the state from messing up its own canonical form
        exp = self.local_expval(ObservablesLibrary[observable.name], observable.site)
        assert exp.imag < 1e-13, f"Measurement should be real, '{exp.real:16f}+{exp.imag:16f}i'."
        return exp.real

    def measure_single_shot(self) -> int:
        """Perform a single-shot measurement on a Matrix Product State (MPS).

        This function simulates a projective measurement on an MPS. For each site, it computes the
        local reduced density matrix from the site's tensor, derives the probability distribution over
        basis states, and randomly selects an outcome. The overall measurement result is encoded as an
        integer corresponding to the measured bitstring.

        Returns:
            int: The measurement outcome represented as an integer.
        """
        temp_state = copy.deepcopy(self)
        bitstring = []
        for site, tensor in enumerate(temp_state.tensors):
            reduced_density_matrix = oe.contract("abc, dbc->ad", tensor, np.conj(tensor))
            probabilities = np.diag(reduced_density_matrix).real
            rng = np.random.default_rng()
            chosen_index = rng.choice(len(probabilities), p=probabilities)
            bitstring.append(chosen_index)
            selected_state = np.zeros(len(probabilities))
            selected_state[chosen_index] = 1
            # Multiply state: project the tensor onto the selected state.
            projected_tensor = oe.contract("a, acd->cd", selected_state, tensor)
            # Propagate the measurement to the next site.
            if site != self.length - 1:
                temp_state.tensors[site + 1] = (  # noqa: B909
                    1
                    / np.sqrt(probabilities[chosen_index])
                    * oe.contract("ab, cbd->cad", projected_tensor, temp_state.tensors[site + 1])
                )
        return sum(c << i for i, c in enumerate(bitstring))

    def measure_shots(self, shots: int) -> dict[int, int]:
        """Perform multiple single-shot measurements on an MPS and aggregate the results.

        This function executes a specified number of measurement shots on the given MPS. For each shot,
        a single-shot measurement is performed, and the outcomes are aggregated into a histogram (dictionary)
        mapping basis states (represented as integers) to the number of times they were observed.

        Args:
            state (MPS): The Matrix Product State to be measured.
            shots (int): The number of measurement shots to perform.

        Returns:
            dict[int, int]: A dictionary where keys are measured basis states (as integers) and values are
            the corresponding counts.

        Notes:
            - When more than one shot is requested, measurements are parallelized using a ProcessPoolExecutor.
            - A progress bar (via tqdm) displays the progress of the measurement process.
        """
        results: dict[int, int] = {}
        if shots > 1:
            max_workers = max(1, multiprocessing.cpu_count() - 1)
            with (
                concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor,
                tqdm(total=shots, desc="Measuring shots", ncols=80) as pbar,
            ):
                futures = [executor.submit(self.measure_single_shot) for _ in range(shots)]
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    results[result] = results.get(result, 0) + 1
                    pbar.update(1)
            return results
        basis_state = self.measure_single_shot()
        results[basis_state] = results.get(basis_state, 0) + 1
        return results

    def norm(self, site: int | None = None) -> np.float64:
        """Norm calculation.

        Calculate the norm of the state.

        Parameters:
        site (int | None): The specific site to calculate the norm from. If None, the norm is calculated for
                           the entire network.

        Returns:
        np.float64: The norm of the state or the specified site.
        """
        if site is not None:
            return self.scalar_product(self, site).real
        return self.scalar_product(self).real

    def check_if_valid_mps(self) -> None:
        """MPS validity check.

        Check if the current tensor network is a valid Matrix Product State (MPS).

        This method verifies that the bond dimensions between consecutive tensors
        in the network are consistent. Specifically, it checks that the second
        dimension of each tensor matches the third dimension of the previous tensor.
        """
        right_bond = self.tensors[0].shape[2]
        for tensor in self.tensors[1::]:
            assert tensor.shape[1] == right_bond
            right_bond = tensor.shape[2]

    def check_canonical_form(self) -> list[int]:
        """Checks canonical form of MPS.

        Checks what canonical form a Matrix Product State (MPS) is in, if any.
        This method verifies if the MPS is in left-canonical form, right-canonical form, or mixed-canonical form.
        It returns a list indicating the canonical form status:
        - [0] if the MPS is in left-canonical form.
        - [self.length - 1] if the MPS is in right-canonical form.
        - [index] if the MPS is in mixed-canonical form, where `index` is the position where the form changes.
        - [-1] if the MPS is not in any canonical form.

        Returns:
            list[int]: A list indicating the canonical form status of the MPS.
        """
        a = copy.deepcopy(self.tensors)
        for i, tensor in enumerate(self.tensors):
            a[i] = np.conj(tensor)
        b = self.tensors

        a_truth = []
        b_truth = []
        epsilon = 1e-12
        for i in range(len(a)):
            mat = oe.contract("ijk, ijl->kl", a[i], b[i])
            mat[epsilon > mat] = 0
            test_identity = np.eye(mat.shape[0], dtype=complex)
            a_truth.append(np.allclose(mat, test_identity))

        for i in range(len(a)):
            mat = oe.contract("ijk, ilk->jl", b[i], a[i])
            mat[epsilon > mat] = 0
            test_identity = np.eye(mat.shape[0], dtype=complex)
            b_truth.append(np.allclose(mat, test_identity))

        if all(b_truth):
            return [0]
        if all(a_truth):
            return [self.length - 1]

        if not (all(a_truth) and all(b_truth)):
            sites = []
            for i, _truth_value in enumerate(a_truth):
                if _truth_value:
                    sites.append(i)
                else:
                    break

            for i, _truth_value in enumerate(b_truth[len(sites) :], start=len(sites)):
                sites.append(i)
            if False in sites:
                return [sites.index(False)]
            for i, value in enumerate(a_truth):
                if not value:
                    return [i - 1, i]
        return [-1]

    def to_vec(self) -> NDArray[np.complex128]:
        r"""Converts the MPS to a full state vector representation.

        Returns:
                A one-dimensional NumPy array of length \(\prod_{\ell=1}^L d_\ell\)
                representing the state vector.
        """
        # Start with the first tensor.
        # Assume each tensor has shape (d, chi_left, chi_right) with chi_left=1 for the first tensor.
        self.flip_network()
        vec = self.tensors[0]  # shape: (d_1, 1, chi_1)

        # Contract sequentially with the remaining tensors.
        for i in range(1, self.length):
            # Contract the last bond of vec with the middle index (left bond) of the next tensor.
            vec = np.tensordot(vec, self.tensors[i], axes=([-1], [1]))
            # After tensordot, if vec had shape (..., chi_i) and the new tensor has shape (d_{i+1}, chi_i, chi_{i+1}),
            # then vec now has shape (..., d_{i+1}, chi_{i+1}).
            # Reshape to merge all physical indices into one index.
            new_shape = (-1, vec.shape[-1])
            vec = np.reshape(vec, new_shape)
        self.flip_network()
        # At the end, the final bond dimension should be 1.
        vec = np.squeeze(vec, axis=-1)
        # Flatten the resulting multi-index into a one-dimensional state vector.
        return vec.flatten()


class MPO:
    """Class representing a Matrix Product Operator (MPO) for quantum many-body systems.

    This class forms the basis of the MPS used in YAQS simulations.
    The index order is (sigma, sigma', chi_l-1, chi_l).

    Methods.
    -------
    init_ising(length: int, J: float, g: float) -> None
        Initializes the MPO for the Ising model with given parameters.
    init_heisenberg(length: int, Jx: float, Jy: float, Jz: float, h: float) -> None
        Initializes the MPO for the Heisenberg model with given parameters.
    init_identity(length: int, physical_dimension: int = 2) -> None
        Initializes the MPO as an identity operator.
    init_custom_hamiltonian(length: int, left_bound: NDArray[np.complex128],
                            inner: NDArray[np.complex128], right_bound: NDArray[np.complex128]) -> None
        Initializes the MPO with custom Hamiltonian tensors.
    init_custom(tensors: list[NDArray[np.complex128]], transpose: bool = True) -> None
        Initializes the MPO with custom tensors.
    to_mps() -> MPS
        Converts the MPO to a Matrix Product State (MPS).
    to_matrix() -> NDArray[np.complex128]
        Converts the MPO to a full matrix representation.
    write_tensor_shapes() -> None
        Writes the shapes of the tensors in the MPO.
    check_if_valid_mpo() -> bool
        Checks if the MPO is valid.
    check_if_identity(fidelity: float) -> bool
        Checks if the MPO represents an identity operator with given fidelity.
    rotate(conjugate: bool = False) -> None
        Rotates the MPO tensors by swapping physical dimensions.
    """

    def init_ising(self, length: int, J: float, g: float) -> None:  # noqa: N803
        """Ising MPO.

        Initialize the Ising model as a Matrix Product Operator (MPO).
        This method constructs the MPO representation of the Ising model with
        specified parameters. The MPO has a 3x3 block structure at each site.

        Left boundary (1, 3, 2, 2)
        [I, -J Z, -g X]

        Inner tensor (3, 3, 2, 2)
        W = [[ I,     -J Z,  -g X ],
              [ 0,       0,     Z  ],
              [ 0,       0,     I  ]]

        Right boundary (3, 1, 2, 2)
        [I, -J Z, -g X]

        Args:
            length (int): The number of sites in the Ising chain.
            J (float): The coupling constant for the interaction.
            g (float): The coupling constant for the field.
        """
        physical_dimension = 2
        np.zeros((physical_dimension, physical_dimension), dtype=complex)
        identity = np.eye(physical_dimension, dtype=complex)
        x = ObservablesLibrary["x"]
        if length == 1:
            tensor: NDArray[np.complex128] = np.reshape(x, (2, 2, 1, 1))
            self.tensors = [tensor]
            self.length = length
            self.physical_dimension = physical_dimension
            return
        z = ObservablesLibrary["z"]

        left_bound = np.array([identity, -J * z, -g * x])[np.newaxis, :]

        # Inner tensors (3x3 block):
        inner = np.zeros((3, 3, physical_dimension, physical_dimension), dtype=complex)
        inner[0, 0] = identity
        inner[0, 1] = -J * z
        inner[0, 2] = -g * x
        inner[1, 2] = z
        inner[2, 2] = identity

        right_bound = np.array([[-g * x], [z], [identity]])

        # Construct the MPO as a list of tensors:
        self.tensors = [left_bound] + [inner] * (length - 2) + [right_bound]
        for i, tensor in enumerate(self.tensors):
            # left, right, sigma, sigma'
            self.tensors[i] = np.transpose(tensor, (2, 3, 0, 1))

        self.length = length
        self.physical_dimension = physical_dimension

    def init_heisenberg(self, length: int, Jx: float, Jy: float, Jz: float, h: float) -> None:  # noqa: N803
        """Heisenberg MPO.

        Initialize the Heisenberg model as a Matrix Product Operator (MPO).

        Left boundary: shape (1, 5, d, d)
        [I, Jx*X, Jy*Y, Jz*Z, h*Z]

        Inner tensor: shape (5, 5, d, d)
        W = [[ I,    Jx*X,  Jy*Y,  Jz*Z,   h*Z ],
              [ 0,     0,     0,     0,     X  ],
              [ 0,     0,     0,     0,     Y  ],
              [ 0,     0,     0,     0,     Z  ],
              [ 0,     0,     0,     0,     I  ]]

        Right boundary: shape (5, 1, d, d)
        [0, X, Y, Z, I]^T

        Parameters:
        length (int): The number of sites in the chain.
        Jx (float): The coupling constant for the X interaction.
        Jy (float): The coupling constant for the Y interaction.
        Jz (float): The coupling constant for the Z interaction.
        h (float): The magnetic field strength.
        """
        physical_dimension = 2
        zero = np.zeros((physical_dimension, physical_dimension), dtype=complex)
        identity = np.eye(physical_dimension, dtype=complex)
        x = ObservablesLibrary["x"]
        y = ObservablesLibrary["y"]
        z = ObservablesLibrary["z"]

        left_bound = np.array([identity, -Jx * x, -Jy * y, -Jz * z, -h * z])[np.newaxis, :]

        inner = np.zeros((5, 5, physical_dimension, physical_dimension), dtype=complex)
        inner[0, 0] = identity
        inner[0, 1] = -Jx * x
        inner[0, 2] = -Jy * y
        inner[0, 3] = -Jz * z
        inner[0, 4] = -h * z
        inner[1, 4] = x
        inner[2, 4] = y
        inner[3, 4] = z
        inner[4, 4] = identity

        right_bound = np.array([zero, x, y, z, identity])[:, np.newaxis]

        # Construct the MPO
        self.tensors = [left_bound] + [inner] * (length - 2) + [right_bound]
        for i, tensor in enumerate(self.tensors):
            # left, right, sigma, sigma'
            self.tensors[i] = np.transpose(tensor, (2, 3, 0, 1))
        self.length = length
        self.physical_dimension = physical_dimension

    def init_general_nearest_neighbor(self, length: int, a: float, b: float, c: float, d: float, e: float, f: float) -> None:  # noqa: N803
        """General nearest neighbor MPO.

        Initialize a general nearest neighbor with the following Hamiltonian as a MPO:
        H = a*XX + b*YY + c*ZZ + d*X + e*Y + f*Z

        Left boundary: shape (1, 5, d, d)
        [I, X, Y, Z, d*X + e*Y + f*Z]

        Inner tensor: shape (5, 5, d, d)
        W = [[ I,      X,     Y,     Z,   d*X + e*Y + f*Z ],
              [ 0,     0,     0,     0,          X  ],
              [ 0,     0,     0,     0,          Y  ],
              [ 0,     0,     0,     0,          Z  ],
              [ 0,     0,     0,     0,          I  ]]

        Right boundary: shape (5, 1, d, d)
        [d*X + e*Y + f*Z, a*X, b*Y, c*Z, I]^T

        Parameters:
        length (int): The number of sites in the chain.
        a (float): The coupling constant for the XX interaction.
        b (float): The coupling constant for the YY interaction.
        c (float): The coupling constant for the ZZ interaction.
        e (float): The coupling constant for the X interaction.
        e (float): The coupling constant for the Y interaction.
        f (float): The coupling constant for the Z interaction.
        """
        physical_dimension = 2
        identity = np.eye(physical_dimension, dtype=complex)
        x = ObservablesLibrary["x"]
        y = ObservablesLibrary["y"]
        z = ObservablesLibrary["z"]

        left_bound = np.array([identity, x, y, z, d*x + e*y + f*z])[np.newaxis, :]

        inner = np.zeros((5, 5, physical_dimension, physical_dimension), dtype=complex)
        inner[0, 0] = identity
        inner[0, 1] = x
        inner[0, 2] = y
        inner[0, 3] = z
        inner[0, 4] = d*x + e*y + f*z
        inner[1, 4] = a*x
        inner[2, 4] = b*y
        inner[3, 4] = c*z
        inner[4, 4] = identity

        right_bound = np.array([d*x + e*y + f*z, a*x, b*y, c*z, identity])[:, np.newaxis]

        # Construct the MPO
        self.tensors = [left_bound] + [inner] * (length - 2) + [right_bound]
        for i, tensor in enumerate(self.tensors):
            # left, right, sigma, sigma'
            self.tensors[i] = np.transpose(tensor, (2, 3, 0, 1))
        self.length = length
        self.physical_dimension = physical_dimension

    def init_identity(self, length: int, physical_dimension: int = 2) -> None:
        """Initialize identity MPO.

        Initializes the network with identity matrices.
        Parameters:
        length (int): The number of identity matrices to initialize.
        physical_dimension (int, optional): The physical dimension of the identity matrices. Default is 2.
        """
        mat = np.eye(2)
        mat = np.expand_dims(mat, (2, 3))
        self.length = length
        self.physical_dimension = physical_dimension

        self.tensors = []
        for _ in range(length):
            self.tensors.append(mat)

    def init_custom_hamiltonian(
        self,
        length: int,
        left_bound: NDArray[np.complex128],
        inner: NDArray[np.complex128],
        right_bound: NDArray[np.complex128],
    ) -> None:
        """Custom Hamiltonian from finite state machine MPO.

        Initialize a custom Hamiltonian as a Matrix Product Operator (MPO).
        This method sets up the Hamiltonian using the provided boundary and inner tensors.
        The tensors are transposed to match the expected shape for MPOs.

        Args:
            length (int): The number of tensors in the MPO.
            left_bound (NDArray[np.complex128]): The tensor at the left boundary.
            inner (NDArray[np.complex128]): The tensor for the inner sites.
            right_bound (NDArray[np.complex128]): The tensor at the right boundary.
        """
        self.tensors = [left_bound] + [inner] * (length - 2) + [right_bound]
        for i, tensor in enumerate(self.tensors):
            # left, right, sigma, sigma'
            self.tensors[i] = np.transpose(tensor, (2, 3, 0, 1))
        assert self.check_if_valid_mpo(), "MPO initialized wrong"
        self.length = len(self.tensors)
        self.physical_dimension = self.tensors[0].shape[0]

    def init_custom(self, tensors: list[NDArray[np.complex128]], *, transpose: bool = True) -> None:
        """Custom MPO from tensors.

        Initialize the custom MPO (Matrix Product Operator) with the given tensors.
        Parameters.
        ----------
        tensors : list[NDArray[np.complex128]]
            A list of tensors to initialize the MPO.
        transpose : bool, optional
            If True, transpose each tensor to the order (2, 3, 0, 1). Default is True.

        Raises:
        ------
        AssertionError
            If the MPO is initialized incorrectly.

        Notes:
        -----
        This method sets the tensors, optionally transposes them, checks if the MPO is valid,
        and initializes the length and physical dimension of the MPO.
        """
        self.tensors = tensors
        if transpose:
            for i, tensor in enumerate(self.tensors):
                # left, right, sigma, sigma'
                self.tensors[i] = np.transpose(tensor, (2, 3, 0, 1))
        assert self.check_if_valid_mpo(), "MPO initialized wrong"
        self.length = len(self.tensors)
        self.physical_dimension = tensors[0].shape[0]

    def to_mps(self) -> MPS:
        """MPO to MPS conversion.

        Converts the current tensor network to a Matrix Product State (MPS) representation.
        This method reshapes each tensor in the network from shape
        (dim1, dim2, dim3, dim4) to (dim1 * dim2, dim3, dim4) and
        returns a new MPS object with the converted tensors.

        Returns:
            MPS: An MPS object containing the reshaped tensors.
        """
        converted_tensors: list[NDArray[np.complex128]] = [
            np.reshape(tensor, (tensor.shape[0] * tensor.shape[1], tensor.shape[2], tensor.shape[3]))
            for tensor in self.tensors
        ]

        return MPS(self.length, converted_tensors)

    def to_matrix(self) -> NDArray[np.complex128]:
        """MPO to matrix conversion.

        Converts a list of tensors into a matrix using Einstein summation convention.
        This method iterates over the list of tensors and performs tensor contractions
        using the Einstein summation convention (`oe.constrain`). The resulting tensor is
        then reshaped accordingly. The final matrix is squeezed to ensure the left and
        right bonds are 1.

        Returns:
            NDArray[np.complex128]: The resulting matrix after tensor contractions and reshaping.
        """
        for i, tensor in enumerate(self.tensors):
            if i == 0:
                mat = tensor
            else:
                mat = oe.contract("abcd, efdg->aebfcg", mat, tensor)
                mat = np.reshape(
                    mat, (mat.shape[0] * mat.shape[1], mat.shape[2] * mat.shape[3], mat.shape[4], mat.shape[5])
                )

        # Final left and right bonds should be 1
        return np.squeeze(mat, axis=(2, 3))

    def check_if_valid_mpo(self) -> bool:
        """MPO validity check.

        Check if the current tensor network is a valid Matrix Product Operator (MPO).
        This method verifies the consistency of the bond dimensions between adjacent tensors
        in the network. Specifically, it checks that the right bond dimension of each tensor
        matches the left bond dimension of the subsequent tensor.

        Returns:
            bool: True if the tensor network is a valid MPO, False otherwise.
        """
        right_bond = self.tensors[0].shape[3]
        for tensor in self.tensors[1::]:
            assert tensor.shape[2] == right_bond
            right_bond = tensor.shape[3]
        return True

    def check_if_identity(self, fidelity: float) -> bool:
        """MPO Identity check.

        Check if the current MPO (Matrix Product Operator) represents an identity operation
        within a given fidelity threshold.

        Args:
            fidelity (float): The fidelity threshold to determine if the MPO is an identity.

        Returns:
            bool: True if the MPO is considered an identity within the given fidelity, False otherwise.
        """
        identity_mpo = MPO()
        identity_mpo.init_identity(self.length)

        identity_mps = identity_mpo.to_mps()
        mps = self.to_mps()
        trace = mps.scalar_product(identity_mps)

        # Checks if trace is not a singular values for partial trace
        return not np.round(np.abs(trace), 1) / 2**self.length < fidelity

    def rotate(self, *, conjugate: bool = False) -> None:
        """Rotates MPO.

        Rotates the tensors in the network by flipping the physical dimensions.
        This method transposes each tensor in the network along specified axes.
        If the `conjugate` parameter is set to True, it also takes the complex
        conjugate of each tensor before transposing.

        Args:
            conjugate (bool): If True, take the complex conjugate of each tensor
                              before transposing. Default is False.
        """
        for i, tensor in enumerate(self.tensors):
            if conjugate:
                self.tensors[i] = np.transpose(np.conj(tensor), (1, 0, 2, 3))
            else:
                self.tensors[i] = np.transpose(tensor, (1, 0, 2, 3))

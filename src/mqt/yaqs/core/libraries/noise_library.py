# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Library of noise processes.

This module defines noise operator classes for quantum systems.
It includes implementations for raising, lowering, and pauli_z, pauli_x, pauli_y noise operators,
each represented as a 2x2 numpy array. The module also provides a NoiseLibrary class
that aggregates these noise operators for convenient access. Future improvements
may extend these implementations to d-level systems.
"""

from __future__ import annotations

import numpy as np


class Raising:
    """Class representing excitation noise.

    Attributes:
        d: The dimension of the Hilbert space. Defaults to 2.
        matrix: A 2x2 matrix representing the excitation operator.
            The matrix is constructed such that matrix[row, col] = 1 if row - col == 1, and 0 otherwise.
    """

    d = 2
    matrix = np.zeros((d, d))
    for row, array in enumerate(matrix):
        for col, _ in enumerate(array):
            if row - col == 1:
                matrix[row][col] = 1


class Lowering:
    """Class representing relaxation noise.

    Attributes:
        d (int): The dimension of the Hilbert space. Defaults to 2.
        matrix (np.ndarray): A 2x2 matrix representing the lowering operator.
            The matrix is constructed such that matrix[row, col] = 1 if col - row == 1, and 0 otherwise.
    """

    d = 2
    matrix = np.zeros((d, d))
    for row, array in enumerate(matrix):
        for col, _ in enumerate(array):
            if col - row == 1:
                matrix[row][col] = 1


class PauliZ:
    """Class representing PauliZ (dephasing) noise.

    Attributes:
        matrix (np.ndarray): A 2x2 matrix representing the dephasing operator,
            defined as [[1, 0], [0, -1]].
    """

    matrix = np.array([[1, 0], [0, -1]])


class PauliX:
    """Class representing PauliX (bitflip) noise.

    Attributes:
        matrix (np.ndarray): A 2x2 matrix representing the Pauli-X operator,
            defined as [[0, 1], [1, 0]].
    """

    matrix = np.array([[0, 1], [1, 0]])


class PauliY:
    """Class representing PauliY (bit-phase flip) noise.

    Attributes:
        matrix (np.ndarray): A 2x2 matrix representing the Pauli-Y operator,
            defined as [[0, -1j], [1j, 0]].
    """

    matrix = np.array([[0, -1j], [1j, 0]])


class TwoSiteRaising:
    """Class representing two-site excitation noise.

    Attributes:
        matrix (np.ndarray): A 4x4 matrix representing the tensor product Excitation x Excitation.
    """

    matrix = np.kron(Raising.matrix, Raising.matrix)


class TwoSiteLowering:
    """Class representing two-site relaxation noise.

    Attributes:
        matrix (np.ndarray): A 4x4 matrix representing the tensor product Lowering x Lowering.
    """

    matrix = np.kron(Lowering.matrix, Lowering.matrix)


class CrossTalkZZ:
    """Class representing cross talk between neighboring sites along the z-axis.

    Attributes:
        matrix (np.ndarray): A 4x4 matrix representing the tensor product PauliZ x PauliZ.
    """

    matrix = np.kron(PauliZ.matrix, PauliZ.matrix)


class CrossTalkXX:
    """Class representing cross talk between neighboring sites along the x-axis.

    Attributes:
        matrix (np.ndarray): A 4x4 matrix representing the tensor product X x X.
    """

    matrix = np.kron(PauliX.matrix, PauliX.matrix)


class CrossTalkYY:
    """Class representing cross talk between neighboring sites along the y-axis.

    Attributes:
        matrix (np.ndarray): A 4x4 matrix representing the tensor product Y x Y.
    """

    matrix = np.kron(PauliY.matrix, PauliY.matrix)


class CrossTalkXY:
    """Class representing cross talk between neighboring sites with X x Y.

    Attributes:
        matrix (np.ndarray): A 4x4 matrix representing the tensor product X x Y.
    """

    matrix = np.kron(PauliX.matrix, PauliY.matrix)


class CrossTalkYX:
    """Class representing cross talk between neighboring sites with Y x X.

    Attributes:
        matrix (np.ndarray): A 4x4 matrix representing the tensor product Y x X.
    """

    matrix = np.kron(PauliY.matrix, PauliX.matrix)


class CrossTalkZY:
    """Class representing cross talk between neighboring sites with Z x Y.

    Attributes:
        matrix (np.ndarray): A 4x4 matrix representing the tensor product Z x Y.
    """

    matrix = np.kron(PauliZ.matrix, PauliY.matrix)


class CrossTalkZX:
    """Class representing cross talk between neighboring sites with Z x X.

    Attributes:
        matrix (np.ndarray): A 4x4 matrix representing the tensor product Z x X.
    """

    matrix = np.kron(PauliZ.matrix, PauliX.matrix)


class CrossTalkYZ:
    """Class representing cross talk between neighboring sites with Y x Z.

    Attributes:
        matrix (np.ndarray): A 4x4 matrix representing the tensor product Y x Z.
    """

    matrix = np.kron(PauliY.matrix, PauliZ.matrix)


class CrossTalkXZ:
    """Class representing cross talk between neighboring sites with X x Z.

    Attributes:
        matrix (np.ndarray): A 4x4 matrix representing the tensor product X x Z.
    """

    matrix = np.kron(PauliX.matrix, PauliZ.matrix)


class LongRangeCrosstalkXX:
    """Long-range crosstalk factors X ⊗ X (factor-only, no full matrix).

    Attributes:
        factors (tuple[np.ndarray, np.ndarray]): Per-site 2x2 operators (A at first site, B at second).
    """

    factors = (PauliX.matrix, PauliX.matrix)


class LongRangeCrosstalkXY:
    """Long-range crosstalk factors X ⊗ Y (factor-only, no full matrix)."""

    factors = (PauliX.matrix, PauliY.matrix)


class LongRangeCrosstalkXZ:
    """Long-range crosstalk factors X ⊗ Z (factor-only, no full matrix)."""

    factors = (PauliX.matrix, PauliZ.matrix)


class LongRangeCrosstalkYX:
    """Long-range crosstalk factors Y ⊗ X (factor-only, no full matrix)."""

    factors = (PauliY.matrix, PauliX.matrix)


class LongRangeCrosstalkYY:
    """Long-range crosstalk factors Y ⊗ Y (factor-only, no full matrix)."""

    factors = (PauliY.matrix, PauliY.matrix)


class LongRangeCrosstalkYZ:
    """Long-range crosstalk factors Y ⊗ Z (factor-only, no full matrix)."""

    factors = (PauliY.matrix, PauliZ.matrix)


class LongRangeCrosstalkZX:
    """Long-range crosstalk factors Z ⊗ X (factor-only, no full matrix)."""

    factors = (PauliZ.matrix, PauliX.matrix)


class LongRangeCrosstalkZY:
    """Long-range crosstalk factors Z ⊗ Y (factor-only, no full matrix)."""

    factors = (PauliZ.matrix, PauliY.matrix)


class LongRangeCrosstalkZZ:
    """Long-range crosstalk factors Z ⊗ Z (factor-only, no full matrix)."""

    factors = (PauliZ.matrix, PauliZ.matrix)


class NoiseLibrary:
    """A library of noise processes.

    Attributes:
        raising: Raising noise (0 --> 1).
        lowering: Lowering noise (1 --> 0).
        pauli_z: PauliZ (dephasing) noise.
        pauli_x: PauliX (bitflip) noise (0 --> 1, 1 --> 0).
        pauli_y: PauliY (bit-phase flip) noise.
        raising_two: Two-site raising noise (00 --> 11).
        lowering_two: Two-site lowering noise (11 --> 00).
        crosstalk_zz: Cross talk between neighboring sites along the z-axis.
        crosstalk_xx: Cross talk between neighboring sites along the x-axis.
        crosstalk_y: Cross talk between neighboring sites along the y-axis.
        crosstalk_xy: Cross talk between neighboring sites with X x Y.
        crosstalk_yx: Cross talk between neighboring sites with Y x X.
        crosstalk_zy: Cross talk between neighboring sites with Z x Y.
        crosstalk_zx: Cross talk between neighboring sites with Z x X.
        crosstalk_yz: Cross talk between neighboring sites with Y x Z.
        crosstalk_xz: Cross talk between neighboring sites with X x Z.
        longrange_crosstalk_xx: Long-range crosstalk factors X,X (factor-only, no full matrix).
        longrange_crosstalk_xy: Long-range crosstalk factors X,Y (factor-only, no full matrix).
        longrange_crosstalk_xz: Long-range crosstalk factors X,Z (factor-only, no full matrix).
        longrange_crosstalk_yx: Long-range crosstalk factors Y,X (factor-only, no full matrix).
        longrange_crosstalk_yy: Long-range crosstalk factors Y,Y (factor-only, no full matrix).
        longrange_crosstalk_yz: Long-range crosstalk factors Y,Z (factor-only, no full matrix).
        longrange_crosstalk_zx: Long-range crosstalk factors Z,X (factor-only, no full matrix).
        longrange_crosstalk_zy: Long-range crosstalk factors Z,Y (factor-only, no full matrix).
        longrange_crosstalk_zz: Long-range crosstalk factors Z,Z (factor-only, no full matrix).
    """

    # Canonical names
    raising = Raising
    lowering = Lowering
    pauli_z = PauliZ
    pauli_x = PauliX
    pauli_y = PauliY
    raising_two = TwoSiteRaising
    lowering_two = TwoSiteLowering
    crosstalk_zz = CrossTalkZZ
    crosstalk_xx = CrossTalkXX
    crosstalk_yy = CrossTalkYY
    crosstalk_xy = CrossTalkXY
    crosstalk_yx = CrossTalkYX
    crosstalk_zy = CrossTalkZY
    crosstalk_zx = CrossTalkZX
    crosstalk_yz = CrossTalkYZ
    crosstalk_xz = CrossTalkXZ
    longrange_crosstalk_xx = LongRangeCrosstalkXX
    longrange_crosstalk_xy = LongRangeCrosstalkXY
    longrange_crosstalk_xz = LongRangeCrosstalkXZ
    longrange_crosstalk_yx = LongRangeCrosstalkYX
    longrange_crosstalk_yy = LongRangeCrosstalkYY
    longrange_crosstalk_yz = LongRangeCrosstalkYZ
    longrange_crosstalk_zx = LongRangeCrosstalkZX
    longrange_crosstalk_zy = LongRangeCrosstalkZY
    longrange_crosstalk_zz = LongRangeCrosstalkZZ

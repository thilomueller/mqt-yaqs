# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Library of noise processes.

This module defines noise operator classes for quantum systems.
It includes implementations for excitation, relaxation, and pauli_z, pauli_x, pauli_y noise operators,
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
        matrix (np.ndarray): A 2x2 matrix representing the relaxation operator.
            The matrix is constructed such that matrix[row, col] = 1 if col - row == 1, and 0 otherwise.
    """

    d = 2
    matrix = np.zeros((d, d))
    for row, array in enumerate(matrix):
        for col, _ in enumerate(array):
            if col - row == 1:
                matrix[row][col] = 1


class Pauli_Z:
    """Class representing Pauli_Z (dephasing) noise.

    Attributes:
        matrix (np.ndarray): A 2x2 matrix representing the dephasing operator,
            defined as [[1, 0], [0, -1]].
    """

    matrix = np.array([[1, 0], [0, -1]])


class Pauli_X:
    """Class representing Pauli_X (bitflip) noise.

    Attributes:
        matrix (np.ndarray): A 2x2 matrix representing the Pauli-X operator,
            defined as [[0, 1], [1, 0]].
    """

    matrix = np.array([[0, 1], [1, 0]])


class Pauli_Y:
    """Class representing Pauli_Y (bit-phase flip) noise.

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
        matrix (np.ndarray): A 4x4 matrix representing the tensor product Relaxation x Relaxation.
    """

    matrix = np.kron(Lowering.matrix, Lowering.matrix)


class CrossTalkZZ:
    """Class representing cross talk between neighboring sites along the z-axis.

    Attributes:
        matrix (np.ndarray): A 4x4 matrix representing the tensor product Pauli_Z x Pauli_Z.
    """

    matrix = np.kron(Pauli_Z.matrix, Pauli_Z.matrix)


class CrossTalkXX:
    """Class representing cross talk between neighboring sites along the x-axis.

    Attributes:
        matrix (np.ndarray): A 4x4 matrix representing the tensor product X x X.
    """

    matrix = np.kron(Pauli_X.matrix, Pauli_X.matrix)


class CrossTalkYY:
    """Class representing cross talk between neighboring sites along the y-axis.

    Attributes:
        matrix (np.ndarray): A 4x4 matrix representing the tensor product Y x Y.
    """

    matrix = np.kron(Pauli_Y.matrix, Pauli_Y.matrix)


class CrossTalkXY:
    """Class representing cross talk between neighboring sites with X x Y.

    Attributes:
        matrix (np.ndarray): A 4x4 matrix representing the tensor product X x Y.
    """

    matrix = np.kron(Pauli_X.matrix, Pauli_Y.matrix)


class CrossTalkYX:
    """Class representing cross talk between neighboring sites with Y x X.

    Attributes:
        matrix (np.ndarray): A 4x4 matrix representing the tensor product Y x X.
    """

    matrix = np.kron(Pauli_Y.matrix, Pauli_X.matrix)


class CrossTalkZY:
    """Class representing cross talk between neighboring sites with Z x Y.

    Attributes:
        matrix (np.ndarray): A 4x4 matrix representing the tensor product Z x Y.
    """

    matrix = np.kron(Pauli_Z.matrix, Pauli_Y.matrix)


class CrossTalkZX:
    """Class representing cross talk between neighboring sites with Z x X.

    Attributes:
        matrix (np.ndarray): A 4x4 matrix representing the tensor product Z x X.
    """

    matrix = np.kron(Pauli_Z.matrix, Pauli_X.matrix)


class CrossTalkYZ:
    """Class representing cross talk between neighboring sites with Y x Z.

    Attributes:
        matrix (np.ndarray): A 4x4 matrix representing the tensor product Y x Z.
    """

    matrix = np.kron(Pauli_Y.matrix, Pauli_Z.matrix)


class CrossTalkXZ:
    """Class representing cross talk between neighboring sites with X x Z.

    Attributes:
        matrix (np.ndarray): A 4x4 matrix representing the tensor product X x Z.
    """

    matrix = np.kron(Pauli_X.matrix, Pauli_Z.matrix)


class NoiseLibrary:
    """A library of noise processes.

    Attributes:
        raising: Raising noise (0 --> 1).
        lowering: Lowering noise (1 --> 0).
        pauli_z: Pauli_Z (dephasing) noise.
        pauli_x: Pauli_X (bitflip) noise (0 --> 1, 1 --> 0).
        pauli_y: Pauli_Y (bit-phase flip) noise.
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
    """

    # Canonical names
    raising = Raising
    lowering = Lowering
    pauli_z = Pauli_Z
    pauli_x = Pauli_X
    pauli_y = Pauli_Y
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

    # Backward-compatibility aliases (expected by tests and external users)
    excitation = Raising
    relaxation = Lowering
    dephasing = Pauli_Z
    bitflip = Pauli_X
    bitphaseflip = Pauli_Y
    excitation_two = TwoSiteRaising
    relaxation_two = TwoSiteLowering
    # Common crosstalk shorthands
    crosstalk = CrossTalkZZ
    crosstalk_x = CrossTalkXX
    crosstalk_y = CrossTalkYY

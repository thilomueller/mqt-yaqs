# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Library of noise processes.

This module defines noise operator classes for quantum systems.
It includes implementations for excitation, relaxation, and dephasing noise operators,
each represented as a 2x2 numpy array. The module also provides a NoiseLibrary class
that aggregates these noise operators for convenient access. Future improvements
may extend these implementations to d-level systems.
"""

from __future__ import annotations

import numpy as np


class Excitation:
    """Class representing the excitation noise operator for a two-level system.

    Attributes:
        d (int): The dimension of the Hilbert space. Defaults to 2.
        matrix (np.ndarray): A 2x2 matrix representing the excitation operator.
            The matrix is constructed such that matrix[row, col] = 1 if row - col == 1, and 0 otherwise.
    """

    d = 2
    matrix = np.zeros((d, d))
    for row, array in enumerate(matrix):
        for col, _ in enumerate(array):
            if row - col == 1:
                matrix[row][col] = 1


class Relaxation:
    """Class representing the relaxation noise operator for a two-level system.

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


class Dephasing:
    """Class representing the dephasing noise operator for a two-level system.

    Attributes:
        matrix (np.ndarray): A 2x2 matrix representing the dephasing operator,
            defined as [[1, 0], [0, -1]].
    """

    matrix = np.array([[1, 0], [0, -1]])


class BitFlip:
    """Class representing the Pauli-X (bit-flip) noise operator for a two-level system.

    Attributes:
        matrix (np.ndarray): A 2x2 matrix representing the Pauli-X operator,
            defined as [[0, 1], [1, 0]].
    """

    matrix = np.array([[0, 1], [1, 0]])


class BitPhaseFlip:
    """Class representing the Pauli-Y noise operator for a two-level system.

    Attributes:
        matrix (np.ndarray): A 2x2 matrix representing the Pauli-Y operator,
            defined as [[0, -1j], [1j, 0]].
    """

    matrix = np.array([[0, -1j], [1j, 0]])


class TwoSiteExcitation:
    """Class representing the two-site excitation noise operator acting on two neighboring sites.

    Attributes:
        matrix (np.ndarray): A 4x4 matrix representing the tensor product Excitation ⊗ Excitation.
    """

    matrix = np.kron(Excitation.matrix, Excitation.matrix)


class TwoSiteRelaxation:
    """Class representing the two-site relaxation noise operator acting on two neighboring sites.

    Attributes:
        matrix (np.ndarray): A 4x4 matrix representing the tensor product Relaxation ⊗ Relaxation.
    """

    matrix = np.kron(Relaxation.matrix, Relaxation.matrix)


class CrossTalk:
    """Class representing the cross-talk operator acting on two neighboring sites.

    Attributes:
        matrix (np.ndarray): A 4x4 matrix representing the tensor product Dephasing ⊗ Dephasing.
    """

    matrix = np.kron(Dephasing.matrix, Dephasing.matrix)


class CrossTalk_X:
    """Class representing the x-axis cross-talk noise operator.

    Attributes:
        matrix (np.ndarray): A 4x4 matrix representing the tensor product X ⊗ X.
    """

    matrix = np.kron(BitFlip.matrix, BitFlip.matrix)


class CrossTalk_Y:
    """Class representing the y-axis cross-talk noise operator.

    Attributes:
        matrix (np.ndarray): A 4x4 matrix representing the tensor product Y ⊗ Y.
    """

    matrix = np.kron(BitPhaseFlip.matrix, BitPhaseFlip.matrix)


class NoiseLibrary:
    """A library of noise operator classes.

    Attributes:
        excitation: Class representing the excitation noise operator.
        relaxation: Class representing the relaxation noise operator.
        dephasing: Class representing the dephasing noise operator.
        double_excitation: Class representing the double excitation noise operator.
        double_relaxation: Class representing the double relaxation noise operator.
        double_dephasing: Class representing the double dephasing noise operator.
        x: Class representing the Pauli-X noise operator.
        y: Class representing the Pauli-Y noise operator.
        xx: Class representing the two-qubit Pauli-XX noise operator.
        yy: Class representing the two-qubit Pauli-YY noise operator.
    """

    excitation = Excitation
    relaxation = Relaxation
    dephasing = Dephasing
    bitflip = BitFlip
    bitphaseflip = BitPhaseFlip
    excitation_two = TwoSiteExcitation
    relaxation_two = TwoSiteRelaxation
    crosstalk = CrossTalk
    crosstalk_x = CrossTalk_X
    crosstalk_y = CrossTalk_Y


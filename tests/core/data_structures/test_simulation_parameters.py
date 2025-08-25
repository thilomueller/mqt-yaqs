# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for simulation parameters classes.

This module contains unit tests for the Observable and AnalogSimParams classes used in
quantum simulation. It verifies that:
  - An Observable is correctly initialized with valid parameters and that invalid parameters
    raise an appropriate error.
  - AnalogSimParams instances are created with the correct attributes (such as elapsed_time, dt, times,
    sample_timesteps, and num_traj) both with explicit and default values.
  - The Observable.initialize method properly sets up the results and trajectories arrays
    depending on whether sample_timesteps is True or False.
"""

# ignore non-lowercase variable names for physics notation

from __future__ import annotations

import numpy as np

from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
from mqt.yaqs.core.libraries.gate_library import GateLibrary, X


def test_observable_creation_valid() -> None:
    """Test that an Observable is created correctly with valid parameters.

    This test constructs an Observable with the name "x" on site 0 and verifies that its attributes
    (name, site, results, and trajectories) are correctly initialized.
    """
    gate = X()
    site = 0
    obs = Observable(gate, site)

    assert np.array_equal(obs.gate.matrix, np.array([[0, 1], [1, 0]]))
    assert obs.sites == site
    assert obs.results is None
    assert obs.trajectories is None


def test_analog_simparams_basic() -> None:
    """Test that AnalogSimParams is initialized with correct parameters.

    This test creates a AnalogSimParams instance with a single observable, total time elapsed_time, time step dt,
    sample_timesteps flag set to True, and a specified number of trajectories num_traj. It then verifies that the
    observables, elapsed_time, dt, times array, sample_timesteps flag, and num_traj are set correctly.
    """
    obs_list = [Observable(X(), 0)]
    elapsed_time = 1.0
    dt = 0.2
    params = AnalogSimParams(obs_list, elapsed_time, dt=dt, sample_timesteps=True, num_traj=50)

    assert params.observables == obs_list
    assert params.elapsed_time == elapsed_time
    assert params.dt == dt
    expected_times = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    assert np.allclose(params.times, expected_times), "Times array should match numpy.arange(0, elapsed_time+dt, dt)."
    assert params.sample_timesteps is True
    assert params.num_traj == 50


def test_analog_simparams_defaults() -> None:
    """Test the default parameters for AnalogSimParams.

    This test constructs a AnalogSimParams instance with an empty observable list and total time elapsed_time,
    and verifies that default values for dt, sample_timesteps, number of trajectories (num_traj), max_bond_dim,
    threshold, and order are correctly assigned.
    """
    obs_list: list[Observable] = []
    elapsed_time = 2.0
    params = AnalogSimParams(obs_list, elapsed_time)

    assert params.observables == obs_list
    assert params.elapsed_time == 2.0
    assert params.dt == 0.1
    assert params.sample_timesteps is True
    # times should be np.arange(0, elapsed_time+dt, dt)
    assert np.isclose(params.times[-1], 2.0)
    assert params.num_traj == 1000
    assert params.max_bond_dim == 4096
    assert params.threshold == 1e-9
    assert params.order == 1


def test_observable_initialize_with_sample_timesteps() -> None:
    """Test that Observable.initialize sets up results and trajectories correctly when sample_timesteps is True.

    This test creates an Observable on site 1 and a AnalogSimParams instance with sample_timesteps=True.
    It verifies that the results array has shape equal to the length of the times array and that the
    trajectories array has shape (num_traj, len(times)).
    """
    obs = Observable(X(), 1)
    sim_params = AnalogSimParams([obs], elapsed_time=1.0, dt=0.5, sample_timesteps=True, num_traj=10)
    # sim_params.times => [0.0, 0.5, 1.0]

    obs.initialize(sim_params)
    assert obs.results is not None
    assert obs.trajectories is not None
    assert obs.results.shape == (3,), "results should match len(sim_params.times)."
    assert obs.trajectories.shape == (sim_params.num_traj, 3), "trajectories should have shape (num_traj, len(times))."


def test_observable_initialize_without_sample_timesteps() -> None:
    """Test that Observable.initialize sets up results and trajectories correctly when sample_timesteps is False.

    This test creates an Observable on site 0 and a AnalogSimParams instance with sample_timesteps=False.
    It verifies that the results array has shape equal to the length of the times array, the trajectories array
    has shape (num_traj, 1), and that the observable's times attribute is set to elapsed_time.
    """
    obs = Observable(X(), 0)
    sim_params = AnalogSimParams([obs], elapsed_time=1.0, dt=0.25, sample_timesteps=False, num_traj=5)
    # times => [0.0, 0.25, 0.5, 0.75, 1.0]

    obs.initialize(sim_params)
    assert obs.results is not None
    assert obs.trajectories is not None
    assert obs.results.shape == (len(sim_params.times),)
    assert obs.trajectories.shape == (sim_params.num_traj, 1)
    assert obs.times == 1.0, "If sample_timesteps=False, obs.times should be equal to elapsed_time."


def test_observable_from_string_runtime_cost() -> None:
    """Constructor maps 'runtime_cost' string to the runtime_cost diagnostic gate."""
    obs = Observable("runtime_cost", sites=0)
    assert obs.gate.name == "runtime_cost"
    # placeholder identity backing for diagnostics
    assert obs.gate.matrix.shape == (2, 2)
    assert np.allclose(obs.gate.matrix, np.eye(2))


def test_observable_from_string_max_total_bond() -> None:
    """Constructor maps 'max_bond' and 'total_bond' to their diagnostic gates."""
    obs_max = Observable("max_bond", sites=1)
    obs_tot = Observable("total_bond", sites=2)

    assert obs_max.gate.name == "max_bond"
    assert obs_tot.gate.name == "total_bond"
    assert np.allclose(obs_max.gate.matrix, np.eye(2))
    assert np.allclose(obs_tot.gate.matrix, np.eye(2))


def test_observable_from_string_entropy_and_spectrum_with_list_sites() -> None:
    """Constructor maps 'entropy' and 'schmidt_spectrum' and accepts list[int] sites."""
    cut = [3, 4]
    obs_ent = Observable("entropy", sites=cut)
    obs_ssp = Observable("schmidt_spectrum", sites=cut)

    assert obs_ent.gate.name == "entropy"
    assert obs_ssp.gate.name == "schmidt_spectrum"
    # meta-observables use identity placeholders for BaseGate compatibility
    assert np.allclose(obs_ent.gate.matrix, np.eye(2))
    assert np.allclose(obs_ssp.gate.matrix, np.eye(2))
    assert obs_ent.sites == cut
    assert obs_ssp.sites == cut


def test_observable_from_string_falls_back_to_pvm() -> None:
    """Any other string is interpreted as a PVM bitstring; gate must store that bitstring."""
    bitstring = "10101"
    obs = Observable(bitstring, sites=None)
    assert obs.gate.name == "pvm"
    # gate must expose the queried bitstring
    assert hasattr(obs.gate, "bitstring")
    assert obs.gate.bitstring == bitstring
    # PVM uses identity placeholder matrix for compatibility in your implementation
    assert np.allclose(obs.gate.matrix, np.eye(2))


def test_observable_from_gate_instance_keeps_gate_and_sites_int() -> None:
    """Passing a concrete BaseGate instance should be preserved and sites can be an int."""
    x_gate = GateLibrary.x()
    obs = Observable(x_gate, sites=5)
    # same object semantics not required; equality via matrix is sufficient
    assert obs.gate.name == "x"
    assert np.allclose(obs.gate.matrix, x_gate.matrix)
    assert obs.sites == 5


def test_observable_from_gate_instance_with_list_sites() -> None:
    """Gate instance + list[int] sites should preserve the list (for two-site ops)."""
    cz_gate = GateLibrary.cz()
    obs = Observable(cz_gate, sites=[1, 3])
    assert obs.gate.name == "cz"
    assert obs.sites == [1, 3]

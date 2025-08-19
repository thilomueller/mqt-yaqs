# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""High-level simulator module for using YAQS.

This module implements the common simulation routine for both circuit-based and Hamiltonian (analog) simulations.
It provides functions to run simulation trajectories in parallel using an MPS representation of the quantum state.
Depending on the type of simulation parameters provided (WeakSimParams, StrongSimParams, or AnalogSimParams),
the simulation is dispatched to the appropriate backend:
  - For circuit simulations, a QuantumCircuit is used and processed via the _run_circuit function.
  - For analog simulations, an MPO is used to represent the Hamiltonian and processed via the _run_analog function.

The module supports both strong and weak simulation schemes, including functionality for:
  - Initializing the state (MPS) to a canonical form (B normalized).
  - Running trajectories with noise (using a provided NoiseModel) and aggregating results.
  - Parallel execution of trajectories using a ProcessPoolExecutor with progress reporting via tqdm.

All simulation results (e.g., observables, measurements) are aggregated and returned as part of the simulation process.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Thread caps MUST be set before importing numpy/qiskit/etc.
# (Placed immediately after __future__ to satisfy Python's import rules.)
# ---------------------------------------------------------------------------
import os as _os

for _k, _v in {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
}.items():
    _os.environ.setdefault(_k, _v)

# ---------------------------------------------------------------------------

import concurrent.futures
import copy
import multiprocessing
import os
from typing import TYPE_CHECKING, Iterable, List, Tuple

# Soft dependency; we guard usage if unavailable
try:
    from threadpoolctl import threadpool_limits, threadpool_info  # type: ignore
except Exception:  # pragma: no cover - optional dep
    threadpool_limits = None  # type: ignore[assignment]
    threadpool_info = None  # type: ignore[assignment]

from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from tqdm import tqdm

from .analog.analog_tjm import analog_tjm_1, analog_tjm_2
from .core.data_structures.networks import MPO
from .core.data_structures.simulation_parameters import AnalogSimParams, StrongSimParams, WeakSimParams
from .digital.digital_tjm import digital_tjm

if TYPE_CHECKING:
    from .core.data_structures.networks import MPS
    from .core.data_structures.noise_model import NoiseModel


# --------------------------
# Linux/cgroup-safe CPU count
# --------------------------
def available_cpus() -> int:
    """Determine the number of CPUs visible to this process.

    Prefers Linux cpu affinity/cgroups when available, then SLURM variables,
    and finally falls back to multiprocessing.cpu_count().
    """
    # 1) Respect Linux affinity/cgroups (taskset, container limits)
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        pass

    # 2) Respect SLURM allocations if present
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        value = os.environ.get(var, "").strip()
        if value:
            try:
                n = int(value)
                if n > 0:
                    return n
            except ValueError:
                pass

    # 3) Fallback
    return multiprocessing.cpu_count() or 1


# ----------------------------------------
# Prevent thread oversubscription per worker
# ----------------------------------------
THREAD_ENV_VARS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",  # harmless on Linux
    "BLIS_NUM_THREADS": "1",
}


def _limit_worker_threads(n_threads: int = 1) -> None:
    """Initializer for worker processes to cap BLAS/OpenMP threads."""
    # Env vars are already set at import time; setdefault keeps user overrides
    for k, v in THREAD_ENV_VARS.items():
        os.environ.setdefault(k, str(n_threads))

    # Optional hardening if these libs are available
    try:
        import numexpr  # type: ignore
        numexpr.set_num_threads(n_threads)
    except Exception:
        pass

    try:
        import mkl  # type: ignore
        mkl.set_num_threads(n_threads)
    except Exception:
        pass

    if threadpool_limits is not None:
        try:
            threadpool_limits(limits=n_threads)
        except Exception:
            pass

    # Debug: print pools per worker if requested
    if os.environ.get("YAQS_THREAD_DEBUG", "") == "1" and threadpool_info is not None:
        try:
            info = threadpool_info()
            print(f"[worker {os.getpid()}] threadpoolctl info: {info}")
        except Exception:
            pass


def _call_backend(backend, arg):
    """Call backend(arg) with strict 1-thread cap even if libraries spawn later."""
    if threadpool_limits is not None:
        try:
            with threadpool_limits(limits=1):
                return backend(arg)
        except Exception:
            return backend(arg)
    return backend(arg)


def _chunks(seq: List, size: int) -> Iterable[List]:
    """Yield successive chunks from seq of length size."""
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _run_batch(backend, batch_args: List[Tuple]):
    """Run a list of backend calls in-process, applying per-call caps."""
    out = []
    for arg in batch_args:
        out.append(_call_backend(backend, arg))
    return out


def _run_strong_sim(
    initial_state: MPS,
    operator: QuantumCircuit,
    sim_params: StrongSimParams,
    noise_model: NoiseModel | None,
    *,
    parallel: bool,
) -> None:
    """Run strong simulation trajectories for a quantum circuit using a strong simulation scheme."""
    backend = digital_tjm

    if noise_model is None or all(proc["strength"] == 0 for proc in noise_model.processes):
        sim_params.num_traj = 1
    else:
        assert not sim_params.get_state, "Cannot return state in noisy circuit simulation due to stochastics."

    if sim_params.sample_layers:
        dag = circuit_to_dag(operator)
        sim_params.num_mid_measurements = sum(
            1
            for n in dag.op_nodes()
            if n.op.name == "barrier" and str(getattr(n.op, "label", "")).strip().upper() == "SAMPLE_OBSERVABLES"
        )
    for observable in sim_params.sorted_observables:
        observable.initialize(sim_params)

    args = [(i, initial_state, noise_model, sim_params, operator) for i in range(sim_params.num_traj)]


    if parallel and sim_params.num_traj > 1:
        max_workers = max(1, available_cpus() - 1)
        ctx = multiprocessing.get_context("spawn")

        # Batch trajectories to reduce overhead
        batch_size = 4
        batches = list(_chunks(args, batch_size))

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=ctx,
            initializer=_limit_worker_threads,
            initargs=(1,),
        ) as executor:
            futures = {executor.submit(_run_batch, backend, batch): idx for idx, batch in enumerate(batches)}
            with tqdm(total=sim_params.num_traj, desc="Running trajectories", ncols=80) as pbar:
                while futures:
                    done, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                    for future in done:
                        batch_index = futures.pop(future)
                        try:
                            results = future.result()  # list of results for that batch
                        except Exception:
                            # simple retry once for the whole batch
                            new_fut = executor.submit(_run_batch, backend, batches[batch_index])
                            futures[new_fut] = batch_index
                            continue

                        # Write back results for this batch
                        batch = batches[batch_index]
                        for local_j, result in enumerate(results):
                            global_i = batch_index * batch_size + local_j
                            if global_i >= len(args):
                                break
                            for obs_index, observable in enumerate(sim_params.sorted_observables):
                                assert observable.trajectories is not None, "Trajectories should have been initialized"
                                observable.trajectories[global_i] = result[obs_index]
                            pbar.update(1)
    else:
        for i, arg in enumerate(args):
            result = _call_backend(backend, arg)
            for obs_index, observable in enumerate(sim_params.sorted_observables):
                assert observable.trajectories is not None, "Trajectories should have been initialized"
                observable.trajectories[i] = result[obs_index]

    sim_params.aggregate_trajectories()


def _run_weak_sim(
    initial_state: MPS,
    operator: QuantumCircuit,
    sim_params: WeakSimParams,
    noise_model: NoiseModel | None,
    *,
    parallel: bool,
) -> None:
    """Run weak simulation trajectories for a quantum circuit using a weak simulation scheme."""
    backend = digital_tjm

    if noise_model is None or all(proc["strength"] == 0 for proc in noise_model.processes):
        sim_params.num_traj = 1
    else:
        sim_params.num_traj = sim_params.shots
        sim_params.shots = 1
        assert not sim_params.get_state, "Cannot return state in noisy circuit simulation due to stochastics."

    args = [(i, initial_state, noise_model, sim_params, operator) for i in range(sim_params.num_traj)]

    if parallel and sim_params.num_traj > 1:
        max_workers = max(1, available_cpus() - 1)
        ctx = multiprocessing.get_context("spawn")

        batch_size = 4
        batches = list(_chunks(args, batch_size))

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=ctx,
            initializer=_limit_worker_threads,
            initargs=(1,),
        ) as executor:
            futures = {executor.submit(_run_batch, backend, batch): idx for idx, batch in enumerate(batches)}
            with tqdm(total=sim_params.num_traj, desc="Running trajectories", ncols=80) as pbar:
                while futures:
                    done, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                    for future in done:
                        batch_index = futures.pop(future)
                        try:
                            results = future.result()
                        except Exception:
                            new_fut = executor.submit(_run_batch, backend, batches[batch_index])
                            futures[new_fut] = batch_index
                            continue

                        batch = batches[batch_index]
                        for local_j, result in enumerate(results):
                            global_i = batch_index * batch_size + local_j
                            if global_i >= len(args):
                                break
                            # For weak sim, result is the measurement outcome structure
                            sim_params.measurements[global_i] = result
                            pbar.update(1)
    else:
        for i, arg in enumerate(args):
            result = _call_backend(backend, arg)
            sim_params.measurements[i] = result
    sim_params.aggregate_measurements()


def _run_circuit(
    initial_state: MPS,
    operator: QuantumCircuit,
    sim_params: WeakSimParams | StrongSimParams,
    noise_model: NoiseModel | None,
    *,
    parallel: bool,
) -> None:
    """Run circuit-based simulation trajectories."""
    assert initial_state.length == operator.num_qubits, "State and circuit qubit counts do not match."
    operator = copy.deepcopy(operator.reverse_bits())

    if isinstance(sim_params, StrongSimParams):
        _run_strong_sim(initial_state, operator, sim_params, noise_model, parallel=parallel)
    elif isinstance(sim_params, WeakSimParams):
        _run_weak_sim(initial_state, operator, sim_params, noise_model, parallel=parallel)


def _run_analog(
    initial_state: MPS, operator: MPO, sim_params: AnalogSimParams, noise_model: NoiseModel | None, *, parallel: bool
) -> None:
    """Run analog simulation trajectories for Hamiltonian evolution."""
    backend = analog_tjm_1 if sim_params.order == 1 else analog_tjm_2

    if noise_model is None or all(proc["strength"] == 0 for proc in noise_model.processes):
        sim_params.num_traj = 1
    else:
        assert not sim_params.get_state, "Cannot return state in noisy analog simulation due to stochastics."

    for observable in sim_params.sorted_observables:
        observable.initialize(sim_params)

    args = [(i, initial_state, noise_model, sim_params, operator) for i in range(sim_params.num_traj)]
    if parallel and sim_params.num_traj > 1:
        max_workers = max(1, available_cpus() - 1)
        ctx = multiprocessing.get_context("spawn")

        batch_size = 1
        batches = list(_chunks(args, batch_size))

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=ctx,
            initializer=_limit_worker_threads,
            initargs=(1,),
        ) as executor:
            futures = {executor.submit(_run_batch, backend, batch): idx for idx, batch in enumerate(batches)}
            with tqdm(total=sim_params.num_traj, desc="Running trajectories", ncols=80) as pbar:
                while futures:
                    done, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                    for future in done:
                        batch_index = futures.pop(future)
                        try:
                            results = future.result()
                        except Exception:
                            new_fut = executor.submit(_run_batch, backend, batches[batch_index])
                            futures[new_fut] = batch_index
                            continue

                        batch = batches[batch_index]
                        for local_j, result in enumerate(results):
                            global_i = batch_index * batch_size + local_j
                            if global_i >= len(args):
                                break
                            for obs_index, observable in enumerate(sim_params.sorted_observables):
                                assert observable.trajectories is not None, "Trajectories should have been initialized"
                                observable.trajectories[global_i] = result[obs_index]
                            pbar.update(1)
    else:
        for i, arg in enumerate(args):
            result = _call_backend(backend, arg)
            for obs_index, observable in enumerate(sim_params.sorted_observables):
                assert observable.trajectories is not None, "Trajectories should have been initialized"
                observable.trajectories[i] = result[obs_index]
    sim_params.aggregate_trajectories()


def run(
    initial_state: MPS,
    operator: MPO | QuantumCircuit,
    sim_params: AnalogSimParams | StrongSimParams | WeakSimParams,
    noise_model: NoiseModel | None = None,
    *,
    parallel: bool = True,
) -> None:
    """Execute the common simulation routine for both circuit and Hamiltonian simulations."""
    # State must start in B normalization
    initial_state.normalize("B")

    if isinstance(sim_params, (StrongSimParams, WeakSimParams)):
        assert isinstance(operator, QuantumCircuit)
        _run_circuit(initial_state, operator, sim_params, noise_model, parallel=parallel)
    elif isinstance(sim_params, AnalogSimParams):
        assert isinstance(operator, MPO)
        _run_analog(initial_state, operator, sim_params, noise_model, parallel=parallel)

## TODO: Long-range two-site crosstalk noise (factor-only implementation)

Goal: Support long-range two-site jump processes that act on exactly two sites i<j but are not necessarily adjacent, restricted to qubit systems (d=2), without constructing full Kronecker operators of size 2^(j−i+1). Store and use only per-site factors (A at site i, B at site j) throughout.

### 0) Scope and constraints
- Only Pauli-type crosstalk pairs (XX, XY, XZ, YX, YY, YZ, ZX, ZY, ZZ) for d=2.
- Do not build full 4×4 or larger long-range matrices; store and propagate only the tuple of one-site factors `(A, B)`.
- Keep existing nearest-neighbor two-site noise (with 4×4 matrices) fully working and unchanged.

### 1) Library additions: factor-only crosstalk
- File: `src/mqt/yaqs/core/libraries/noise_library.py`
  - Add classes `LongRangeCrosstalkXX`, `LongRangeCrosstalkXY`, `LongRangeCrosstalkXZ`, `LongRangeCrosstalkYX`, `LongRangeCrosstalkYY`, `LongRangeCrosstalkYZ`, `LongRangeCrosstalkZX`, `LongRangeCrosstalkZY`, `LongRangeCrosstalkZZ`.
  - Each class must expose a `factors` attribute with shape `(2×2, 2×2)` referencing the existing Pauli single-site operators (e.g., `(PauliX.matrix, PauliY.matrix)`).
  - Do NOT define a `matrix` attribute on these classes.
  - Register them in `NoiseLibrary` as canonical names (e.g., `longrange_crosstalk_xy = LongRangeCrosstalkXY`).

Done when:
- Importing these names from `NoiseLibrary` yields classes whose instances have `.factors` and no `.matrix`.

### 2) Model: accept factor-only processes
- File: `src/mqt/yaqs/core/data_structures/noise_model.py`
  - Extend `NoiseModel.__init__` to support processes that reference factor-only operators:
    - Retain current behavior for 1-site and nearest-neighbor 2-site operators (which supply or derive `matrix`).
    - For long-range classes (detected via operator class having `factors`), set `process["factors"] = operator_class().factors` and DO NOT attempt to fill `process["matrix"]`.
  - Keep `get_operator` unchanged for matrix-backed operators.

Notes:
- Existing code that uses `process["matrix"]` must not be invoked for long-range processes.

Done when:
- Constructing a `NoiseModel` with `{"name": "LongRangeCrosstalkXY", "sites": [i, j], "strength": γ}` results in processes containing `factors` and no `matrix`.

### 3) Digital: include long-range processes in local scoping
- File: `src/mqt/yaqs/digital/digital_tjm.py`
  - Function: `create_local_noise_model(noise_model, first_site, last_site)`
    - Current: selects 1-site processes within the window and adjacent 2-site pairs `[[i, i+1]]`.
    - Change: include any 2-site process whose both endpoints lie within `[first_site, last_site]` (adjacent or not).
    - Preserve the rest of the behavior.

Done when:
- Test cases for windows include non-adjacent pairs `[u, v]` inside the window and exclude pairs that cross the window boundary.

### 4) Dissipation: endpoint application for factor-only long-range processes
- File: `src/mqt/yaqs/core/methods/dissipation.py`
  - Keep existing code paths:
    - 1-site dissipators use `expm(-0.5 dt γ L†L)` locally.
    - Adjacent 2-site dissipators act on merged bonds via `matrix`.
  - Add support for non-adjacent 2-site processes (those with `"factors"` and distance > 1):
    - During the main reverse sweep `for i in reversed(range(state.length))`:
      - After handling 1-site processes on site `i`, also handle each long-range process endpoint:
        - If `i == sites[0]`, apply `diss_i = expm(-0.5 * dt * γ * (A†A))` with `A = factors[0]` to `state.tensors[i]`.
        - If `i == sites[1]`, apply `diss_j = expm(-0.5 * dt * γ * (B†B))` with `B = factors[1]` to `state.tensors[i]`.
      - Do not attempt to merge tensors for long-range pairs.
    - Keep the subsequent nearest-neighbor bond logic unchanged.

Rationale:
- For Pauli factors, `A†A = I` and `B†B = I`, so the dissipators reduce to scalar rescalings at the endpoints, which is exact and efficient.

Done when:
- Long-range factor-only processes modify only the specified endpoints and do not rely on `process["matrix"]`.
- Nearest-neighbor behavior remains unchanged (tests still pass).

### 5) Stochastic process: weighting and application for long-range
- File: `src/mqt/yaqs/core/methods/stochastic_process.py`
  - `create_probability_distribution`:
    - Preserve current behavior for 1-site and adjacent 2-site processes.
    - Add handling for long-range factor-only processes (len(sites)==2 and distance > 1):
      - Compute `dp_m = dt * γ * state.norm(0)` (state may be non-unit after dissipation; proportionality is sufficient as the routine normalizes across channels).
      - Append entries with:
        - `sites = [i, j]` (ascending)
        - `strengths.append(γ)`
        - Store the factors for later application, e.g., add a parallel list `jump_factors.append((A, B))` (do not put a large `matrix` into `jumps`).
    - Maintain the normalization of probabilities at the end.
  - `stochastic_process`:
    - Remove the nearest-neighbor assertion.
    - Apply by case:
      - 1-site: unchanged.
      - 2-site adjacent with `jump_op` present: keep the existing merge–apply–split path.
      - 2-site non-adjacent: retrieve `(A, B)` from the probability dictionary (e.g., `jump_factors[choice]`) and apply independently to endpoints: `state.tensors[i] ← A @ ...` and `state.tensors[j] ← B @ ...`.
    - Normalize the state to B form at the end (as currently done).

Done when:
- Long-range jumps are included in the sampling distribution and, when selected, update both endpoints without tensor merging.

### 6) Tests
- Add new tests to cover long-range factor-only behavior:
  - Noise library: verify that each `LongRangeCrosstalk..` class exposes correct `factors` using Pauli matrices.
  - `create_local_noise_model`: include processes with endpoints at distance > 1 inside the window; ensure they are included/excluded correctly per window.
  - Dissipation: construct a 3–5 qubit `MPS`, add one long-range process (e.g., `LongRangeCrosstalkXZ` on `[0,2]`), call `apply_dissipation`; check that only endpoints are affected and that canonical center shifting still completes (e.g., `check_canonical_form()[0] == 0`).
  - Stochastic: force a jump (e.g., scale first tensor or set large γ), include a long-range process; confirm both endpoints’ tensors change when a long-range jump is drawn.
  - Regression: re-run existing tests to ensure nearest-neighbor behavior is unaffected.

Done when:
- All new tests pass and existing suites remain green.

### 7) Documentation
- Update docs to state that long-range (non-nearest) two-site Pauli crosstalk is supported via factor-only operators.
- Explain that dissipation and jump application avoid building full Kronecker operators and act only at endpoints.
- Note the current limitation to qubit (2×2) factors; outline how to extend to higher local dimensions.

### 8) Edge cases and safeguards
- Always store `sites` in ascending order; validate input in `NoiseModel.__init__` if necessary.
- Ensure factor shapes match the site physical dimensions (guard against transmon or mixed-d systems).
- Avoid adding `matrix` for factor-only processes so that matrix-based code paths never pick them up accidentally.
- Keep performance by avoiding deep copies where not needed in probability computation (long-range `dp_m` does not require a cloned state).

### 9) Acceptance checklist
- Long-range processes specified as `{name: LongRangeCrosstalk.., sites: [i, j], strength: γ}` are recognized, kept in window scoping, and:
  - Change state through dissipation (endpoint-only) and jumps (endpoint-only) without adjacency assumptions.
  - Do not construct large intermediate Kronecker matrices.
  - Do not affect existing 1-site and nearest-neighbor behavior.

### 10) Follow-ups (optional)
- Consider unifying the representation by adding a tiny helper to expand `(A, B)` into a light-weight MPO across `[i..j]` when needed for other future algorithms.
- If higher local dimensions are needed, extend `NoiseLibrary` to provide correct `d×d` factors and gate scoping to exclude incompatible sites.



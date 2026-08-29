# Next Build Steps

Every item from the original roadmap is closed -- implemented, or dropped with a
stated reason. `TASKS.md` records the outcome of each.

## API notes for Qiskit 2.x

Kept because they are the conventions this project had to learn the hard way, and
each one fails *silently*:

- V1 `Estimator`/`Sampler` no longer exist. Use `StatevectorEstimator` /
  `StatevectorSampler`, or Aer's `EstimatorV2` / `SamplerV2` -- all wrapped by
  `QiskitBackendAdapter`.
- V2 primitives take *pubs*: `estimator.run([(circuit, observable, params)])`,
  read back as `result[0].data.evs`. Sampler results come back under the
  classical register name.
- VQE and QAOA are not in qiskit core. This project runs its own
  `scipy.optimize` loops.
- Aer's `EstimatorV2` seeds only via `run_options["seed_simulator"]`; a plain
  `seed` is accepted and ignored.
- `PauliEvolutionGate` re-synthesises on every estimator call unless decomposed
  up front (2.56s vs 0.006s), and evaluates as the *exact* exponential if left
  undecomposed.

## Where the work could go next

Ordered by value, not by how much is left.

1. **A dataset where a quantum method might actually win.** Every QML result here
   is a tie or a loss, and the Hetionet features may simply be classically easy.
   Finding a task with genuine quantum-friendly structure would test the methods
   rather than the encoding.
2. **Bärtschi-Eidenbenz Dicke preparation.** The current warm start uses generic
   state preparation, depth 272 at `n=6` -- more than the circuit it warms. The
   `O(kn)` construction would make the XY mixer hardware-plausible.
3. **Logistics routing QUBO**, the one remaining scaffold. The QUBO builder and
   QAOA loop are both reusable.
4. **Error mitigation** -- zero-noise extrapolation over the existing noise
   presets, measured against the uncorrected results already recorded.
5. **Larger instances.** Most conclusions here are drawn at 4-8 qubits, where
   classical methods win trivially. Whether any of them survive at 20+ qubits is
   genuinely unknown from this evidence.

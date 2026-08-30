# Benchmark Methodology

How every number in this repository is produced, what it is allowed to mean, and
where it stops being evidence.

## The rule

No quantum method is presented without a classical baseline, and the comparison
that could embarrass it is a named field on the result object rather than a line
of prose. That is a structural choice: a result you have to go out of your way to
compute is a result that quietly gets dropped.

| Field | Answers |
| --- | --- |
| `beats_hartree_fock`, `chemical_accuracy_reached` | did VQE earn its keep against the mean-field reference? |
| `optimal_probability_lift` | did QAOA beat *uniform sampling over feasible states*? 1.0 means no. |
| `quantum_beats_all_classical`, `quantum_vs_rbf` | did the quantum kernel beat every classical model, and by more than fold noise? |
| `matches_brute_force`, `normalized_approximation_ratio` | did the optimiser find the true optimum? |
| `success_probability`, `circuit_depth`, `two_qubit_gates` | what did HHL *cost* to get its answer? |

## What a noise preset is

Three presets, defined in `src/qprac_lab/backends/noise.py` and mirrored into
`configs/noise_model.yaml`, which `tests/test_configs.py` checks against the code
so the two cannot drift.

| Preset | 1-qubit | 2-qubit | Readout | Stands for |
| --- | --- | --- | --- | --- |
| `light` | 1e-4 | 1e-3 | 5e-3 | optimistic near-term hardware, better than most devices today |
| `moderate` | 3e-4 | 6e-3 | 1.5e-2 | roughly a present-day IBM superconducting device |
| `heavy` | 1e-3 | 2e-2 | 4e-2 | a poorly calibrated device, or deep circuits on a mediocre one |

Each is a **depolarizing channel** on every gate in the basis, plus a **symmetric
readout flip** at the stated probability. Rates were chosen to bracket published
superconducting device calibrations rather than to reproduce one machine.

Note that shot noise and device noise are different things and the distinction is
load-bearing. Shot noise shrinks as `1/sqrt(shots)` and vanishes given enough
sampling. Device noise does not. It is the difference between "this algorithm
needs a lot of shots" and "this algorithm does not work on current hardware."

### Transpilation, and a failure that is not loud

Aer attaches noise to *specific gate names*. A circuit containing gates outside
the model's basis has those gates applied **perfectly**, and the run still
succeeds — it simply reports less noise than a device would. This is the exact
shape of bug this project keeps finding: a plausible number rather than an error.
`QiskitBackendAdapter.prepare` transpiles into the noise basis to prevent it, and
the quantum-kernel path additionally injects a noisy sampler *and* a matching pass
manager into `ComputeUncompute`, because `FidelityQuantumKernel` builds its own
compute–uncompute circuits that would otherwise escape the model.

## What the presets do not capture

This is the load-bearing half of the page. A preset lists its error rates, which
makes it easy to over-trust. It omits, deliberately:

- **Coherent error.** Every error here is stochastic. Real miscalibration is
  systematic and *accumulates with circuit depth* rather than averaging out, so a
  deep circuit is punished harder on hardware than these presets suggest.
- **Amplitude damping / T1, T2.** No decoherence-in-time. Gate duration is not
  modelled at all, so a slow circuit and a fast one of equal gate count are
  treated identically. Excluded because it needs gate-duration modelling that
  would obscure the single variable these tutorials isolate.
- **Crosstalk.** Errors are independent per qubit. Simultaneous-gate interference
  is absent.
- **Drift.** Rates are fixed for the whole run. A real device recalibrates, and
  results move between calibrations.
- **Topology.** Noise is `add_all_qubit_*`: every qubit is equally good and
  all-to-all connected. No SWAP overhead from limited connectivity, and no bad
  qubit to route around. On hardware, routing typically *dominates* two-qubit
  count on circuits this shape.
- **Leakage** out of the computational subspace, and **measurement-induced
  state change**.

Taken together these bias every preset **optimistic**. Treat the noise results as
a lower bound on damage, not an estimate of it.

## How the sweep is run

```bash
python scripts/run_noise_sweep.py     # ~6 minutes
```

Four levels — `ideal`, `light`, `moderate`, `heavy` — with **everything else held
fixed**: same seed (42), same ansatz, same optimiser, `maxiter=150`, `shots=2048`,
10 kernel points. Holding the budget fixed is what makes the comparison paired;
the only variable that moves across rows is the noise model.

The budget is deliberately smaller than the tutorials' own defaults. Noisy
simulation is 60–90x slower than statevector because Aer propagates a density
matrix, and a sweep nobody runs is worse than a slightly cheaper one that people
do. **This means sweep numbers are not directly comparable to the headline
tutorial numbers**, which use the full budget. Compare across rows of the sweep,
not between the sweep and the README.

Outputs land in `results/noise_sweep.{json,csv,png}`.

## Which comparisons are legitimate

**Across noise levels, within one tutorial — yes.** This is what the sweep is
designed for. Everything but the noise model is identical.

**Across tutorials — only for shape.** VQE error in hartree and a Max-Cut
approximation ratio are not the same kind of number, and neither is a feasibility
percentage. The cross-tutorial claim this project makes is about *how each
degrades*, never which is larger:

> Methods that need a precise number break first; methods that need only an
> ordering last longest.

VQE needs an energy correct to 1.6 mHa, and misses chemical accuracy at even the
optimistic preset on a two-qubit circuit. Max-Cut needs only the *ranking* of cuts
to survive, and keeps 85% of its ideal quality at `heavy`. That comparison is
about the structure of the requirement, not the magnitude of the metric.

**A structural guarantee is not preserved by noise.** The XY mixer gives exactly
100% feasibility on an ideal simulator, because `(XX+YY)/2` commutes with the
number operator. That is a property of the *ideal unitary*. Measured at `moderate`
it is 46%, and at `heavy` its lift falls below random guessing. A proof about the
algebra is not a claim about the device.

## Run-to-run spread

Report the distribution, not the draw. Where a result is not stable, publish the
range and the cause rather than one number:

- QAOA optimality is a **lottery**. The same `p=6` configuration ranged from 0.1%
  to 100% probability on the optimum depending only on the optimiser's opening
  angles — standard deviation 43 percentage points. `restarts=5` is therefore the
  default, and `restart_objectives` records every attempt so the spread stays
  visible in the result object.
- Classification uses **5x4 repeated stratified CV**, because a single split on
  200 pairs swings ROC-AUC by more than the gap between the models. A ranking is
  only reported as a win when `difference_exceeds_noise` is true; otherwise it is
  a tie, and says so.
- Seeds are fixed and passed explicitly. Aer ignores a plain `seed`, which is one
  of the bugs that produced plausible numbers here — `QiskitBackendAdapter` sets
  `seed_simulator`.

## Metrics by domain

Each is a real field on the returned dataclass, not an aspiration.

**Simulation** — `absolute_error` (Ha), `chemical_accuracy_reached`,
`correlation_recovery_fraction`, `function_evaluations`, `convergence_history`.

**Optimization** — `objective_value`, `normalized_approximation_ratio` (min–max
normalised, because the mean-variance objective can be negative and a raw ratio
is then meaningless), `feasible_probability`, `optimal_probability_lift`,
`constraint_report`, `matches_brute_force`.

**PDEs** — `fidelity` against the classical solve, `success_probability` (HHL's
postselection cost), `circuit_depth`, `two_qubit_gates`, `vqls_cost`, and
`norm_tracking`, because a normalised statevector cannot represent the solution's
scale and it must be carried classically alongside.

**QML** — ROC-AUC / F1 / accuracy per model, `kernel_alignment` as a
pre-training go/no-go, `quantum_vs_rbf` with `difference_exceeds_noise`, and
`mean_self_fidelity`, which reads circuit fidelity directly because
`FidelityQuantumKernel` *assumes* `K(x,x)=1` and skips those circuits by default.

## Is the benchmark easy for the wrong reason?

Measure the shortcut before reporting the win. The Hetionet dataset samples
degree-matched negatives specifically so that node degree alone cannot solve the
task: with matching, degree scores **0.538** ROC-AUC; without it, **0.689**. The
dataset reports both rather than asserting the task is hard.

The earlier synthetic-blob version of this tutorial is the cautionary case. It
showed RBF beating the quantum kernel clearly — and that conclusion turned out to
be an artifact of the generator, not a property of either method.

## What would change these conclusions

**Real device data.** Everything here is simulated, and the presets are
optimistic for the reasons listed above. The first thing a hardware run would
likely overturn is the mapping between preset and reality — expect a device to
land worse than `heavy` on circuits of this depth once routing and coherent error
are included.

If that happens, the presets stay as they are. Retuning them to match a device
after the fact converts a measurement into a fit, which is precisely what this
project exists not to do. The finding goes here instead.

See also
[`cross_framework_verification.md`](cross_framework_verification.md) for how
results are checked against an independent PennyLane implementation, and
[`noise_benchmark.md`](noise_benchmark.md) for the sweep's actual results.

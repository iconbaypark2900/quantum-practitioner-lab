# Benchmark: every tutorial under device noise

Each tutorial's "when not to use this" section can only gesture at hardware from
an ideal simulator. This is the measurement.

```bash
python scripts/run_noise_sweep.py
```

## Noise models

Depolarizing gate errors plus readout error, bracketing current superconducting
hardware. Amplitude damping and crosstalk are deliberately excluded -- they need
gate-duration and topology modelling that would obscure the one variable being
isolated.

| Preset | 1-qubit | 2-qubit | Readout | Roughly |
| --- | --- | --- | --- | --- |
| `light` | 1e-4 | 1e-3 | 5e-3 | optimistic; better than most devices today |
| `moderate` | 3e-4 | 6e-3 | 1.5e-2 | a present-day IBM superconducting device |
| `heavy` | 1e-3 | 2e-2 | 4e-2 | poor calibration, or deep circuits on a mediocre device |

Two things must be true for a noise model to mean anything, and both fail
silently when they are not:

- **The circuit must be transpiled into the noise model's basis.** Aer attaches
  errors to specific gate *names*. A gate outside the basis is applied perfectly
  and the run simply under-reports noise. `QiskitBackendAdapter.prepare()` does
  this.
- **Aer must be seeded via `run_options["seed_simulator"]`.** A plain `seed` is
  accepted and ignored.

## Results

| Noise | VQE error (Ha) | Max-Cut E[ratio] | XY feasibility | XY lift | Kernel self-fidelity |
| --- | --- | --- | --- | --- | --- |
| ideal | `5.6e-10` | 0.886 | **100%** | 19.9x | 1.000 |
| light | `2.2e-03` | 0.877 | 82.7% | 15.7x | 0.936 |
| moderate | `1.1e-02` | 0.824 | 46.2% | 5.8x | 0.734 |
| heavy | `3.7e-02` | 0.756 | 33.2% | 0.83x | 0.388 |

Chemical accuracy is `1.6e-3` Ha. Random guessing scores 0.600 on Max-Cut and
1.00x lift on the portfolio problem.

## What this says

**VQE is the most fragile thing in this repository.** Even the *optimistic*
preset misses chemical accuracy -- `2.2e-3` against a `1.6e-3` threshold -- on a
**two-qubit** circuit of depth ~10. That is about as small as a useful quantum
chemistry circuit can be. The ideal simulator says VQE nails H2 to ten decimal
places; light noise costs seven of them. Any VQE result quoted without a noise
model is a statement about arithmetic, not chemistry.

**Max-Cut is the most robust.** Still 0.756 under heavy noise, comfortably above
the 0.600 random baseline, having lost only 15% of its ideal quality. The reason
is structural: Max-Cut only needs the *ranking* of bitstrings to survive, not
precise amplitudes. Noise flattens the distribution without necessarily reordering
its peak. VQE, by contrast, needs an expectation value accurate to `1e-3`, and
depolarizing noise attacks exactly that.

> **The pattern worth taking away:** algorithms that need a precise number break
> first; algorithms that need only an ordering last longest.

**A structural guarantee is not preserved by noise.** This one deserves emphasis,
because the ideal result is strong enough to invite over-claiming. The XY mixer
gives *exactly* 100% feasible samples on an ideal simulator -- a mathematical
consequence of `(XX+YY)/2` commuting with the number operator, asserted in the
test suite as exact zero infeasible probability.

That guarantee is a property of the **ideal unitary**. Depolarizing and readout
errors move amplitude straight out of the fixed-Hamming-weight subspace, and
feasibility collapses to 46% at moderate noise and 33% at heavy. Worse, the lift
over random feasible guessing falls to **0.83x** under heavy noise -- below 1.0,
meaning the algorithm is now *worse* than picking a feasible portfolio at random.

So the honest reading of the XY mixer: it removes the penalty-versus-feasibility
tradeoff in theory and is clearly the better construction, but on noisy hardware
you still need the feasibility filter that the penalty encoding required. It
changes what you must check, not whether you must check.

**Self-fidelity is free diagnostics.** `K(x, x)` is 1 by definition, so measuring
it reads out directly how much of the circuit survived: 0.936 / 0.734 / 0.388
across the three presets. At heavy noise the compute-uncompute circuit has lost
61% of its fidelity and the kernel is mostly measuring the device.

Two related traps in `FidelityQuantumKernel`, both of which hide noise:

- `evaluate_duplicates="off_diagonal"` (the default) **assumes** the diagonal is
  1 and skips those circuits. Under noise the matrix becomes internally
  inconsistent: a diagonal claiming perfect self-similarity above off-diagonals
  that noise has pulled together.
- `enforce_psd=True` (also the default) silently projects the matrix back to
  positive semi-definite. The measured noisy kernel is genuinely not PSD -- its
  minimum eigenvalue is about `-0.03` -- so this is a real correction being
  applied under the hood, not a formality.

## Reproducing

```bash
python scripts/run_noise_sweep.py     # ~6 minutes
```

Noisy simulation runs 60-90x slower than statevector because Aer propagates a
density matrix, which is why the tutorials default to ideal simulation and the
sweep uses a smaller optimiser budget. Every noise level gets the same budget, so
the comparison stays fair.

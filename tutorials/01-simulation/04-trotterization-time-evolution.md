# Tutorial: Trotterization — simulating time evolution

## Concept

Simulating how a quantum system evolves means applying `exp(-iHt)`. Doing that
directly requires exponentiating a `2ⁿ × 2ⁿ` matrix — precisely the thing a
quantum computer is supposed to avoid.

Product formulas split it into pieces that *are* implementable:

```text
first order  (Lie–Trotter):  exp(-iHt) ≈ [ ∏ₖ exp(-i Hₖ t/r) ]^r
second order (Suzuki):       symmetrised — forward then reverse within each step
```

This is only an approximation because the pieces do not commute. If every `Hₖ`
commuted, the product would be exact at `r = 1` and there would be nothing to
study.

The trade that follows is the entire practical story, and it is why this tutorial
sits next to [the noise benchmark](../05-benchmarking/noise_benchmark.md): **more
steps mean less algorithmic error and more circuit depth.**

## Math intuition

The model is the transverse-field Ising Hamiltonian, the standard testbed:

```text
H = -J Σ Zᵢ Zᵢ₊₁ - h Σ Xᵢ
```

Its two halves are chosen to conflict. `ZZ` is diagonal, `X` is not, so
`[H_ZZ, H_X] ≠ 0` — asserted in the test suite, because a commuting Hamiltonian
would make every measurement below trivially zero.

The leading error terms come from that commutator:

```text
first order:   error ~ t²‖[H_ZZ, H_X]‖ / r      →  falls as 1/r
second order:  the O(t²/r) term cancels          →  falls as 1/r²
```

Second order costs roughly twice the gates per step and buys an extra order.

## ⚠ The trap this tutorial exists to avoid

Evaluate the **decomposed** circuit.

```python
circuit = QuantumCircuit(4)
circuit.append(PauliEvolutionGate(H, time=1.5, synthesis=LieTrotter(reps=1)), range(4))

Operator(circuit)                  # error = 1.0e-15   ← the exact exponential
Operator(circuit.decompose(reps=4))  # error = 1.9      ← the actual Trotter circuit
```

An undecomposed `PauliEvolutionGate` evaluates as `expm(-iHt)` and ignores the
synthesis entirely. A Trotter study built on it reports `1e-15` at every step
count and looks like a spectacular result. It is measuring SciPy.

`trotter_circuit()` returns a decomposed circuit for this reason, and a
regression test pins both numbers.

## Minimal example

```python
from qprac_lab.algorithms.simulation.trotterization import (
    transverse_field_ising_hamiltonian, trotter_operator_error,
)

H = transverse_field_ising_hamiltonian(num_qubits=4)
print(trotter_operator_error(H, time=1.5, steps=1, order=1))   # 1.947
print(trotter_operator_error(H, time=1.5, steps=32, order=2))  # 0.00358
```

## Runnable implementation

```bash
python scripts/run_demo.py --algorithm trotterization
```

## Classical baseline

Dense matrix exponential, `scipy.linalg.expm`. Exact, and the reason this
tutorial can quote errors at all — but it costs `O(8ⁿ)` time and `O(4ⁿ)` memory,
so it stops being available around 15 qubits on a workstation. Everything below
is verified against a baseline that would not exist at interesting problem sizes.

## Benchmark

TFIM, 4 qubits, `J = h = 1`, `t = 1.5`. Error is the spectral norm
`‖U_trotter − U_exact‖₂`, which is state-independent — a fidelity on one chosen
input can flatter the method.

| Steps | 1st-order error | 2nd-order error | Depth (1st) | Depth (2nd) |
| --- | --- | --- | --- | --- |
| 1 | `1.95` | `1.95` | 10 | 20 |
| 2 | `1.65` | `1.20` | 17 | 40 |
| 4 | `6.93e-01` | `2.45e-01` | 31 | 80 |
| 8 | `3.25e-01` | `5.81e-02` | 59 | 160 |
| 16 | `1.60e-01` | `1.44e-02` | 115 | 320 |
| 32 | `7.95e-02` | `3.58e-03` | 227 | 640 |

Fitted scaling exponents: **1.09** (first order) and **2.09** (second order),
against the predicted 1 and 2. Successive error ratios converge to 2.01 and 4.01
respectively — the theory is reproduced to two digits, and this is the cleanest
quantitative agreement anywhere in this repository.

Depth grows linearly with steps, which is the bill for that accuracy.

## Where it stops working

Everything above is an ideal simulator, where more steps is unambiguously better.
Repeating the measurement with a device noise model, tracking total magnetisation
from an all-up initial state:

| Steps | Depth | Ideal error | **Noisy error** |
| --- | --- | --- | --- |
| 1 | 20 | 3.812 | 3.660 |
| 2 | 40 | 0.0270 | **0.0257** ← best |
| 4 | 80 | 0.0173 | 0.0279 |
| 8 | 160 | 0.0053 | 0.0318 |
| 16 | 320 | 0.0014 | 0.0457 |
| 32 | 640 | 0.0003 | 0.0485 |

**The curves separate immediately.** Ideal error falls four orders of magnitude
across the table. Noisy error bottoms out at **2 steps** and then climbs.

At 32 steps the circuit is 100× more accurate in principle and roughly **twice as
wrong** in practice. The optimal step count on hardware is finite, it is small,
and it is nowhere near where the ideal analysis points.

Two things not to misread:

- **The one-step noisy result (3.660) beats its ideal counterpart (3.812).** Noise
  happened to pull a badly-wrong answer toward the exact value. A noisy result
  landing closer to truth is coincidence, not error mitigation.
- **The optimum depends on everything.** Two steps is optimal for *this* model, at
  *this* evolution time, under *this* noise model. It is not a number to reuse; it
  is a calculation to redo.

## Visualization

```bash
python scripts/run_demo.py --algorithm trotterization
```

`results/trotter_tradeoff.png` — power laws on the left, the divergence between
ideal and noisy on the right.

## Real use case

```text
Material or spin model
  → lattice Hamiltonian
  → Trotterised time evolution
  → measure observables (magnetisation, correlations, spectra)
  → compare against experiment
```

Hamiltonian simulation is the application with the least ambiguous quantum
advantage argument: the cost is polynomial in qubit count where exact classical
simulation is exponential. Unlike VQE or QAOA there is no heuristic optimiser in
the loop and no question of whether the algorithm "works" — only how much noise
the circuit can absorb.

## When not to use this

- **When exact simulation fits.** Below ~15 qubits, `expm` is exact and instant.
- **At long evolution times.** Error grows with `t²/r`, so doubling `t` costs
  quadratically more steps to hold accuracy — and every step is depth.
- **On noisy hardware past the optimum**, which the table above puts at 2 steps.
  Adding steps beyond it makes the answer worse while looking like progress.
- **Without checking the decomposition.** See the trap above.

## Source papers

- Lloyd, "Universal Quantum Simulators" (Science, 1996) — Trotterised simulation
  of local Hamiltonians.
- Suzuki, "Fractal decomposition of exponential operators" (1990) — the
  higher-order formulas.
- Childs et al., "Theory of Trotter Error with Commutator Scaling" (PRX, 2021) —
  the tight commutator-based error bounds this tutorial measures against.

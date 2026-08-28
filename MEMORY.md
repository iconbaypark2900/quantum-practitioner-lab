# Project Memory

## Current state

The three priority tutorials plus QAOA Max-Cut are **real quantum
implementations**, not scaffolds. Remaining secondary tutorials (PDEs,
ADAPT-VQE, Trotter, VQC) are still classical scaffolds and are labelled as such
by `qprac-lab list`.

## Stack decision

Qiskit **2.x with V2 primitives only**. Qiskit 2.0 removed the V1
`Estimator`/`Sampler`, and VQE/QAOA no longer ship in qiskit core. This project
implements its own optimisation loops on top of `StatevectorEstimator` /
`StatevectorSampler` (and the Aer equivalents) plus `scipy.optimize`, so it does
not depend on `qiskit-algorithms`.

## Priority

1. VQE for Molecular Energy — done
2. QAOA for Portfolio Selection — done
3. Quantum Kernel for Biomedical Classification — done
4. QAOA Max-Cut — done (unconstrained reference problem)

## Important design decisions

- Every quantum algorithm must include a classical baseline.
- Qiskit is an **optional extra**. The package imports and tests without it;
  quantum code paths raise `QiskitNotInstalledError` with an install hint rather
  than failing deep in a stack trace. CI covers both configurations.
- Results are reported honestly. Two of three tutorials show the quantum method
  failing to beat its classical baseline, and both say so in their headline
  metrics rather than in a footnote.

## Verified reference values

H2 / STO-3G at R = 0.735 A, parity-mapped to 2 qubits. Cross-checked between the
built-in coefficient table and a live PySCF calculation:

```text
electronic energy   -1.857275030 Ha    (what the qubit operator measures)
nuclear repulsion   +0.719968994 Ha    (= 1/R in atomic units)
exact (FCI) total   -1.137306036 Ha
Hartree-Fock total  -1.116998997 Ha
correlation energy  -0.020307039 Ha
```

The electronic/nuclear split is the usual source of confusion when comparing
against published numbers — the qubit operator alone gives -1.857, not -1.137.

## Measured findings worth keeping

- **VQE**: one-parameter UCC ansatz reaches the exact energy to `5.6e-10` in 23
  evaluations; hardware-efficient `efficient_su2` needs 12 parameters and 268.
  At 8192 shots neither reaches chemical accuracy — shot noise is the binding
  constraint, and a noisy estimate can fall *below* the exact energy without
  violating the variational principle.
- **QAOA portfolio**: the penalty encoding is only ~**1.1x** better than uniform
  sampling over feasible portfolios — it learns feasibility and little else,
  because the penalty (6.17) dwarfs the objective spread (~2.1). Lowering it to
  0.5 doubles the lift but drops feasibility to 55%.
- **XY mixer removes that tradeoff.** `(XX+YY)/2` commutes with total Hamming
  weight, so feasibility becomes structural: **exactly 100%** at every depth and
  topology, with no penalty term at all.
- **The 20x optimality lift is NOT robust -- corrected.** It was once documented as
  "reproducible across sampling seeds", which is true and misleading: sampling
  seeds only affect shot noise. Perturbing the optimiser's opening gamma at p=6
  gives 0.1% / 8.4% / 100% / 23% / 100% / 7.1% -- mean 39.8%, s.d. 43.1%, i.e.
  anywhere from 20x *worse* than random to perfect. `restarts=5` is now the
  default for both QAOA tutorials and `restart_objectives` reports the spread.
- **Dicke warm start** (`xy_initial_state="dicke"`) trades peak for reliability:
  better at low depth (p=3: 1.26x vs 0.39x; p=4: 2.31x vs 0.29x), lower peak
  (8.24x vs 20x at p=6), about a third the variance (s.d. 14.5% vs 43.1%). Cost is
  depth -- naive state prep is 272 (n=6,k=3) / 1241 (n=8,k=4), more than the QAOA
  circuit it warms. Baertschi-Eidenbenz O(kn) is the scalable route, not implemented.
- **<C> is not the metric you want.** Restarts keep the best expected cost, which
  is all you can measure without knowing the answer -- but a distribution spread
  over several good portfolios beats one peaked on the best. So restarts made
  P(optimum) *worse* at p=3,4 while improving Max-Cut's expected ratio
  (0.886 -> 0.935), where the metric and the objective coincide.
- **QAOA Max-Cut**: 0.905 expected approximation ratio vs 0.600 for random
  assignment, 49.7% of shots optimal. Greedy also ties the exact optimum
  instantly, which is what an 8-vertex problem looks like.
- **Quantum kernel**: on **real Hetionet** drug--disease link prediction it ranks
  first (0.587 +/- 0.096 vs RBF 0.577 +/- 0.076) but wins only 12/20 paired folds
  — a statistical tie, reported as `difference_exceeds_noise: false`. This
  **reversed** the earlier Gaussian-blob result (RBF 0.826 vs quantum 0.694):
  that conclusion was an artifact of the generator, since blob geometry is near
  ideal for RBF. Kernel-target alignment predicted the ranking correctly in both
  cases (blobs 0.091 vs 0.136; Hetionet 0.027 vs 0.010).
- **Benchmark construction matters more than the algorithm here.** With uniform
  random negatives, node degree alone scores 0.689 — a classifier can look good
  knowing only how well studied a compound is. Degree-matched negatives drop that
  to 0.538 and the biology score from 0.729 to 0.616. The task got harder and the
  numbers got worse, which is what fixing a benchmark looks like.
- **A single split cannot resolve this.** On 80 samples, test ROC-AUC ranged
  0.54-0.85 across split seeds alone — wider than any gap between models. The
  tutorial uses 5x4 repeated CV; the full n x n quantum kernel is computed once
  and every fold reuses submatrices, so 20 evaluations cost the quantum time of
  one.

## Noise findings (scripts/run_noise_sweep.py)

| Noise | VQE error | Max-Cut E[ratio] | XY feasible | XY lift | Kernel self-fid |
| --- | --- | --- | --- | --- | --- |
| ideal | 5.6e-10 | 0.886 | 100% | 19.9x | 1.000 |
| light | 2.2e-03 | 0.877 | 82.7% | 15.7x | 0.936 |
| moderate | 1.1e-02 | 0.824 | 46.2% | 5.8x | 0.734 |
| heavy | 3.7e-02 | 0.756 | 33.2% | 0.83x | 0.388 |

- VQE is the most noise-fragile method here: even the *optimistic* preset misses
  chemical accuracy (2.2e-3 vs 1.6e-3) on a two-qubit depth-10 circuit.
- Max-Cut is the most robust (0.756 at heavy noise vs 0.600 random). General
  pattern: methods needing a precise **number** break first; methods needing only
  an **ordering** survive longest.
- **The XY mixer's 100% feasibility guarantee does not survive noise** -- it is a
  property of the ideal unitary. 46% at moderate, 33% at heavy, with lift falling
  to 0.83x (worse than random feasible guessing). Keep the feasibility filter on
  hardware.
- `FidelityQuantumKernel` hides noise by default twice over:
  `evaluate_duplicates="off_diagonal"` *assumes* K(x,x)=1 instead of measuring it
  (true value 0.734 at moderate noise), and `enforce_psd=True` silently repairs a
  genuinely non-PSD matrix (min eigenvalue about -0.03).

## ADAPT-VQE findings

Given only the Hamiltonian and a generic 6-operator Pauli pool, ADAPT selects
exactly `XY` -- the same generator a human derives from chemistry for H2 -- and
stops after one operator (remaining gradients 3e-5). Its gradient is 0.180931,
exactly the `XX` coefficient, because `XX` is the only term that mixes
determinants and only an XY-type generator can act on it.

- 1 parameter vs 12 for `efficient_su2`; 35 total expectation values (23 optimiser
  + 12 pool scans) vs 268.
- **At R = 2.5 A the fixed ansatz fails**: `efficient_su2` exhausts 300
  evaluations at 2.5e-03 Ha (outside chemical accuracy) while ADAPT reaches
  2.8e-11 with one parameter. Strong correlation is where adaptivity pays.
- **No depth win at this size**: ADAPT depth 10 vs `efficient_su2` depth 8. Fewer
  parameters did not buy a shorter circuit on 2 qubits; the win here is optimiser
  cost and reliability, not depth. Do not claim depth on H2.
- Pool choice decides reachability: restricted to weight-1 operators every
  gradient is zero, ADAPT selects nothing and returns exactly Hartree-Fock.

## VQC findings

Same Hetionet data, same feature map, same 5 folds as the quantum kernel, so the
comparison isolates the learning strategy.

- Four-way tie: VQC 0.600 +- 0.121, RBF 0.580 +- 0.099, quantum kernel
  0.555 +- 0.079, RF 0.471 +- 0.071. VQC ranks first and also has the widest
  spread; paired vs kernel reports `difference_exceeds_noise: false`. The ranking
  order is not stable information.
- **Cost is not a tie**: VQC 96,000 circuit evaluations (linear, every one of 600
  optimiser iterations) vs kernel 19,900 (quadratic, paid once). 4.8x more for an
  indistinguishable result, and the kernel's SVM is convex so it gets its global
  optimum by construction.
- **Barren plateaus measured**: gradient variance 6.85e-2 / 1.66e-2 / 6.89e-3 /
  1.51e-3 at 2/4/6/8 qubits -- roughly 4x decay per two qubits, exponential in
  width. The quantum kernel has no trainable circuit and therefore no barren
  plateau; that is the sharpest structural difference between the two.
- Batching is what makes training feasible: V2 primitives take a parameter
  *array*, so 200 samples go in one pub (337 ms) instead of 200 calls. A test
  asserts batched == per-sample, since a broadcasting bug produces
  plausible-looking garbage.

## Trotterization findings

TFIM (4 qubits, J=h=1, t=1.5), error = spectral norm vs exact `expm`:

- Fitted scaling exponents **1.09** (first order) and **2.09** (second order)
  against theory's 1 and 2; successive error ratios converge to 2.01 and 4.01.
  Cleanest quantitative agreement in the repo.
- **Noise reverses the trade.** Ideal error falls 3.81 -> 0.0003 across 1..32
  steps; noisy error bottoms out at **2 steps** (0.0257) then climbs to 0.0485.
  At 32 steps the circuit is 100x more accurate in principle and ~2x more wrong
  in practice. The optimum is finite, small, and problem-specific.
- The 1-step noisy result (3.660) beats its ideal counterpart (3.812) -- noise
  pulling a badly-wrong answer toward truth. Coincidence, not mitigation.

**Trap:** `Operator(circuit)` on an *undecomposed* `PauliEvolutionGate` returns
the exact matrix exponential and ignores the synthesis -- 1.0e-15 error at every
step count. Decomposed, the same reps=1 circuit has error 1.9. Any Trotter study
skipping the decomposition is benchmarking SciPy. `trotter_circuit()` returns
decomposed; a regression test pins both numbers.

**Observable choice matters:** |+...+> is an eigenstate of the TFIM's spin-flip
symmetry, so magnetisation stays pinned at zero and every step count scores an
identical, meaningless error. Use |0...0>.

## PDE findings

**HHL** (2x2, eigenvalues 1 and 2, kappa=2, 2 clock qubits): fidelity exactly
1.000000000000 vs the classical solve, P(ancilla=1) = 0.625, depth 86.

- Eigenvalues must land on integer clock values. Varying only `t`: fidelity
  1.000 / 0.631 / 0.481 / 0.444. In practice you do not know the spectrum in
  advance -- that is what you are computing -- and a real incommensurate spectrum
  has no exact `t` at all.
- kappa penalty is visible: kappa 2/4/8 -> fidelity 1.000/0.984/0.558,
  P(success) 0.625/0.421/0.098.
- Readout is the killer: O(N/eps^2) shots. 20,000 shots to read a **2-element**
  solution to 1%. Reading the vector throws away the exponential speedup, so HHL
  is only useful for summary statistics like <x|M|x>.

**Variational heat equation** (VQLS): fidelity 0.999999999, depth **10** vs HHL's
86, no postselection -- but non-convex, so restarts are the default.

- **The normalisation caveat is physics, not bookkeeping.** ||u|| goes 1.870829
  -> 1.837448 in one step; that 0.0334 *is* the heat leaving the system, and a
  normalised |psi> cannot carry it. Recovered from the state: exactly 0.0. Any
  quantum PDE solver tracks the scale classically on the side.

**Black-Scholes**: FD matches the closed form to <1% (10.3886 vs 10.4506 at the
money). The number worth keeping: a variational solve would cost ~**2,000,000
circuit evaluations** against **one** call to a formula. Where quantum finance
actually argues advantage is amplitude estimation for path-dependent Monte Carlo
-- quadratic, not exponential, and a different algorithm.

## Cross-framework verification (PennyLane)

Replaces the IBM Runtime adapter: no credentials, CI-testable, and -- decisively --
it exercises a *different* stack, where Runtime would have re-run the same Qiskit
code and verified nothing.

- VQE agrees with Qiskit to 4.4e-16 at R = 0.735/1.0/2.5 A.
- QUBO -> Ising agrees over all 64 assignments to 1.8e-14.
- PennyLane implementations are written **natively**, not translated -- a
  translation carries over the convention errors it is meant to detect.
- **Ordering trap**: Qiskit labels are little-endian (leftmost char = highest
  qubit), PennyLane addresses wires explicitly, so char `i` maps to wire
  `n-1-i`. A flipped mapping gives a mirror-image Hamiltonian with an *identical
  spectrum*, so eigenvalue tests all pass. Only an asymmetric operator catches
  it; the test suite pins one.
- Limit: agreement rules out shared *implementation* bugs, not shared
  *conceptual* ones. If the Hamiltonian is wrong both reproduce it faithfully --
  hence the separate checks against published FCI values.

## Performance trap worth remembering

`QAOAAnsatz` holds `PauliEvolutionGate`s whose synthesis is redone on **every**
estimator call. Measured at 2.56s per call versus 0.006s after a one-time
`.decompose(reps=3)` on an 8-qubit graph — a ~400x difference that turned a
2-second optimisation into 13 minutes. `run_qaoa` decomposes up front.

Similarly, Aer's `EstimatorV2` seeds only via `run_options["seed_simulator"]`;
a plain `seed` key is accepted and ignored, leaving runs irreproducible.

Two more Aer traps: noise attaches to gate *names*, so an untranspiled circuit
has out-of-basis gates applied perfectly and under-reports noise; and Aer cannot
execute `XXPlusYYGate` at all (`AerError: unknown instruction: xx_plus_yy`).
`QiskitBackendAdapter.prepare()` transpiles for both reasons, on any Aer run.
Noisy simulation is 60-90x slower than statevector (density matrix), which is why
the tutorials default to ideal and the sweep uses a smaller optimiser budget.

## Dataset provenance

Hetionet v1.0 (Himmelstein et al., eLife 2017), CC0. Cached to `data/raw`
(gitignored, ~12 MB) by `python scripts/download_data.py`; override the location
with `QPRAC_DATA_DIR`. Target edge type `CtD` (Compound-treats-Disease), 755
positives. Features use only compound--gene and gene--disease edges — `CtD` and
`CpD` (palliates) are both excluded, so there is no label path to mask per split.

## Next milestone

- Notebook walkthroughs for the finished tutorials.
- Dicke-state warm start for the XY mixer, to replace the single k-hot basis
  state and reduce the depth sensitivity seen above.
- IBM Runtime and CUDA-Q adapters.

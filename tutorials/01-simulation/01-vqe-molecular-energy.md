# Tutorial 1: VQE for Molecular Energy

## Concept

The ground-state energy of a molecule is the lowest eigenvalue of its electronic
Hamiltonian. Exact diagonalisation costs memory exponential in the number of
orbitals, so it stops being possible somewhere around 20 qubits on a workstation.

VQE sidesteps the eigenvalue problem. It prepares a parameterised trial state
`|psi(theta)>` on a quantum computer, measures `<psi(theta)|H|psi(theta)>`, and
hands that single number to a classical optimiser which adjusts `theta`. The
quantum computer never diagonalises anything -- it only evaluates expectation
values. That division of labour is what makes VQE *hybrid*, and it is why VQE
can run on shallow, noisy hardware.

The variational principle is the guarantee that makes this sound:

```text
<psi(theta)|H|psi(theta)>  >=  E_ground     for every theta
```

You can never accidentally go below the true answer, so lower is always better.

## Math intuition

For H2 in an STO-3G basis, a parity mapping with two-qubit reduction gives a
Hamiltonian on just **two qubits**:

```text
H = -1.052373 II + 0.397937 IZ - 0.397937 ZI - 0.011280 ZZ + 0.180931 XX
```

Two things about this operator matter and are routinely got wrong:

1. **It is the electronic Hamiltonian only.** Diagonalising it gives
   `-1.857275 Ha`, not the `-1.137 Ha` you see quoted for H2. The nuclear
   repulsion energy has to be added back:

   ```text
   E_total = E_electronic + E_nuclear      E_nuclear = 1/R  (atomic units)
           = -1.857275 + 0.719969
           = -1.137306 Ha
   ```

2. **The `XX` term is the whole problem.** The `Z` terms are diagonal and
   classical; a computational basis state diagonalises them. Only `XX` mixes
   determinants, and that mixing *is* electron correlation. Drop it and you get
   Hartree-Fock.

## Minimal example

The Hartree-Fock state is the computational basis state `|01>`, and its energy
is a one-line calculation:

```python
from qiskit.quantum_info import Statevector
from qprac_lab.algorithms.simulation.hamiltonian_utils import h2_hamiltonian_builtin

h = h2_hamiltonian_builtin()
print(h.hartree_fock_total_energy())   # -1.116999  (mean field, no correlation)
print(h.exact_total_energy())          # -1.137306  (exact diagonalisation)
print(h.correlation_energy())          # -0.020307  (the 20 mHa VQE must recover)
```

VQE's entire job is to close that 20 mHa gap.

## Runnable implementation

```bash
python scripts/run_demo.py --algorithm vqe_molecular_energy
```

The ansatz is the one-parameter UCC circuit: start in `|01>`, then apply a
single excitation `exp(-i theta/2 * X0 Y1)`. One parameter is enough because the
relevant subspace of two-qubit H2 is two-dimensional.

```python
from qprac_lab.algorithms.simulation.vqe_molecular_energy import run_vqe_molecular_energy_tutorial

result = run_vqe_molecular_energy_tutorial()
print(result.vqe_energy, result.absolute_error, result.chemical_accuracy_reached)
```

At `theta = 0` the circuit *is* the Hartree-Fock state, so the optimisation
provably starts at the HF energy and can only improve from there.

## Classical baselines

Both are mandatory, and they answer different questions.

| Baseline | Energy (Ha) | What it tells you |
| --- | --- | --- |
| Hartree-Fock | `-1.116999` | The bar to beat. Below this, VQE is doing something. |
| Exact diagonalisation | `-1.137306` | The floor. Below this, VQE is doing something *wrong*. |

## Benchmark

Measured on a statevector simulator, R = 0.735 A:

| Ansatz | Backend | Shots | Parameters | Evaluations | Error (Ha) | Chemical accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| `two_qubit_uccsd` | statevector | exact | 1 | 23 | `5.6e-10` | yes |
| `efficient_su2` | statevector | exact | 12 | 268-300 | `3e-10` to `5e-09` | yes |
| `two_qubit_uccsd` | statevector | 8192 | 1 | 23 | `3.4e-03` | **no** |
| `two_qubit_uccsd` | aer | 8192 | 1 | 23 | `3.4e-03` | **no** |

The two 8192-shot rows agree exactly, which is the expected result: same seed,
same shot budget, two simulators that should therefore draw the same samples.
They only agree because the Aer adapter seeds through
`run_options["seed_simulator"]` -- Aer's `EstimatorV2` silently ignores a plain
`seed` option, which leaves runs unseeded and irreproducible.

Read the bottom two rows carefully -- they are the useful ones.

- **Shot noise, not the algorithm, is the binding constraint.** The exact rows
  reach chemical accuracy with room to spare. At 8192 shots the same circuit and
  the same optimiser miss it by 2-7x. Estimator precision scales as
  `1/sqrt(shots)`, so buying another decimal place costs 100x the shots.
- **A noisy estimate can dip below the exact energy.** The Aer row lands at
  `-1.149 Ha`, which is *below* the true ground state. That does not violate the
  variational principle: the principle bounds the true expectation value, not a
  finite-shot estimate of it. Treating a sampled energy as a guaranteed upper
  bound is a real and common mistake.
- **Chemistry knowledge buys a ~12x cheaper optimisation.** The problem-informed
  ansatz needs 1 parameter and 23 evaluations where the hardware-efficient one
  needs 12 parameters and roughly 270-300, for the same answer.

### A note on reproducibility

The `two_qubit_uccsd` rows reproduce exactly, run after run. The `efficient_su2`
row does not: it bifurcates between 268 evaluations (error `3.4e-10`) and 300
(error `4.9e-09`) depending on `PYTHONHASHSEED`, which Python randomises per
process. Fix it (`PYTHONHASHSEED=0`) and the result is stable.

The cause is dict/set iteration order somewhere in the multi-parameter circuit
construction, which changes floating-point summation order by the last bit or
two. Near a flat 12-parameter optimum that is enough to send COBYLA down a
different path. Both outcomes sit far inside chemical accuracy, so nothing about
the physics changes -- but it is why this tutorial publishes a range for that row
and why the test suite asserts on energies rather than on iteration counts.

## Visualization

```bash
python scripts/run_first_three_tutorial_outputs.py
```

- `results/vqe_energy_convergence.png` -- convergence with the exact and HF
  reference lines, plus a log-scale error panel. The linear panel alone is
  nearly useless: the optimiser's early exploration compresses the converged
  region onto the reference line.
- `results/vqe_dissociation_curve.png` -- the full potential energy surface.

## Real use case

Materials and drug discovery screening pipelines:

```text
Candidate structures
  -> classical ML surrogate screening (cheap, approximate)
  -> DFT / Hartree-Fock refinement
  -> VQE refinement where correlation dominates
  -> stability and energy ranking
  -> lab validation
```

The dissociation curve shows exactly where that last step earns its cost:

| R (A) | Exact | Hartree-Fock | HF error |
| --- | --- | --- | --- |
| 0.50 | `-1.055160` | `-1.042996` | `0.012` |
| 0.735 | `-1.137306` | `-1.116999` | `0.020` |
| 1.00 | `-1.101150` | `-1.066109` | `0.035` |
| 1.50 | `-0.998149` | `-0.910874` | `0.087` |
| 2.50 | `-0.936055` | `-0.702944` | `0.233` |

Near equilibrium, HF is off by 20 mHa and is often good enough. At 2.5 A it is
off by 233 mHa -- a factor of 12 worse, and qualitatively wrong about a breaking
bond. Restricted Hartree-Fock cannot describe bond dissociation at all, because
a single determinant cannot represent the two-configuration character of a
stretched bond. VQE tracks the exact curve to `1e-10` everywhere.

**Bond breaking, transition states, and stretched geometries are where
correlated methods stop being optional.** Equilibrium geometries usually are not.

## When not to use this

- **Near equilibrium with a cheap alternative available.** For H2 at 0.735 A,
  CCSD(T) gets the same answer in milliseconds on a laptop. VQE is interesting
  where classical correlated methods scale badly, not where they work fine.
- **When you need many digits.** Shot noise sets a `1/sqrt(shots)` floor on
  precision. Chemical accuracy on a real device is a research problem, not a
  configuration setting.
- **Large hardware-efficient ansaetze.** The `efficient_su2` run needed 12
  parameters for 2 qubits. That growth, plus barren plateaus, is the central
  open obstacle to scaling VQE.

## Source papers

- Peruzzo et al., "A variational eigenvalue solver on a photonic quantum
  processor" (2014) -- introduces VQE.
- O'Malley et al., "Scalable quantum simulation of molecular energies" (2016) --
  the two-qubit H2 reduction and one-parameter ansatz used here.
- Grimsley et al., "An adaptive variational algorithm for exact molecular
  simulations on a quantum computer" (2019) -- ADAPT-VQE, which builds the ansatz
  operator by operator instead of fixing it up front.

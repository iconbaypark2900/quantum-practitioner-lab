# Tutorial 1: VQE for Molecular Energy

## Use case

Materials discovery refinement.

## Algorithm type

Hybrid variational eigensolver.

## Classical baseline

- Exact diagonalization
- Hartree-Fock reference energy

## Source papers

- Peruzzo et al., "A variational eigenvalue solver on a photonic quantum processor"
- Grimsley et al., "An adaptive variational algorithm for exact molecular simulations on a quantum computer"

## Required output

- Energy convergence plot
- Final energy estimate
- Exact diagonalization comparison
- Hartree-Fock reference comparison
- Error table
- Circuit depth
- Optimizer iteration count

## Build flow

```text
Molecule
  → fermionic Hamiltonian
  → qubit Hamiltonian
  → ansatz circuit
  → estimator
  → optimizer
  → energy convergence plot
  → benchmark report
```

## Real use case

```text
Candidate materials
  → ML surrogate screening
  → VQE refinement
  → stability / energy ranking
  → human review or lab validation
```

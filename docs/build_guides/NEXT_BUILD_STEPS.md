# Next Build Steps

## Step 1: Upgrade VQE

Replace toy energy function with:

- Qiskit SparsePauliOp Hamiltonian
- Estimator primitive
- optimizer callback
- exact diagonalization comparison

## Step 2: Upgrade QAOA

Replace baseline-only portfolio selection with:

- QUBO matrix
- Qiskit optimization problem
- QAOA ansatz / sampler
- sampled bitstring report

## Step 3: Upgrade Quantum Kernel

Replace classical kernel preview with:

- ZZFeatureMap
- FidelityQuantumKernel
- QSVC
- kernel matrix visualization

## Step 4: Add PDE notebooks

Implement:

- HHL intro
- finite difference baseline
- variational heat equation scaffold

# Tasks

## Phase 1: Lock full scaffold

- [x] Add full module structure
- [x] Add first three priority tutorials
- [x] Add PDE module
- [x] Add benchmarking module
- [x] Add source-paper library
- [x] Add configs
- [x] Add notebooks placeholders
- [x] Add backend adapter placeholders

## Phase 2: Implement first three tutorials

All three now run real quantum circuits through Qiskit 2.x V2 primitives.

### VQE for Molecular Energy

- [x] Add scaffold implementation
- [x] Add convergence plot output
- [x] Add Qiskit molecular Hamiltonian support (built-in table + PySCF/Nature path)
- [x] Add exact diagonalization baseline
- [x] Add Hartree-Fock baseline note/result
- [x] Add optimizer history (real objective-evaluation trace, not a parameter sweep)
- [x] Add real Estimator loop (one-parameter UCC and hardware-efficient ansaetze)
- [x] Add dissociation curve showing where Hartree-Fock fails
- [x] Add notebook walkthrough
- [ ] Add ADAPT-VQE ansatz construction

### QAOA for Portfolio Selection

- [x] Add scaffold implementation
- [x] Add brute-force baseline
- [x] Add greedy baseline
- [x] Add constraint report
- [x] Add simulated annealing baseline
- [x] Add QUBO builder (with exhaustive QUBO/Ising equivalence tests)
- [x] Add Qiskit QAOA/Sampler implementation
- [x] Add sampling-quality metrics (feasible rate, lift over uniform)
- [x] Add notebook walkthrough
- [x] Add XY mixer to preserve the cardinality constraint by construction

### Quantum Kernel for Biomedical Classification

- [x] Add scaffold implementation
- [x] Add RBF-SVM baseline
- [x] Add Random Forest baseline
- [x] Add kernel matrix preview
- [x] Add optional XGBoost baseline (`pip install -e ".[xgboost]"`)
- [x] Add Qiskit FidelityQuantumKernel
- [x] Add QSVC (via `SVC(kernel="precomputed")`, which is what QSVC wraps)
- [x] Add kernel-target alignment as a pre-training go/no-go check
- [x] Add biomedical KG dataset loader (real Hetionet CtD link prediction)
- [x] Add degree-matched negative sampling and a leakage diagnostic
- [x] Replace the single train/test split with repeated cross-validation
- [x] Add notebook walkthrough

## Phase 3: Implement secondary tutorials

- [ ] ADAPT-VQE
- [x] Hamiltonians and expectation values (`hamiltonian_utils`)
- [ ] Trotterization
- [x] QAOA Max-Cut (+ notebook walkthrough)
- [x] QUBO / Ising mapping
- [ ] HHL intro
- [ ] Variational heat equation
- [ ] Black-Scholes PDE
- [ ] VQC classifier
- [ ] QML link prediction

## Phase 4: Backend expansion

- [x] Qiskit backend adapter (statevector + Aer, V2 primitives, shot control)
- [ ] IBM Runtime backend adapter
- [ ] CUDA-Q backend adapter
- [x] Noise model support (depolarizing + readout presets: light/moderate/heavy)
- [x] Benchmark all tutorials against ideal/noisy simulation (`scripts/run_noise_sweep.py`)

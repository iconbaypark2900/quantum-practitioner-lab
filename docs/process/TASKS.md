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
- [x] Add ADAPT-VQE ansatz construction

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

- [x] ADAPT-VQE
- [x] Hamiltonians and expectation values (`hamiltonian_utils`)
- [x] Trotterization
- [x] QAOA Max-Cut (+ notebook walkthrough)
- [x] QUBO / Ising mapping
- [x] HHL intro
- [x] Variational heat equation
- [x] Black-Scholes PDE
- [x] VQC classifier
- [x] QML link prediction (Hetionet CtD, shared by the kernel and VQC tutorials)

## Phase 4: Backend expansion

All items resolved -- two implemented, two deliberately dropped with reasons.

- [x] Qiskit backend adapter (statevector + Aer, V2 primitives, shot control)
- [x] Dicke-state warm start for the XY mixer (+ optimiser restarts)
- [x] PennyLane backend adapter (replaces the IBM Runtime item: no credentials
      needed, open source, and CI can actually test it)
- [x] Cross-framework verification of VQE and the QUBO/Ising mapping
- [x] ~~CUDA-Q backend adapter~~ — dropped: no GPU access on the target machine
- [x] ~~IBM Runtime backend adapter~~ — dropped: needs credentials, cannot be
      CI-tested, and drives the same Qiskit stack. Replaced by PennyLane.
- [x] Noise model support (depolarizing + readout presets: light/moderate/heavy)
- [x] Benchmark all tutorials against ideal/noisy simulation (`scripts/run_noise_sweep.py`)

## Phase 5: Publish and maintain

The repository went public on 2026-08-30 under Apache-2.0.

- [x] Apache-2.0 licence, PEP 639 metadata, description, topics, CI badges
- [x] CITATION.cff
- [x] Migrate off the Qiskit APIs 3.0 removes (`efficient_su2`, `real_amplitudes`,
      `qaoa_ansatz`, explicit `Gate.control(annotated=)`) — project-code
      deprecations 5 → 0, every published number held, QAOA depth 70 → 28
- [x] `forward-compat` workflow: weekly pre-release check plus a deprecation
      inventory, advisory only, and honest about the case where no pre-release
      exists
- [x] Delete or wire in every unimported module — found a latent bug in
      `exact_lowest_eigenvalue` doing it (0.0 for Pauli-Y instead of -1.0)
- [x] State the operating range in the README (2–8 qubits, simulation only)
- [x] Write `benchmark_methodology.md`, including what the presets do **not** model
- [x] Collapse five paper stores into one generated from `configs/papers.yaml`;
      16 papers, every identifier verified to resolve
- [x] Rewrite the nine stub tutorial pages; add `tests/test_doc_links.py` so a
      README cannot advertise a tutorial that does not exist
- [x] Move process docs to `docs/process/`; bump actions off Node 20
- [x] Prepare `scripts/run_hardware_vqe.py` (one-off, run by hand, `--dry-run`
      exercises it without credentials)
- [ ] **Run it.** Needs an IBM Quantum account; no credential belongs in this repo.

### Known, and not ours to fix

`qiskit-nature` constructs a `BlueprintCircuit` in `hartree_fock.py`, which
Qiskit 3.0 removes. Only the optional `[nature]` extra reaches it, so core and
`[qiskit]` installs are 3.0-clean. Watched by `forward-compat`.


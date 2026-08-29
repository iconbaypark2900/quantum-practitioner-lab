# Next Build Steps

Steps 1-3 of the original guide are **done**. It was written against the Qiskit
V1 primitives, which Qiskit 2.0 removed; what follows targets the V2 API the
project now uses.

## API notes for Qiskit 2.x

- `qiskit.primitives.Estimator` / `Sampler` (V1) **no longer exist**. Use
  `StatevectorEstimator` / `StatevectorSampler`, or `qiskit_aer.primitives`
  `EstimatorV2` / `SamplerV2`. All of these are wrapped by
  `qprac_lab.backends.qiskit_adapter.QiskitBackendAdapter`.
- V2 primitives take *pubs*: `estimator.run([(circuit, observable, params)])`,
  read back with `result[0].data.evs`. Sampler results come back under the
  classical register name, e.g. `result[0].data.meas.get_counts()`.
- VQE and QAOA are **not** in qiskit core any more. This project runs its own
  `scipy.optimize` loops rather than depending on `qiskit-algorithms`.
- Estimator precision replaces shot count: `precision = 1/sqrt(shots)`, and
  `precision = 0.0` means exact expectation values.

## Step 1: Notebook walkthroughs

`notebooks/01..03` are still placeholders. Each should narrate the tutorial it
matches, reusing the library functions rather than re-implementing them.

## Step 2: Constraint-preserving QAOA

The penalty encoding measured only a 1.12x lift over uniform sampling of
feasible portfolios. Implement an **XY mixer** that confines the state to the
fixed-cardinality subspace, so feasibility holds by construction and the
optimiser spends its capacity on the objective instead.

## Step 3: A real biomedical dataset

`make_biomedical_pair_features` returns `make_classification` Gaussian blobs —
precisely the geometry an RBF kernel handles best, which stacks the deck against
the quantum kernel. Add a loader for genuine KG-derived pair features.

## Step 4: Noise models

Shot noise is supported (`shots=` on the adapter). Device noise is not. Add Aer
noise models to `QiskitBackendAdapter`, then re-run the benchmarks to measure how
far each tutorial degrades under realistic error rates.

## Step 5: Promote the secondary scaffolds

The QUBO/Ising machinery and the QAOA loop are both reusable, so **QAOA Max-Cut**
is the cheapest next real implementation — `maxcut_qubo` already exists and is
tested. After that: Trotterization, ADAPT-VQE, VQC.

## Step 6: Backends

IBM Runtime and CUDA-Q adapters, behind the same `QiskitBackendAdapter`
interface.

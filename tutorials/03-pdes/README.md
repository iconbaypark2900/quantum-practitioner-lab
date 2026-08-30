# 03 Quantum PDEs

Linear systems and variational residual minimisation. This is the section with the
widest gap between the theoretical speedup and what you can actually collect, and
the tutorials measure that gap rather than mentioning it.

## Tutorials

| # | Tutorial | What it shows |
| --- | --- | --- |
| 1 | [HHL Linear Systems Intro](01-hhl-linear-systems-intro.md) | The exponential speedup on paper, then every caveat measured: state preparation, conditioning, and readout. Fidelity 1.0 against the classical solve — at **depth 86**, 40 two-qubit gates, and a **62.5% postselection success rate** you pay on every shot. |
| 2 | [Variational Heat Equation](02-variational-heat-equation.md) | VQLS at **depth 10 against HHL's 86**, fidelity `0.999999999` to the exact direction — and the thing it structurally cannot represent: the solution's *norm*. Diffusion loses `0.033381` of it per step, and that has to be tracked classically alongside. |
| 3 | [Black-Scholes PDE](03-black-scholes-pde.md) | The cost comparison stated plainly: **≈2,000,000 circuit evaluations** for the full variational solve, against one closed-form evaluation of a European call. |

## Use case

[Financial PDE pricing](use_cases/financial_pde_pricing.md).

## Not covered here

A **Poisson equation** tutorial does not exist. The variational machinery it would
use is already shown by the heat equation, and the caveat that matters — a
normalised statevector cannot carry the solution's scale — is the same one.

## Papers

[Source papers for this section](papers.md).

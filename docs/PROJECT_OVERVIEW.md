# Project Overview

TransSolveStack is a Taichi/GPU-first platform for sparse linear-system solver
selection. It is organized around measured GPU solver evidence, selector-data
generation, learned-policy contracts, and guarded runtime execution.

## The Problem

Large sparse systems appear across CAE, CFD, FEA, graphics, and HPC workloads.
Solver performance depends on the matrix structure, precision, preconditioner,
stopping criteria, and fallback route. A static default solver is often fragile,
while a learned selector is unsafe unless it is trained from measured evidence
and guarded at runtime.

TransSolveStack builds the engineering stack around that idea:

```text
matrix evidence -> GPU solver profiles -> selector rows -> learned policy ->
quality gate -> runtime guard -> fallback chain
```

## Basic Structure

| Layer | Role |
|---|---|
| Dataset layer | Plans downloads, indexes SuiteSparse metadata, selects bounded subsets, probes archive headers, and imports CSR fixtures. |
| Diagnostic layer | Computes selector-oriented matrix facts such as shape, symmetry, diagonal health, and precision hints. |
| GPU runtime layer | Executes Taichi CSR operators, Krylov solvers, preconditioners, residual checks, and smoke benchmarks. |
| Artifact layer | Stores solver outcomes, manifests, summaries, schemas, and reproducibility reports. |
| Policy-data layer | Converts diagnostics and solver evidence into selector rows, learning rows, model contracts, tensors, and external training packages. |
| Guard layer | Keeps learned predictions behind quality gates, shadow mode, confidence thresholds, fallback-chain enforcement, and runtime guards. |

## Integrated Solvers

TransSolveStack treats solver integration as a full path from GPU execution to
selector evidence, not just a callable algorithm. The integrated CSR solver
families include:

- CG and PCG for symmetric systems.
- BiCGSTAB for general/nonsymmetric systems.
- Restarted GMRES for nonsymmetric fallback candidates.
- Weighted Richardson as a lightweight baseline and failure-mode probe.
- Chebyshev/Jacobi when usable spectral bounds are available.
- Preconditioner and scaling variants including Jacobi, ILU0, symmetric
  diagonal equilibration, and row/column equilibration probes.

Each solver family can contribute successes, failures, applicability rows, and
negative evidence to the selector pipeline.

## Runtime Guard

The guard layer protects runtime execution from unsafe learned-policy outputs.
It combines:

- shadow mode,
- offline quality-gate eligibility,
- confidence thresholds,
- exact matrix/context profiled-success checks,
- fallback-chain enforcement,
- post-attempt timeout handling,
- guarded auto-solve metadata.

The design goal is to let learned policies improve solver selection without
letting unverified predictions bypass numerical evidence or fallback safety.

## Design Principles

- GPU execution is the first-class runtime target.
- CPU code is allowed for import, screening, diagnostics, and reference checks.
- Every solver or preconditioner claim needs a functional or numeric gate.
- Learned policy outputs stay shadowed until quality and runtime guards pass.
- Full datasets and heavy artifacts stay outside the public repository.

## Current Capabilities

- SuiteSparse Matrix Market planning, indexing, selection, and bounded CSR
  import.
- Taichi GPU CSR matvec, residual, dot, norm, and vector primitives.
- Taichi GPU CSR solve evidence for CG/PCG, BiCGSTAB, GMRES, Richardson,
  Chebyshev, symmetric equilibration, row/column equilibration, and ILU0.
- Selector artifacts covering successes, failures, non-success fallback rows,
  applicability rows, and unresolved matrix diagnostics.
- Transformer-ready tensor bundle, training entrypoint, external training package, and consumer-side model submission dry run.
- Offline quality gate for learned selector candidates.
- Learned runtime guard with shadow/promotion boundaries, confidence checks,
  fallback enforcement, and guarded auto-solve integration.
- Public API smoke artifacts and release hygiene checks.

## Advantages

- Matrix and solver evidence are kept together instead of separated across
  notebooks, logs, and one-off scripts.
- Solver selection can evolve from artifact-backed rules to learned policies
  without skipping safety checks.
- Negative evidence is first-class: failed numerical gates, structural
  non-applicability, and unresolved fallback cases are preserved for training
  and diagnostics.
- GPU implementation work is tied to verifiable artifacts, making claims easier
  to reproduce and audit.
- The same pipeline can support lightweight public smoke tests and larger
  external benchmark campaigns.

## Development Boundary

- The public repository keeps only lightweight reproducibility artifacts.
- Full benchmark campaigns and large datasets live outside git.
- Learned selector promotion is guarded; raw model predictions are not accepted
  as runtime decisions without quality and fallback checks.
- Commercial use is permission-only under the selected license route.

## Where To Look

| Question | File |
|---|---|
| What can I run quickly? | `README.md` |
| What does the project provide? | `README.md` |
| What is the technical overview? | `docs/PROJECT_OVERVIEW.md` |
| What is the license route? | `LICENSE_POLICY.md` |
| How can I contribute? | `CONTRIBUTING.md` |

## Suggested Contribution Path

1. Run the lightweight checks from `README.md`.
2. Read `CONTRIBUTING.md` and `MODEL_CONTRIBUTION_TERMS.md`.
3. Pick one bounded contribution:
   - new Taichi GPU solver/preconditioner evidence,
   - additional SuiteSparse benchmark rows,
   - better matrix diagnostics,
   - Transformer selector training with a model card,
   - runtime guard or fallback-chain improvements.
4. Add tests and artifact verifier checks.
5. Update public documentation only when it clarifies the user-facing capability,
   contribution path, or reproducibility boundary.

# TransSolveStack

TransSolveStack is a Taichi/GPU-first platform for sparse linear-system solver
selection. It connects matrix diagnostics, GPU solver profiling, selector-data
generation, learned-policy training contracts, and runtime safety guards into
one auditable stack.

The target users are researchers and engineers working with CAE, CFD, FEA,
graphics, and HPC sparse systems where solver choice is part of performance and
robustness, not a fixed implementation detail.

## What It Does

- Imports and diagnoses sparse matrices, especially CSR systems from
  SuiteSparse-style Matrix Market archives.
- Executes GPU solver probes through Taichi CSR operators and records numerical
  evidence.
- Builds selector rows for solver, preconditioner, precision, parameter, and
  fallback decisions.
- Exports Transformer-ready training tensors and model contracts.
- Packages external Transformer training inputs and validates submitted models
  through the guarded intake boundary.
- Runs learned-policy candidates behind offline quality gates and runtime
  guards before promotion.
- Preserves artifacts, provenance, and reproducibility checks for every
  benchmark-like boundary.

## System Structure

```text
Matrix sources
  -> CSR import and diagnostics
  -> Taichi GPU solver/preconditioner probes
  -> selector rows and benchmark artifacts
  -> Transformer-ready tensors and model contract
  -> offline quality gate
  -> learned runtime guard
  -> guarded solve / fallback chain
```

## Integrated Solver Stack

TransSolveStack exposes solver capability through auditable `PolicyPlan`
records and Taichi GPU CSR execution. The current integrated stack includes:

| Family | Integrated forms | Typical role |
|---|---|---|
| CG / PCG | `cg`, `pcg`, Jacobi preconditioning | Symmetric positive-definite CSR systems and baseline GPU sparse solves. |
| BiCGSTAB | none, Jacobi, ILU0, row/column equilibration probes | General or nonsymmetric CSR systems where CG is not applicable. |
| GMRES | restarted GMRES with configurable restart and Jacobi preconditioning | Nonsymmetric systems that need a more robust Krylov fallback candidate. |
| Richardson | weighted Richardson with Jacobi | Lightweight baseline and failure-mode evidence. |
| Chebyshev | Chebyshev/Jacobi with explicit spectral bounds | Semi-iterative candidate when usable bounds are known. |
| Preconditioner and scaling probes | Jacobi, ILU0, symmetric diagonal equilibration, row/column equilibration | Candidate-specific evidence for selector and fallback policies. |

Every integrated solver path is expected to produce numerical evidence: solver
status, residual history, CPU recomputed residual, solution-error checks when a
known RHS is available, runtime metadata, and artifact provenance.

## Transformer-Driven Policy Layer

The intended learned component is not a replacement for numerical solvers. It is
a solver-selection policy that learns from matrix structure, CSR diagnostics,
profiling rows, residual behavior, failure cases, and fallback outcomes.

For each sparse system, the policy can rank:

- solver family,
- preconditioner or scaling choice,
- precision and solver parameters,
- fallback chain order,
- whether the learned choice is safe enough to execute or should remain in
  shadow mode.

TransSolveStack exports Transformer-ready tensors and model contracts so
community training runs can target the same interface. A useful contributed
model should include model weights, a model card, training-data provenance,
quality-gate results, and guarded-solve evidence showing that the policy
improves solver choice without bypassing the runtime guard.

The repository also includes a trainer-facing package manifest and a
consumer-side dry run for external model submissions. This lets a contributor
train a model outside the repository, map the expected outputs back into the
TransSolveStack model-artifact format, and run replay, quality-gate,
submission, acceptance, and guarded-shadow checks before any runtime use.

The long-term goal is a shared benchmark and model ecosystem: larger SuiteSparse
and application-derived training campaigns should produce stronger selector
models, while the core platform keeps execution auditable through confidence
thresholds, exact-success checks, and fallback enforcement.

## Guarded Policy Runtime

The runtime guard is a core part of the platform, not an add-on around the
model. Learned selector predictions pass through these boundaries before they
can affect execution:

- **Shadow mode:** record learned predictions without changing the executable
  artifact-backed plan.
- **Offline quality gate:** require model evaluation evidence before runtime
  eligibility.
- **Confidence threshold:** block low-confidence learned selections.
- **Exact success requirement:** promote only candidates with profiled success
  for the exact matrix/context.
- **Fallback-chain enforcement:** keep executable fallbacks restricted to
  profiled-success candidates.
- **Runtime fallback guard:** retry fallback plans after failed status,
  exception, or post-attempt timeout.
- **Artifact audit trail:** attach selector, guard, and fallback decisions to
  `SolveResult` and `RunTrace` metadata.

## Why It Is Different

- **GPU-first runtime:** Taichi GPU execution is the primary target; CPU paths
  are for import, screening, diagnostics, and reference checks.
- **Evidence-driven selection:** Runtime candidates come from measured solver
  artifacts, not from hard-coded solver preference alone.
- **Learned-policy safety:** Learned selector outputs can be shadowed, gated,
  confidence-checked, and forced through profiled fallback chains.
- **Integrated solver coverage:** Krylov solvers, preconditioners, scaling
  probes, selector rows, and runtime guards are wired through the same planning
  and artifact system.
- **Reproducible artifacts:** Small JSON/Markdown artifacts under `runs/`
  record what was tested, how it passed, and which files back each claim.
- **Research-to-runtime bridge:** The stack is designed to accept Transformer
  selector training without bypassing numerical gates or runtime guards.

## Use Cases

| Use case | What TransSolveStack provides |
|---|---|
| GPU sparse solver experiments | Taichi CSR operators, solver/preconditioner probes, residual and solution-error checks. |
| Solver policy research | Selector rows, labels, failure rows, model contracts, and Transformer-ready tensors. |
| SuiteSparse benchmark campaigns | Matrix selection, header probing, bounded CSR import, artifact manifests, and resource-aware plans. |
| Safe learned solver selection | Offline quality gates, shadow mode, confidence thresholds, fallback enforcement, and guarded auto-solve. |
| Reproducible evaluation | Verifier scripts, artifact manifests, and compact campaign summaries. |

## Python Example

This example uses a small committed CSR artifact and runs a Taichi GPU CSR solve.

```python
import transsolvestack as tss
from transsolvestack.datasets.csr import csr_matrix_from_record
from transsolvestack.profiling.artifacts import read_jsonl

rows = read_jsonl("runs/phase1_suitesparse_csr_import/csr_matrices.jsonl")
record = next(row for row in rows if row["matrix_id"] == "suitesparse:Bai/cdde1")

csr = csr_matrix_from_record(record)
rhs = csr.matvec([1.0] * csr.n_cols)

result = tss.solve_csr(
    csr,
    rhs,
    solver="bicgstab",
    preconditioner="ilu0",
    precision="float64",
)

print(result.status)
print(result.trace.backend)
print(result.trace.final_residual_norm)
```

## Lightweight Validation

The committed artifact set is intentionally small. These checks do not download
the full SuiteSparse collection.

```bash
python3 -m venv .venv-tss
.venv-tss/bin/python -m pip install -e '.[taichi,dev]'
.venv-tss/bin/python scripts/tss_public_release_hygiene.py
.venv-tss/bin/python scripts/tss_verify_artifacts.py
pytest -q
```

Expected current result:

```text
artifact verification: passed
235 passed, 18 skipped
```

Full benchmark expansion uses external datasets under
`/mnt/tss_external/TransSolveStack/datasets` and should be run separately from
the lightweight public checks.

## Project Map

- Technical overview: [docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md)
- License route: [LICENSE_POLICY.md](LICENSE_POLICY.md)
- Commercial-use boundary: [COMMERCIAL_USE.md](COMMERCIAL_USE.md)
- Contribution rules: [CONTRIBUTING.md](CONTRIBUTING.md)
- Model contribution terms: [MODEL_CONTRIBUTION_TERMS.md](MODEL_CONTRIBUTION_TERMS.md)

## Contributing

Useful contributions are GPU solver/preconditioner implementations, safer
fallback chains, SuiteSparse benchmark artifacts, matrix diagnostics, and
Transformer selector training runs with model cards and quality-gate evidence.

Start with [CONTRIBUTING.md](CONTRIBUTING.md),
[MODEL_CONTRIBUTION_TERMS.md](MODEL_CONTRIBUTION_TERMS.md), and
[CONTRIBUTOR_LICENSE_AGREEMENT.md](CONTRIBUTOR_LICENSE_AGREEMENT.md).

## License

TransSolveStack source code uses the PolyForm Noncommercial License 1.0.0 route
with attribution to Wei CUI. Commercial use requires separate written permission
from Wei CUI. The project is not Apache-2.0/MIT/BSD licensed.

See [LICENSE](LICENSE), [LICENSE_POLICY.md](LICENSE_POLICY.md), and
[COMMERCIAL_USE.md](COMMERCIAL_USE.md) before using, redistributing, or building
on the project.

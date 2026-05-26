# Contributing

TransSolveStack is a GPU-first sparse linear solver policy platform. Contributions
should strengthen that direction rather than adding CPU-only solver shortcuts.

## Good Contributions

- Taichi GPU CSR solver and preconditioner implementations.
- Matrix diagnostics that improve solver or fallback selection.
- SuiteSparse benchmark rows produced under the resource policy.
- Transformer selector training experiments that use the existing model contract.
- Runtime guard, fallback-chain, timeout, and artifact-verifier improvements.
- Documentation that clarifies reproducibility, scope, or limitations.

## Required Evidence

Every functional change should include the smallest real check that proves it:

- Unit or smoke tests for narrow API behavior.
- Real GPU artifact rows for solver/preconditioner changes.
- Artifact verifier updates when a new `runs/` boundary is introduced.
- Footage and milestone notes for milestone-level capabilities.

Do not mark a benchmark, solver, selector, or guard stage passed unless the
numeric or functional gate actually ran.

## Licensing Boundary

The final project license is not selected yet. Contributions must be compatible
with the license policy in `LICENSE_POLICY.md` and must preserve attribution to
Wei CUI and TransSolveStack.

Do not submit code copied from projects with incompatible licenses. Do not submit
model weights, large datasets, or third-party benchmark archives unless their
redistribution terms are known and documented.

## Development Rules

- Keep Taichi GPU execution as the primary runtime target.
- Use CPU code only for import, screening, diagnostics, and reference checks.
- Keep full SuiteSparse archives outside git under the external dataset path.
- Keep generated heavy artifacts out of git.
- Run the public release and artifact gates before proposing a release snapshot.

```bash
.venv-tss/bin/python scripts/tss_public_release_hygiene.py
.venv-tss/bin/python scripts/tss_verify_artifacts.py
pytest -q
```

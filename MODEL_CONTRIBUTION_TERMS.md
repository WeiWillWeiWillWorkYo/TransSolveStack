# Model Contribution Terms

These terms describe the intended contribution boundary for Transformer selector
models trained for TransSolveStack.

## Intent

TransSolveStack should welcome open research contributions of trained selector
models while preventing unattributed or unapproved commercial packaging of the
project.

## Accepted Model Contributions

A model contribution should include:

- Model weights or a reproducible training recipe.
- Model card with architecture, training data, hardware, and limitations.
- Training configuration and random seeds when practical.
- Benchmark provenance and selector-row artifact references.
- Offline quality-gate result.
- Runtime guard/shadow-mode result before any promotion claim.

## Default Model Terms

Unless a separate written agreement says otherwise:

- Model weights contributed to TransSolveStack may be used by the project for
  research, evaluation, documentation, comparison, and future releases.
- Contributors should publish model weights under noncommercial terms compatible
  with the TransSolveStack source-code license route.
- Commercial deployment, commercial redistribution, paid hosted service use, or
  proprietary product integration of contributed TransSolveStack selector models
  requires separate written permission from Wei CUI.
- Attribution to Wei CUI, TransSolveStack, and the model contributor must be
  preserved.

## Official Inclusion

A trained selector model is not an official TransSolveStack model until it has:

- Reproducible training metadata.
- A model card.
- Passing offline quality-gate evidence.
- Passing guarded runtime shadow or promotion evidence.
- Maintainer approval for inclusion.

## Current Status

No final production Transformer selector model has been trained or accepted yet.

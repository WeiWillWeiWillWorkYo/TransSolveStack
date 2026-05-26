# Public Release Checklist

- hygiene_ready: `True`
- github_upload_ready: `True`
- project_owner: `Wei CUI`
- license_route_selected: `True`
- formal_license_finalized: `True`
- source_code_license: `PolyForm Noncommercial License 1.0.0`
- commercial_use_requires_permission: `True`
- permissive_license_family_rejected: `True`
- git_repository_initialized: `True`
- git_init_required: `False`
- runtime_selector_changed: `False`
- large_unignored_file_count: `0`
- secret_like_file_count: `0`
- validation_error_count: `0`

Before public upload:

1. Final-review the selected license and contribution documents.
2. Run the artifact verifier and full test suite.
3. Initialize git only after reviewing ignored and generated files.
4. Commit source, tests, professional docs, configs, and small JSON/Markdown runs artifacts.
5. Keep virtual environments, caches, internal development docs, paper-prep notes, footage, full SuiteSparse data, raw matrix archives, and heavy binary artifacts out of git.
6. Push to a fresh GitHub repository after `git status` matches the release manifest.

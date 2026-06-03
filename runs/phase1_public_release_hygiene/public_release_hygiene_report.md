# Public Release Hygiene

- status: `passed`
- schema_version: `phase1_public_release_hygiene_v1`
- public_release_hygiene_ready: `True`
- github_upload_ready: `True`
- project_owner: `Wei CUI`
- license_route_selected: `True`
- formal_license_finalized: `True`
- source_code_license: `PolyForm Noncommercial License 1.0.0`
- commercial_use_requires_permission: `True`
- permissive_license_family_rejected: `True`
- git_repository_initialized: `True`
- git_init_required: `False`
- workspace_file_count_scanned: `1181`
- workspace_bytes_scanned: `91057311`
- local_only_dir_count: `16`
- ignored_heavy_file_count: `14`
- large_unignored_file_count: `0`
- secret_like_file_count: `0`
- validation_error_count: `0`

| row_kind | status | path | detail |
|---|---|---|---|
| git_repository | initialized | .git | git repository is initialized |
| gitignore_required_pattern | present | .venv*/ | required public-release ignore rule |
| gitignore_required_pattern | present | __pycache__/ | required public-release ignore rule |
| gitignore_required_pattern | present | *.py[cod] | required public-release ignore rule |
| gitignore_required_pattern | present | .pytest_cache/ | required public-release ignore rule |
| gitignore_required_pattern | present | .env | required public-release ignore rule |
| gitignore_required_pattern | present | .env.* | required public-release ignore rule |
| gitignore_required_pattern | present | *.pem | required public-release ignore rule |
| gitignore_required_pattern | present | *.key | required public-release ignore rule |
| gitignore_required_pattern | present | /datasets/ | required public-release ignore rule |
| gitignore_required_pattern | present | /external/ | required public-release ignore rule |
| gitignore_required_pattern | present | /downloads/ | required public-release ignore rule |
| gitignore_required_pattern | present | /data/raw/ | required public-release ignore rule |
| gitignore_required_pattern | present | /data/external/ | required public-release ignore rule |
| gitignore_required_pattern | present | /data/suitesparse/ | required public-release ignore rule |
| gitignore_required_pattern | present | runs/**/*.npy | required public-release ignore rule |
| gitignore_required_pattern | present | runs/**/*.npz | required public-release ignore rule |
| gitignore_required_pattern | present | runs/**/*.pkl | required public-release ignore rule |
| gitignore_required_pattern | present | runs/**/*.bin | required public-release ignore rule |
| gitignore_required_pattern | present | runs/**/*.mp4 | required public-release ignore rule |
| gitignore_required_pattern | present | runs/**/*.zip | required public-release ignore rule |
| gitignore_required_pattern | present | runs/**/csr_matrices.jsonl | required public-release ignore rule |
| gitignore_required_pattern | present | ADVICE.md | required public-release ignore rule |
| gitignore_required_pattern | present | MILESTONE_LOG.md | required public-release ignore rule |
| gitignore_required_pattern | present | FOOTAGE_SPEC.md | required public-release ignore rule |
| gitignore_required_pattern | present | PUBLIC_RELEASE.md | required public-release ignore rule |
| gitignore_required_pattern | present | GITHUB_RELEASE_GATE.md | required public-release ignore rule |
| gitignore_required_pattern | present | PHASE1_ROADMAP.md | required public-release ignore rule |
| gitignore_required_pattern | present | advice.md | required public-release ignore rule |
| gitignore_required_pattern | present | scope.md | required public-release ignore rule |
| gitignore_required_pattern | present | footage/ | required public-release ignore rule |
| gitignore_required_pattern | present | docs/toms_paper/ | required public-release ignore rule |
| gitignore_required_pattern | present | reports/ | required public-release ignore rule |
| public_source_path | present | transsolvestack | directory |
| public_source_path | present | scripts | directory |
| public_source_path | present | tests | directory |
| public_source_path | present | configs | directory |
| public_source_path | present | runs | directory |
| public_source_path | present | LICENSE | file |
| public_source_path | present | README.md | file |
| public_source_path | present | docs/PROJECT_OVERVIEW.md | file |
| public_source_path | present | LICENSE_POLICY.md | file |
| public_source_path | present | COMMERCIAL_USE.md | file |
| public_source_path | present | MODEL_CONTRIBUTION_TERMS.md | file |
| public_source_path | present | CONTRIBUTOR_LICENSE_AGREEMENT.md | file |
| public_source_path | present | CITATION.cff | file |
| public_source_path | present | CONTRIBUTING.md | file |
| public_source_path | present | .gitignore | file |
| public_source_path | present | pyproject.toml | file |
| release_required_document | present | README.md | file |
| release_required_document | present | docs/PROJECT_OVERVIEW.md | file |
| release_required_document | present | LICENSE | file |
| release_required_document | present | LICENSE_POLICY.md | file |
| release_required_document | present | COMMERCIAL_USE.md | file |
| release_required_document | present | MODEL_CONTRIBUTION_TERMS.md | file |
| release_required_document | present | CONTRIBUTOR_LICENSE_AGREEMENT.md | file |
| release_required_document | present | CITATION.cff | file |
| release_required_document | present | CONTRIBUTING.md | file |
| license_decision | polyform_noncommercial_selected | LICENSE | commercial use requires separate permission from Wei CUI |
| license_decision | permissive_family_rejected | LICENSE_POLICY.md | Apache-2.0/MIT/BSD are not the intended route |
| license_decision | noncommercial_model_terms_documented | MODEL_CONTRIBUTION_TERMS.md | model cards, provenance, and quality gates required |
| license_decision | cla_light_documented | CONTRIBUTOR_LICENSE_AGREEMENT.md | contributions grant project reuse and relicensing rights |
| citation | present | CITATION.cff | author: Wei CUI |
| local_only_dir_present | ignored_by_policy | .git | must not be committed |
| local_only_dir_present | ignored_by_policy | .pytest_cache | must not be committed |
| local_only_dir_present | ignored_by_policy | .venv-tss | must not be committed |
| local_only_dir_present | ignored_by_policy | scripts/__pycache__ | must not be committed |
| local_only_dir_present | ignored_by_policy | tests/__pycache__ | must not be committed |
| local_only_dir_present | ignored_by_policy | transsolvestack/__pycache__ | must not be committed |
| local_only_dir_present | ignored_by_policy | transsolvestack/backends/__pycache__ | must not be committed |
| local_only_dir_present | ignored_by_policy | transsolvestack/benchmarks/__pycache__ | must not be committed |
| local_only_dir_present | ignored_by_policy | transsolvestack/core/__pycache__ | must not be committed |
| local_only_dir_present | ignored_by_policy | transsolvestack/datasets/__pycache__ | must not be committed |
| local_only_dir_present | ignored_by_policy | transsolvestack/operators/__pycache__ | must not be committed |
| local_only_dir_present | ignored_by_policy | transsolvestack/policies/__pycache__ | must not be committed |
| local_only_dir_present | ignored_by_policy | transsolvestack/preconditioners/__pycache__ | must not be committed |
| local_only_dir_present | ignored_by_policy | transsolvestack/profiling/__pycache__ | must not be committed |
| local_only_dir_present | ignored_by_policy | transsolvestack/runtime/__pycache__ | must not be committed |
| local_only_dir_present | ignored_by_policy | transsolvestack/solvers/__pycache__ | must not be committed |
| gitignore_line_count | present | .gitignore | 64 |

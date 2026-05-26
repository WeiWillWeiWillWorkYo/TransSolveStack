from pathlib import Path


def test_public_release_license_and_citation_docs_are_present():
    license_policy = Path("LICENSE_POLICY.md").read_text(encoding="utf-8")
    license_notice = Path("LICENSE").read_text(encoding="utf-8")
    commercial_use = Path("COMMERCIAL_USE.md").read_text(encoding="utf-8")
    model_terms = Path("MODEL_CONTRIBUTION_TERMS.md").read_text(encoding="utf-8")
    cla = Path("CONTRIBUTOR_LICENSE_AGREEMENT.md").read_text(encoding="utf-8")
    citation = Path("CITATION.cff").read_text(encoding="utf-8")
    contributing = Path("CONTRIBUTING.md").read_text(encoding="utf-8")
    readme = Path("README.md").read_text(encoding="utf-8")
    overview = Path("docs/PROJECT_OVERVIEW.md").read_text(encoding="utf-8")

    assert "Wei CUI" in license_policy
    assert "PolyForm Noncommercial License 1.0.0" in license_policy
    assert "Apache-2.0, MIT, BSD" in license_policy
    assert "Commercial use is not automatically granted" in license_policy
    assert "PolyForm Noncommercial License 1.0.0" in license_notice
    assert "Commercial use requires separate written permission" in license_notice
    assert "commercial product" in commercial_use
    assert "Model weights" in model_terms
    assert "quality-gate" in model_terms
    assert "commercial licensing of TransSolveStack by Wei CUI" in cla
    assert "given-names: \"Wei\"" in citation
    assert "family-names: \"CUI\"" in citation
    assert "LICENSE_POLICY.md" in contributing
    assert "LICENSE_POLICY.md" in readme
    assert "Commercial use requires separate written permission" in readme
    assert "What It Does" in readme
    assert "System Structure" in readme
    assert "Integrated Solver Stack" in readme
    assert "Transformer-Driven Policy Layer" in readme
    assert "Guarded Policy Runtime" in readme
    assert "Why It Is Different" in readme
    assert "Use Cases" in readme
    assert "Python Example" in readme
    assert "Lightweight Validation" in readme
    assert "193 passed, 18 skipped" in readme
    assert "docs/PROJECT_OVERVIEW.md" in readme
    assert "tss.solve_csr" in readme
    assert "BiCGSTAB" in readme
    assert "GMRES" in readme
    assert "ILU0" in readme
    assert "Transformer-ready tensors" in readme
    assert "community training runs" in readme
    assert "model weights" in readme
    assert "model card" in readme
    assert "fallback chain order" in readme
    assert "Shadow mode" in readme
    assert "Fallback-chain enforcement" in readme
    assert "SuiteSparse benchmark campaigns" in readme
    assert "Current Capabilities" in overview
    assert "Basic Structure" in overview
    assert "Integrated Solvers" in overview
    assert "Runtime Guard" in overview
    assert "Advantages" in overview
    assert "Development Boundary" in overview
    assert "Suggested Contribution Path" in overview
    assert "Dataset layer" in overview
    assert "Guard layer" in overview
    assert "Restarted GMRES" in overview
    assert "offline quality-gate eligibility" in overview
    for private_link in (
        "ADVICE.md",
        "MILESTONE_LOG.md",
        "FOOTAGE_SPEC.md",
        "docs/toms_paper",
        "GITHUB_RELEASE_GATE.md",
        "PUBLIC_RELEASE.md",
    ):
        assert private_link not in readme
        assert private_link not in overview

from __future__ import annotations

from pathlib import Path

from scripts.check_docs import check_docs, check_document, check_public_claim_language

PROJECT_ROOT = Path(__file__).parents[2]


def test_checked_in_documentation_links_anchors_and_evidence_are_valid() -> None:
    assert check_docs(PROJECT_ROOT) == ()


def test_docs_policy_rejects_broken_links_placeholders_and_reserved_claims(
    tmp_path: Path,
) -> None:
    docs = tmp_path / "docs" / "operations"
    docs.mkdir(parents=True)
    page = docs / "guide.md"
    page.write_text(
        "# Guide\n\nTODO: call this engineering-grade.\n\n[missing](absent.md#nope)\n",
        encoding="utf-8",
    )
    document_errors = check_document(page, root=tmp_path)
    claim_errors = check_public_claim_language(page, root=tmp_path)
    assert any("unresolved documentation placeholder" in error for error in document_errors)
    assert any("broken relative link" in error for error in document_errors)
    assert any("reserved public claim language" in error for error in claim_errors)

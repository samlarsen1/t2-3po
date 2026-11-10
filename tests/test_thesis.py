from pathlib import Path

import pytest

from t2_3po.thesis import ThesisError, discover_theses, parse_thesis, read_document


def _write(tmp_path: Path, text: str, name: str = "t.md") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_parses_front_matter_and_resolves_documents(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "---\nticker: NVDA\nt212_ticker: NVDA_US_EQ\ndocuments: [docs/a.txt]\n---\n\nBody text\n",
    )
    thesis = parse_thesis(path)
    assert thesis.meta.ticker == "NVDA"
    assert thesis.meta.t212_ticker == "NVDA_US_EQ"
    assert thesis.meta.documents == [tmp_path / "docs/a.txt"]
    assert thesis.body == "Body text"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("No front matter", "must start"),
        ("---\nticker: NVDA\nBody", "not closed"),
        ("---\nticker: [unclosed\n---\nBody", "invalid front matter"),
        ("---\nname: missing ticker\n---\nBody", "invalid front matter"),
        ("---\nticker: ../../etc\n---\nBody", "invalid front matter"),
        ("---\nticker: NVDA\ntypo_field: 1\n---\nBody", "invalid front matter"),
        ("---\nticker: NVDA\n---\n\n", "empty"),
    ],
)
def test_rejects_malformed_theses(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(ThesisError, match=message):
        parse_thesis(_write(tmp_path, text))


def test_discovery_skips_examples(tmp_path: Path) -> None:
    keep = _write(tmp_path, "x", "NVDA.md")
    _write(tmp_path, "x", "AMD.example.md")
    assert discover_theses(tmp_path) == [keep]


def test_missing_document_raises(tmp_path: Path) -> None:
    with pytest.raises(ThesisError, match="can't read document"):
        read_document(tmp_path / "missing.txt")

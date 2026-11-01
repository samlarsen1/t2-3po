"""Reads investment theses: markdown files with YAML front matter."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

EXAMPLE_SUFFIX = ".example.md"


class ThesisError(Exception):
    """A thesis file or a document it references can't be used."""


class ThesisMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticker: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    t212_ticker: str | None = None
    yahoo_ticker: str | None = None
    documents: list[Path] = Field(default_factory=list)


class Thesis(BaseModel):
    path: Path
    meta: ThesisMeta
    body: str


def parse_thesis(path: Path) -> Thesis:
    """Parse a thesis file.

    Args:
        path: Markdown file starting with a ``---`` delimited YAML front matter block.

    Returns:
        The parsed thesis, with ``documents`` resolved relative to the thesis file.

    Raises:
        ThesisError: If the file is unreadable, malformed, or has no thesis text.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ThesisError(f"{path}: can't read file ({type(exc).__name__})") from exc

    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ThesisError(f"{path}: must start with a YAML front matter block")
    try:
        end = next(i for i, line in enumerate(lines[1:], start=1) if line.strip() == "---")
    except StopIteration:
        raise ThesisError(f"{path}: front matter block is not closed") from None

    try:
        meta = ThesisMeta.model_validate(yaml.safe_load("\n".join(lines[1:end])))
    except (yaml.YAMLError, ValidationError) as exc:
        raise ThesisError(f"{path}: invalid front matter: {exc}") from exc

    body = "\n".join(lines[end + 1 :]).strip()
    if not body:
        raise ThesisError(f"{path}: thesis text is empty")

    meta.documents = [path.parent / document for document in meta.documents]
    return Thesis(path=path, meta=meta, body=body)


def discover_theses(directory: Path) -> list[Path]:
    """List thesis files in a directory, skipping ``*.example.md`` templates."""
    return sorted(p for p in directory.glob("*.md") if not p.name.endswith(EXAMPLE_SUFFIX))


def read_document(path: Path) -> str:
    """Read a referenced text document in full, such as a transcript or 10-K.

    Raises:
        ThesisError: If the document is missing or isn't UTF-8 text.
    """
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ThesisError(f"{path}: can't read document ({type(exc).__name__})") from exc

#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from typing import Any

ATTACHMENT_MD_RE = re.compile(r"!\[[^\]]*\]\(attachment:[^)]+\)")
ATTACHMENT_HTML_RE = re.compile(r"<img[^>]*src=[\"']attachment:[^\"']+[\"'][^>]*>")


def _to_text(source: Any) -> tuple[str, bool]:
    if isinstance(source, list):
        return "".join(source), True
    return str(source), False


def _from_text(text: str, was_list: bool) -> Any:
    if not was_list:
        return text
    lines = text.splitlines(keepends=True)
    if text and not text.endswith(("\n", "\r")) and lines:
        lines[-1] = lines[-1].rstrip("\n\r")
    return lines


def clean_notebook(nb: dict[str, Any]) -> dict[str, Any]:
    cells = nb.get("cells", [])
    if isinstance(cells, list):
        for cell in cells:
            if not isinstance(cell, dict):
                continue

            if cell.get("cell_type") == "code":
                cell["outputs"] = []
                cell["execution_count"] = None

            if "attachments" in cell:
                cell.pop("attachments", None)

            if cell.get("cell_type") == "markdown" and "source" in cell:
                text, was_list = _to_text(cell.get("source", ""))
                text = ATTACHMENT_MD_RE.sub("", text)
                text = ATTACHMENT_HTML_RE.sub("", text)
                cell["source"] = _from_text(text, was_list)

    metadata = nb.get("metadata")
    if isinstance(metadata, dict):
        metadata.pop("signature", None)
        metadata.pop("widgets", None)

    return nb


def main() -> int:
    raw = sys.stdin.read()
    if not raw.strip():
        return 0

    try:
        notebook = json.loads(raw)
    except json.JSONDecodeError:
        sys.stdout.write(raw)
        return 0

    cleaned = clean_notebook(notebook)
    sys.stdout.write(json.dumps(cleaned, ensure_ascii=False, indent=1))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

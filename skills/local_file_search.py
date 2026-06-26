"""Skill: local_file_search — search and list files in the workspace."""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Annotated

from langchain_core.tools import tool

from skills.file_reader import _safe_path, _workspace_root


@tool
def local_file_search(
    query: Annotated[
        str | None,
        "Filename pattern to search for (glob, e.g. '*.py' or 'report*'). "
        "Omit to list all files.",
    ] = None,
    directory: Annotated[
        str | None,
        "Subdirectory to search in, relative to workspace root. Defaults to root.",
    ] = None,
    max_depth: Annotated[
        int,
        "Maximum recursion depth for search. Defaults to 3.",
    ] = 3,
) -> str:
    """Search for files matching a pattern, or list directory contents.

    Returns matching file paths (one per line) or an empty result message.
    """
    root = _workspace_root.resolve()
    target = root
    if directory:
        try:
            target = _safe_path(directory)
        except ValueError as exc:
            return str(exc)

    if not target.exists():
        return f"Error: directory not found — {directory or '.'}"

    exclude = {"__pycache__", ".git", ".svn", ".hg"}

    lines: list[str] = []
    for entry in sorted(target.rglob("*")):
        rel = entry.relative_to(root)
        depth = len(rel.parts)
        if depth > max_depth:
            continue
        # Skip excluded directories
        if any(p in exclude for p in rel.parts):
            continue

        if query and entry.is_file():
            if fnmatch.fnmatch(entry.name, query):
                prefix = "  " * (depth - 1)
                lines.append(f"{prefix}[F] {rel}")
        elif not query:
            prefix = "  " * (depth - 1)
            tag = "[D]" if entry.is_dir() else "[F]"
            lines.append(f"{prefix}{tag} {rel.name if depth > 0 else rel}")

    return "\n".join(lines) if lines else "(no matches)"

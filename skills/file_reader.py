"""Skill: file_reader — read file contents from a workspace directory.

Paths are resolved relative to a configurable workspace root to prevent
escaping the allowed directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from langchain_core.tools import tool

# Workspace root, overridable at runtime via b2_run_skill or configs.
_workspace_root: Path = Path.cwd()


def set_workspace_root(path: str | Path) -> None:
    """Change the workspace root for file operations."""
    global _workspace_root
    _workspace_root = Path(path).resolve()
    _workspace_root.mkdir(parents=True, exist_ok=True)


def _safe_path(relative: str) -> Path:
    """Resolve a relative path and ensure it stays inside the workspace."""
    root = _workspace_root.resolve()
    target = (root / relative).resolve()
    if root not in target.parents and target != root:
        raise ValueError(f"Access denied: {relative!r} escapes workspace {root}")
    return target


@tool
def file_reader(
    path: Annotated[str, "Path to the file, relative to the workspace root."],
    max_size_kb: Annotated[
        int | None,
        "Maximum file size in KB to read. Defaults to 512.",
    ] = None,
) -> str:
    """Read the contents of a file in the workspace.

    Returns the file text or an error message if the file cannot be read.
    """
    limit_bytes = (max_size_kb or 512) * 1024

    try:
        target = _safe_path(path)
    except ValueError as exc:
        return str(exc)

    if not target.exists():
        return f"Error: file not found — {path}"
    if not target.is_file():
        return f"Error: not a file — {path}"
    if target.stat().st_size > limit_bytes:
        return f"Error: file exceeds {max_size_kb or 512} KB limit"

    try:
        return target.read_text(encoding="utf-8")
    except Exception as exc:
        return f"Error reading {path}: {exc}"

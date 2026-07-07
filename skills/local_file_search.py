from __future__ import annotations

import re

from skills import resolve_data_path


def _snippet(text: str, terms: list[str], radius: int = 60) -> str:
    lowered = text.casefold()
    positions = [lowered.find(term.casefold()) for term in terms]
    positions = [position for position in positions if position >= 0]
    start = max(0, (min(positions) if positions else 0) - radius)
    end = min(len(text), start + radius * 2)
    prefix = "..." if start else ""
    suffix = "..." if end < len(text) else ""
    return prefix + text[start:end].replace("\n", " ").strip() + suffix


def local_file_search(
    query: str,
    root_dir: str = "docs",
    file_types: list[str] | None = None,
    top_k: int = 5,
    *,
    data_root: str | None = None,
) -> dict:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    search_root, data_root_path = resolve_data_path(root_dir, data_root)
    if not search_root.is_dir():
        raise FileNotFoundError(f"search directory not found: {root_dir}")
    extensions = file_types or ["txt", "md"]
    normalized_extensions = {f".{item.lower().lstrip('.')}" for item in extensions}
    if not normalized_extensions.issubset({".txt", ".md"}):
        raise ValueError("local_file_search only supports txt and md")
    terms = [term for term in re.split(r"\s+", query.strip()) if term]
    results = []
    for path in sorted(search_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in normalized_extensions:
            continue
        text = path.read_text(encoding="utf-8")
        lowered = text.casefold()
        score = sum(lowered.count(term.casefold()) for term in terms)
        if score:
            results.append(
                {
                    "path": path.relative_to(data_root_path).as_posix(),
                    "score": score,
                    "snippet": _snippet(text, terms),
                }
            )
    results.sort(key=lambda item: (-item["score"], item["path"]))
    return {"results": results[:top_k]}

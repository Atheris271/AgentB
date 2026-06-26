"""Skill: table_analyzer — analyze and query tabular data (CSV, TSV, etc.)."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Annotated

from langchain_core.tools import tool

from skills.file_reader import _safe_path


def _detect_delimiter(text: str) -> str:
    """Heuristic: return ',' for CSV, '\\t' for TSV."""
    first_line = text.split("\n")[0] if text else ""
    tabs = first_line.count("\t")
    commas = first_line.count(",")
    return "\t" if tabs > commas else ","


def _read_table(source: str, is_path: bool, max_rows: int) -> tuple[list[str], list[list[str]], str]:
    """Parse tabular data and return (headers, rows, delimiter_used)."""
    if is_path:
        target = _safe_path(source)
        if not target.exists():
            raise FileNotFoundError(f"File not found: {source}")
        text = target.read_text(encoding="utf-8")
    else:
        text = source

    delimiter = _detect_delimiter(text)
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    rows = list(reader)
    if not rows:
        raise ValueError("Empty table")

    headers = rows[0]
    data = rows[1:][:max_rows]
    return headers, data, delimiter


@tool
def table_analyzer(
    source: Annotated[
        str,
        "Path to a CSV/TSV file (relative to workspace), or inline CSV/TSV text.",
    ],
    action: Annotated[
        str,
        "What to do: 'summary' (row count, columns, types), "
        "'head:N' (first N rows, N defaults to 5), "
        "'columns' (list column names), or "
        "'query:COLUMN CONDITION' (filter rows, e.g. 'query:age > 30').",
    ] = "summary",
    max_rows: Annotated[
        int,
        "Maximum rows to process. Defaults to 1000.",
    ] = 1000,
) -> str:
    """Analyze tabular data from a file or inline text.

    Supports CSV and TSV. Call with action='summary' for an overview,
    action='head:10' for first 10 rows, or action='columns' for column names.
    """
    # Determine whether source is a file path or inline data.
    is_path = not ("\n" in source or "," in source or "\t" in source)

    try:
        headers, data, delim = _read_table(source, is_path, max_rows)
    except Exception as exc:
        return f"Error: {exc}"

    delim_name = "TSV (tab)" if delim == "\t" else "CSV (comma)"

    if action == "summary" or action.startswith("summary"):
        col_info = []
        for i, h in enumerate(headers):
            sample = [r[i] for r in data[:5] if i < len(r)]
            numeric = all(
                (v.strip().replace(".", "").replace("-", "").isdigit())
                for v in sample if v.strip()
            )
            col_info.append(
                f"  {h}: {'numeric' if numeric and sample else 'text'} "
                f"(sample: {', '.join(sample[:3])})"
            )
        return (
            f"Format: {delim_name}\n"
            f"Rows: {len(data)} (showing up to {max_rows})\n"
            f"Columns ({len(headers)}):\n" + "\n".join(col_info)
        )

    if action == "columns":
        return "Columns:\n" + "\n".join(f"  [{i}] {h}" for i, h in enumerate(headers))

    if action.startswith("head"):
        n = 5
        if ":" in action:
            try:
                n = int(action.split(":")[1])
            except ValueError:
                pass
        lines = [delim.join(headers)]
        lines.extend(delim.join(r) for r in data[:n])
        return f"First {min(n, len(data))} rows ({delim_name}):\n" + "\n".join(lines)

    if action.startswith("query:"):
        expr = action.split(":", 1)[1].strip()
        # Simple query syntax:  COLUMN OP VALUE   e.g. "age > 30"
        parts = expr.split(None, 2)
        if len(parts) < 3:
            return "Error: query format is 'COLUMN OP VALUE', e.g. 'age > 30'"
        col, op, val = parts[0], parts[1], parts[2]

        if col not in headers:
            return f"Error: column {col!r} not found. Available: {headers}"
        ci = headers.index(col)

        matched = []
        for r in data:
            if ci >= len(r):
                continue
            cell = r[ci].strip()
            try:
                cv = float(cell) if cell.replace(".", "").replace("-", "").isdigit() else cell
                tv = float(val) if val.replace(".", "").replace("-", "").isdigit() else val
                ok = False
                if op == ">" and cv > tv: ok = True
                elif op == "<" and cv < tv: ok = True
                elif op == ">=" and cv >= tv: ok = True
                elif op == "<=" and cv <= tv: ok = True
                elif op == "==" and cv == tv: ok = True
                elif op == "!=" and cv != tv: ok = True
                elif op == "contains" and val in cell: ok = True
                if ok:
                    matched.append(r)
            except (ValueError, TypeError):
                continue

        lines = [delim.join(headers)]
        lines.extend(delim.join(r) for r in matched[:50])
        return (
            f"Query: {col} {op} {val} → {len(matched)} rows matched (showing first 50):\n"
            + "\n".join(lines)
        )

    return f"Unknown action: {action!r}. Use: summary, head:N, columns, or query:COL OP VAL"

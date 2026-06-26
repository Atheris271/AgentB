"""Skill: format_converter — convert data between JSON, CSV, YAML, and plain text."""

from __future__ import annotations

import csv
import io
import json
from typing import Annotated

from langchain_core.tools import tool

try:
    import yaml  # optional: pip install pyyaml
except ImportError:
    yaml = None  # type: ignore[assignment]


_FORMATS = {"json", "csv", "yaml", "txt"}


@tool
def format_converter(
    input_data: Annotated[
        str,
        "The data to convert (inline text, or a path relative to the workspace).",
    ],
    from_format: Annotated[
        str,
        "Source format: json, csv, yaml, or txt.",
    ],
    to_format: Annotated[
        str,
        "Target format: json, csv, yaml, or txt.",
    ],
    indent: Annotated[
        int,
        "Indentation for JSON/YAML output. Defaults to 2.",
    ] = 2,
) -> str:
    """Convert data between common formats.

    Formats supported: json, csv, yaml, txt.
    For CSV conversion, the data must be tabular (list of dicts, or
    list-of-lists with a header row).
    """
    f = from_format.lower().strip()
    t = to_format.lower().strip()

    if f not in _FORMATS:
        return f"Error: unsupported source format {f!r}. Use: {_FORMATS}"
    if t not in _FORMATS:
        return f"Error: unsupported target format {t!r}. Use: {_FORMATS}"
    if f == t:
        return "Input and output formats are the same — nothing to convert."

    # ---- Parse input ----
    try:
        parsed = _parse(input_data, f)
    except Exception as exc:
        return f"Parse error ({f}): {exc}"

    # ---- Serialize output ----
    try:
        output = _serialize(parsed, t, indent)
    except Exception as exc:
        return f"Serialize error ({t}): {exc}"

    return output


def _parse(data: str, fmt: str) -> object:
    """Parse a string in the given format into a Python object."""
    if fmt == "json":
        return json.loads(data)
    if fmt == "yaml":
        if yaml is None:
            raise ImportError("pyyaml is not installed. Run: pip install pyyaml")
        return yaml.safe_load(data)
    if fmt == "csv":
        # Detect delimiter
        first = data.split("\n")[0] if data else ""
        delim = "\t" if first.count("\t") > first.count(",") else ","
        reader = csv.DictReader(io.StringIO(data), delimiter=delim)
        return list(reader)
    if fmt == "txt":
        return {"text": data}
    raise ValueError(f"Unknown format: {fmt}")


def _serialize(obj: object, fmt: str, indent: int) -> str:
    """Serialize a Python object to the given format string."""
    if fmt == "json":
        return json.dumps(obj, indent=indent, ensure_ascii=False)
    if fmt == "yaml":
        if yaml is None:
            raise ImportError("pyyaml is not installed. Run: pip install pyyaml")
        return yaml.dump(obj, indent=indent, allow_unicode=True, sort_keys=False)
    if fmt == "csv":
        records: list[dict] = []
        if isinstance(obj, dict) and "text" in obj:
            return str(obj["text"])
        if isinstance(obj, list):
            records = obj
        elif isinstance(obj, dict):
            records = [obj]
        if not records:
            return ""
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
        return output.getvalue()
    if fmt == "txt":
        if isinstance(obj, dict):
            if "text" in obj:
                return str(obj["text"])
            return json.dumps(obj, indent=indent, ensure_ascii=False)
        if isinstance(obj, str):
            return obj
        return str(obj)
    raise ValueError(f"Unknown format: {fmt}")

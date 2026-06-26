"""B2: Skill工具函数模块 — skill registry, loading, and execution.

Responsibilities:
    - ToolRegistry:   stores and retrieves tools by name.
    - load_skills():  dynamically imports skill modules from the ../skills/ directory
                      and returns BaseTool instances.
    - Skill look-up:  agents use this module to discover what tools are available.

This module does NOT handle tool binding (that is B1's job) — it only manages
the skill catalogue and provides a simple API to load & list skills.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool


# =========================================================================
# ToolRegistry
# =========================================================================


class ToolRegistry:
    """Holds all available skill tools.

    Tools are registered once and retrieved by name or as a full list for
    binding to an LLM via ``model.bind_tools(registry.list_all())``.

    Usage::

        registry = ToolRegistry()
        registry.register(my_tool)
        registry.register_many([tool_a, tool_b])
        tools = registry.list_all()  # -> list[BaseTool]
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, t: BaseTool | Callable[..., Any]) -> None:
        if not isinstance(t, BaseTool):
            raise TypeError(
                f"Expected a BaseTool instance, got {type(t).__name__}. "
                "Use the @tool decorator from langchain_core.tools."
            )
        self._tools[t.name] = t

    def register_many(self, tools: list[BaseTool | Callable[..., Any]]) -> None:
        for t in tools:
            self.register(t)

    def get(self, name: str) -> BaseTool:
        if name not in self._tools:
            raise KeyError(
                f"Tool {name!r} not found. Available: {list(self._tools)}"
            )
        return self._tools[name]

    def list_all(self) -> list[BaseTool]:
        return list(self._tools.values())

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __repr__(self) -> str:
        return f"ToolRegistry(tools={self.names()})"


# =========================================================================
# Dynamic skill loading
# =========================================================================

# Map of skill name → Python module path under the skills package.
_SKILL_MODULE_MAP: dict[str, str] = {
    "calculator":         "skills.calculator",
    "file_reader":        "skills.file_reader",
    "local_file_search":  "skills.local_file_search",
    "table_analyzer":     "skills.table_analyzer",
    "format_converter":   "skills.format_converter",
}


def _ensure_skills_on_path() -> None:
    """Add the project root to sys.path so 'skills' is importable."""
    root = Path(__file__).resolve().parent.parent  # agent/
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)


def load_skills(names: list[str] | None = None) -> list[BaseTool]:
    """Dynamically import skill modules and return their @tool instances.

    Args:
        names: Skill names to load (e.g. ``['calculator', 'file_reader']``).
               If None or empty, loads ALL known skills.

    Returns:
        A list of ``BaseTool`` instances ready for binding to an LLM.

    Raises:
        ImportError: If a skill module cannot be imported.
        AttributeError: If the expected tool function is not found in the module.
    """
    _ensure_skills_on_path()

    targets = names or list(_SKILL_MODULE_MAP)
    tools: list[BaseTool] = []

    for name in targets:
        module_path = _SKILL_MODULE_MAP.get(name)
        if module_path is None:
            print(f"⚠️  Unknown skill {name!r} — skipped. Known: {list(_SKILL_MODULE_MAP)}")
            continue

        try:
            mod = importlib.import_module(module_path)
        except ImportError as exc:
            print(f"⚠️  Cannot import {module_path}: {exc}")
            continue

        # Each skill module exports a function with the same name as the skill.
        tool_fn = getattr(mod, name, None)
        if tool_fn is None:
            print(f"⚠️  Module {module_path} has no function {name!r} — skipped")
            continue

        if not isinstance(tool_fn, BaseTool):
            print(f"⚠️  {module_path}.{name} is not a BaseTool — skipped")
            continue

        tools.append(tool_fn)

    return tools


def load_skills_into_registry(
    registry: ToolRegistry,
    names: list[str] | None = None,
) -> ToolRegistry:
    """Convenience: load skills and register them all at once."""
    tools = load_skills(names)
    registry.register_many(tools)
    return registry

"""Skill: calculator — safe arithmetic expression evaluator.

A standalone @tool function. Does NOT handle binding or invocation —
that's the job of b2_run_skill / b1_agent_runtime.
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Annotated

from langchain_core.tools import tool

# Whitelist of safe operations.
_SAFE_OPS: dict[str, callable] = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "//": operator.floordiv,
    "%": operator.mod,
    "**": operator.pow,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "pi": math.pi,
    "e": math.e,
}


def _evaluate(node: ast.AST) -> float | int:
    """Walk an AST expression node and return a numeric result."""
    match node:
        case ast.Constant(value=v) if isinstance(v, int | float):
            return v
        case ast.BinOp(left=l, op=op, right=r):
            op_name = type(op).__name__.replace("Mult", "*").replace("Add", "+") \
                .replace("Sub", "-").replace("Div", "/").replace("Pow", "**") \
                .replace("Mod", "%").replace("FloorDiv", "//")
            fn = _SAFE_OPS.get(op_name)
            if fn is None:
                raise ValueError(f"Operator {op_name!r} not allowed")
            return fn(_evaluate(l), _evaluate(r))
        case ast.UnaryOp(op=ast.USub(), operand=o):
            return -_evaluate(o)
        case ast.UnaryOp(op=ast.UAdd(), operand=o):
            return +_evaluate(o)
        case ast.Call(func=ast.Name(id=name), args=args):
            fn = _SAFE_OPS.get(name)
            if fn is None:
                raise ValueError(f"Function {name!r} not allowed")
            return fn(*(_evaluate(a) for a in args))
        case ast.Name(id=name):
            val = _SAFE_OPS.get(name)
            if val is None:
                raise ValueError(f"Unknown name: {name!r}")
            if isinstance(val, int | float):
                return val
            raise ValueError(f"{name!r} is not a constant")
        case _:
            raise ValueError(f"Unsupported expression: {ast.dump(node)}")


@tool
def calculator(
    expression: Annotated[str, "A math expression to evaluate, e.g. '2 + 3 * 4' or 'sqrt(16)'."],
) -> str:
    """Evaluate a mathematical expression safely.

    Supports: +, -, *, /, //, %, **, abs, round, min, max, sqrt, log, log10,
    sin, cos, tan. Constants: pi, e.
    """
    expression = expression.strip()
    if not expression:
        return "Error: empty expression"

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        return f"Syntax error: {exc}"

    try:
        result = _evaluate(tree.body)
    except (ValueError, ZeroDivisionError) as exc:
        return f"Error: {exc}"

    if isinstance(result, float) and result == int(result):
        return str(int(result))
    return str(result)

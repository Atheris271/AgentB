"""B4 extension evaluator for tool-call accuracy and token usage.

Run this script from the server ``code`` directory after copying it there.
It reuses ``generate_ai_message`` from ``b4_local_agent_llm.py``.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from b4_local_agent_llm import generate_ai_message


JsonDict = dict[str, Any]


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, data: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def strip_schema_descriptions(tools_schema: list[JsonDict]) -> list[JsonDict]:
    stripped = deepcopy(tools_schema)
    for item in stripped:
        fn = item.get("function") if isinstance(item, dict) else None
        if not isinstance(fn, dict):
            continue
        fn["description"] = ""
        parameters = fn.get("parameters") or {}
        for prop in (parameters.get("properties") or {}).values():
            if isinstance(prop, dict):
                prop["description"] = ""
    return stripped


def tool_names(ai_message: JsonDict) -> list[str]:
    return [str(call.get("name")) for call in ai_message.get("tool_calls") or []]


def score_case(actual: list[str], expected: list[str]) -> JsonDict:
    actual_set = set(actual)
    expected_set = set(expected)
    matched = sorted(actual_set & expected_set)
    missing = sorted(expected_set - actual_set)
    unexpected = sorted(actual_set - expected_set)
    return {
        "ok": not missing and not unexpected,
        "matched": matched,
        "missing": missing,
        "unexpected": unexpected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate B4 tool-call accuracy.")
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--tools_schema", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--mode", default="mock", choices=["mock", "prompt_json"])
    parser.add_argument("--schema_variant", default="full", choices=["full", "stripped"])
    parser.add_argument("--outdir", default="../outputs/B4_extension_eval")
    args = parser.parse_args()

    tools_schema = read_json(args.tools_schema)
    if args.schema_variant == "stripped":
        tools_schema = strip_schema_descriptions(tools_schema)

    cases = read_json(args.cases)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    write_json(outdir / f"tools_schema_{args.schema_variant}.json", tools_schema)

    records: list[JsonDict] = []
    for index, case in enumerate(cases, start=1):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a local tool-using agent. Use available tools when needed. "
                    "If the user asks for multiple independent operations, you may return multiple tool_calls."
                ),
            },
            {"role": "user", "content": case["user"]},
        ]
        result = generate_ai_message(
            args.model_config,
            messages,
            tools_schema,
            args.mode,
            outdir / "cases",
            f"{index:03d}_{case['id']}_{args.schema_variant}",
        )
        ai_message = result.get("ai_message") or {}
        actual_tools = tool_names(ai_message)
        expected_tools = [str(name) for name in case.get("expected_tools", [])]
        score = score_case(actual_tools, expected_tools)
        raw = result.get("raw_model_output") or {}
        records.append(
            {
                "id": case["id"],
                "schema_variant": args.schema_variant,
                "mode": args.mode,
                "expected_tools": expected_tools,
                "actual_tools": actual_tools,
                "score": score,
                "status": result.get("status"),
                "error": result.get("error"),
                "token_usage": raw.get("token_usage"),
            }
        )

    total = len(records)
    success = sum(1 for item in records if item["score"]["ok"])
    token_totals = [
        int((item.get("token_usage") or {}).get("total_tokens") or 0)
        for item in records
    ]
    summary = {
        "schema_variant": args.schema_variant,
        "mode": args.mode,
        "total_cases": total,
        "success_cases": success,
        "success_rate": round(success / total, 4) if total else 0,
        "avg_total_tokens": round(sum(token_totals) / len(token_totals), 2) if token_totals else 0,
        "records": records,
    }
    write_json(outdir / f"summary_{args.schema_variant}_{args.mode}.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

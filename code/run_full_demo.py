from __future__ import annotations

import argparse
import sys
from pathlib import Path

from b1_agent_runtime import run_agent
from common.io_utils import read_json, write_text
from common.path_utils import resolve_cli_path


def _build_report(result: dict, outdir: Path) -> str:
    messages = read_json(outdir / "messages.json")
    trace = read_json(outdir / "trace.json")
    selected = read_json(outdir / "selected_memory.json")
    tools = read_json(outdir / "tools_schema.json")
    roles = " → ".join(message["role"] for message in messages)
    files = sorted(path.relative_to(outdir).as_posix() for path in outdir.rglob("*") if path.is_file())
    return (
        "# Full Agent Demo Report\n\n"
        f"- Conversation: `{result['conversation_id']}`\n"
        f"- Status: `{trace['status']}`\n"
        f"- Message flow: `{roles}`\n"
        f"- Tool rounds: `{trace['tool_rounds_used']}`\n"
        f"- LLM calls: `{trace['llm_call_count']}`\n"
        f"- Loaded memory documents: `{len(selected['selected_memory_docs'])}`\n"
        f"- Available tools: `{len(tools)}`\n\n"
        "## Final Answer\n\n"
        f"{result['final_answer']}\n\n"
        "## Output Files\n\n"
        + "\n".join(f"- `{path}`" for path in files)
        + "\n"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the complete local Agent demonstration.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--tools_config", required=True)
    parser.add_argument("--memory_config", required=True)
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--llm_mode", choices=["mock", "prompt_json"], required=True)
    parser.add_argument("--outdir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        outdir = resolve_cli_path(args.outdir)
        result = run_agent(
            str(resolve_cli_path(args.input)),
            str(resolve_cli_path(args.tools_config)),
            str(resolve_cli_path(args.memory_config)),
            str(resolve_cli_path(args.model_config)),
            str(outdir),
            args.llm_mode,
        )
        write_text(_build_report(result, outdir), outdir / "demo_report.md")
        print(outdir / "demo_report.md")
        return 0
    except Exception as exc:
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

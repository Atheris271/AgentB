from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path
from time import perf_counter

from common.io_utils import append_jsonl, read_json, read_text, read_yaml, write_json, write_text
from common.logging_utils import now_iso
from common.path_utils import resolve_cli_path, resolve_from_file
from common.schemas import validate_ai_message


def _validate_runtime_input(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("runtime_input.json must contain an object")
    execution_mode = payload.setdefault("execution_mode", "integrated")
    if execution_mode not in {"integrated", "fixture"}:
        raise ValueError("execution_mode must be integrated or fixture")
    required = ["conversation_id", "user_input", "system_prompt_path", "toolset", "max_turns", "save_memory"]
    missing = [field for field in required if field not in payload]
    if missing:
        raise ValueError(f"runtime input missing: {', '.join(missing)}")
    if not isinstance(payload["conversation_id"], str) or not payload["conversation_id"]:
        raise ValueError("conversation_id must be a non-empty string")
    if not isinstance(payload["user_input"], str) or not payload["user_input"].strip():
        raise ValueError("user_input must be a non-empty string")
    if not isinstance(payload["max_turns"], int) or isinstance(payload["max_turns"], bool) or payload["max_turns"] < 1:
        raise ValueError("max_turns must be a positive integer")
    if payload["save_memory"] not in {"none", "conversation", "global"}:
        raise ValueError("save_memory must be none, conversation, or global")
    if execution_mode == "fixture":
        fixtures = payload.get("fixtures")
        if not isinstance(fixtures, dict):
            raise ValueError("fixture mode requires a fixtures object")
        required_fixtures = [
            "selected_memory_path",
            "tools_schema_path",
            "ai_messages_path",
            "tool_messages_path",
        ]
        missing_fixtures = [field for field in required_fixtures if not isinstance(fixtures.get(field), str)]
        if missing_fixtures:
            raise ValueError(f"fixtures missing paths: {', '.join(missing_fixtures)}")
        if payload["save_memory"] != "none":
            raise ValueError("fixture mode requires save_memory=none")
    else:
        selected_ids = payload.setdefault("selected_memory_ids", [])
        if not isinstance(selected_ids, list) or not all(isinstance(item, str) for item in selected_ids):
            raise ValueError("selected_memory_ids must be a list of strings")
        payload.setdefault("use_global_memory", False)
        if not isinstance(payload["use_global_memory"], bool):
            raise ValueError("use_global_memory must be boolean")
    return payload


def _memory_context(selected_memory: dict) -> str:
    sections = []
    for document in selected_memory.get("selected_memory_docs", []):
        sections.append(
            f'<memory id="{document["memory_id"]}" type="{document["memory_type"]}">\n'
            f'{document["content"].strip()}\n</memory>'
        )
    return "\n\n".join(sections)


def _estimate_tokens(messages: list[dict]) -> int:
    """Rough token estimate: ~2 chars per token for Chinese/English mixed text."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        total += len(str(content)) // 2
        tool_calls = msg.get("tool_calls", [])
        if tool_calls:
            total += len(str(tool_calls)) // 2
    return max(total, 1)


def _summarize_history(
    messages: list[dict],
    keep_last_n: int,
    model_config_path: str,
    mode: str,
    output_dir: Path,
    call_index: int,
) -> list[dict]:
    """Compress conversation history: summarize older rounds, keep last *keep_last_n* rounds.
    Each "round" = one user+assistant+tool exchange.  Returns a new messages list."""

    # ── collect rounds (user as boundary) ──
    rounds: list[list[dict]] = []
    current: list[dict] = []
    for msg in messages:
        if msg.get("role") == "system":
            continue  # handled separately
        if msg.get("role") == "user" and current:
            rounds.append(current)
            current = [msg]
        else:
            current.append(msg)
    if current:
        rounds.append(current)

    if len(rounds) <= keep_last_n:
        return list(messages)  # nothing to compress

    old_rounds = rounds[:-keep_last_n]
    keep_rounds = rounds[-keep_last_n:]

    # ── flatten old rounds to a single text for summarization ──
    old_text_parts: list[str] = []
    for rnd in old_rounds:
        for msg in rnd:
            role = msg.get("role", "?")
            content = msg.get("content", "")
            if msg.get("tool_calls"):
                names = [tc.get("name", "?") for tc in msg["tool_calls"]]
                content = f"[调用工具: {', '.join(names)}]"
            old_text_parts.append(f"[{role}] {content}")
    old_text = "\n".join(old_text_parts)

    # ── ask LLM to summarize (no tools) ──
    summary_prompt = [
        {"role": "system", "content": "你是一个对话摘要助手。用中文输出摘要，不超过200字。"},
        {"role": "user", "content": (
            "请总结以下对话片段的关键信息，保留事实、用户意图和重要结论，不要遗漏任何关键数据。\n\n"
            + old_text
        )},
    ]
    try:
        result = generate_ai_message(
            model_config_path,
            summary_prompt,
            [],  # no tools — force plain text
            mode,
            str(output_dir / "llm_calls"),
            f"llm_call_{call_index:03d}_compress",
        )
        summary = result.get("ai_message", {}).get("content", "").strip()
        if not summary:
            raise ValueError("empty summary")
    except Exception:
        # Fallback: truncation-based summary — keep first user message as context hint
        first_users = [r[0]["content"] for r in old_rounds if r[0].get("role") == "user"]
        summary = "（历史对话摘要）用户曾提到：" + "；".join(first_users[:3])

    # ── assemble compressed messages ──
    system_msg = messages[0]  # original system prompt
    compressed = [system_msg, {"role": "user", "content": f"[对话历史摘要]\n{summary}"}]
    for rnd in keep_rounds:
        compressed.extend(rnd)
    return compressed


def _default_llm_mode(model_config: Path) -> str:
    config = read_yaml(model_config)
    return config.get("runtime", {}).get("default_mode", "mock")


def generate_ai_message(*args, **kwargs) -> dict:
    """Lazy B4 proxy retained as the integrated-mode injection point."""
    from b4_local_agent_llm import generate_ai_message as b4_generate_ai_message

    return b4_generate_ai_message(*args, **kwargs)


def _load_fixture_inputs(input_file: Path, runtime: dict) -> dict:
    fixtures = runtime["fixtures"]
    selected_memory = read_json(resolve_from_file(fixtures["selected_memory_path"], input_file))
    tools_schema = read_json(resolve_from_file(fixtures["tools_schema_path"], input_file))
    ai_messages = read_json(resolve_from_file(fixtures["ai_messages_path"], input_file))
    tool_messages = read_json(resolve_from_file(fixtures["tool_messages_path"], input_file))
    if not isinstance(selected_memory, dict):
        raise ValueError("preset memory must be a JSON object")
    if not isinstance(tools_schema, list):
        raise ValueError("preset tools_schema must be a JSON array")
    if not isinstance(ai_messages, list) or not ai_messages:
        raise ValueError("preset AI messages must be a non-empty JSON array")
    if not isinstance(tool_messages, dict):
        raise ValueError("preset ToolMessages must be an object keyed by tool_call_id")
    for message in ai_messages:
        validate_ai_message(message)
    return {
        "selected_memory": selected_memory,
        "tools_schema": tools_schema,
        "ai_messages": ai_messages,
        "tool_messages": tool_messages,
    }


def _fixture_tool_messages(tool_calls: list[dict], preset_messages: dict) -> list[dict]:
    results = []
    for call in tool_calls:
        call_id = call.get("id")
        message = deepcopy(preset_messages.get(call_id))
        if not isinstance(message, dict):
            raise ValueError(f"fixture ToolMessage does not exist for tool_call_id: {call_id}")
        if message.get("role") != "tool" or message.get("tool_call_id") != call_id:
            raise ValueError(f"invalid fixture ToolMessage for tool_call_id: {call_id}")
        if message.get("name") != call.get("name"):
            raise ValueError(f"fixture ToolMessage name does not match call: {call_id}")
        results.append(message)
    return results


def run_agent(
    input_path: str | None = None,
    tools_config: str | None = None,
    memory_config: str | None = None,
    model_config: str | None = None,
    outdir: str = "",
    llm_mode: str | None = None,
    interactive: bool = False,
    resume_path: str | None = None,
    system_prompt_path: str = "",
    conversation_id: str = "",
    toolset: str = "basic_tools",
    max_turns: int = 5,
    save_memory: str = "none",
    selected_memory_ids: list[str] | None = None,
    use_global_memory: bool = False,
) -> dict:
    started = perf_counter()
    base_dir = Path(outdir).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    # interactive mode: auto-create per-session subdirectory
    if interactive:
        stamp = now_iso().replace(":", "-")[:19]  # "2026-07-12T09-30-00"
        cid = conversation_id or "conv"
        session_dir = base_dir / f"{cid}_{stamp}"
        session_dir.mkdir(parents=True, exist_ok=True)
        output_dir = session_dir
    else:
        output_dir = base_dir

    # ── resolve which init path we are on ──────────────────────────
    if resume_path:
        init_mode = "resume"
    elif input_path:
        init_mode = "file"
    else:
        init_mode = "standalone"

    # ── load runtime config ────────────────────────────────────────
    if init_mode == "file":
        input_file = Path(input_path).resolve()  # type: ignore[arg-type]
        runtime = _validate_runtime_input(read_json(input_file))
        execution_mode = runtime["execution_mode"]
        selected_memory_ids = runtime.get("selected_memory_ids", [])
        use_global_memory = runtime.get("use_global_memory", False)
        user_text_first = runtime["user_input"]
    else:
        input_file = Path(".")  # fallback for resolve_from_file
        execution_mode = "integrated"
        _sel_ids = selected_memory_ids or []
        _use_global = use_global_memory
        user_text_first = ""
        runtime = {
            "execution_mode": "integrated",
            "conversation_id": conversation_id,
            "user_input": "",
            "system_prompt_path": system_prompt_path,
            "toolset": toolset,
            "max_turns": max_turns,
            "save_memory": save_memory,
            "selected_memory_ids": _sel_ids,
            "use_global_memory": _use_global,
        }

    if init_mode == "resume":
        resume_file = Path(resume_path).resolve()  # type: ignore[arg-type]
        resume_messages = read_json(resume_file)
        if not isinstance(resume_messages, list) or not resume_messages:
            raise ValueError("resume file must contain a non-empty messages array")
        if resume_messages[0].get("role") != "system":
            raise ValueError("resume messages must start with a system message")
        system_prompt = resume_messages[0]["content"]
        runtime["conversation_id"] = resume_file.parent.name or "resumed"
        # try to read trace.json for the real conversation_id
        trace_file = resume_file.with_name("trace.json")
        if trace_file.exists():
            try:
                prev_trace = read_json(trace_file)
                if isinstance(prev_trace, dict) and prev_trace.get("conversation_id"):
                    runtime["conversation_id"] = prev_trace["conversation_id"]
            except Exception:
                pass
        execution_mode = "integrated"
        runtime["execution_mode"] = "integrated"
        runtime["system_prompt_path"] = "n/a (resumed)"
        runtime["toolset"] = "basic_tools"
        # extract last non-system role for user_text_first fallback
        user_text_first = ""
    elif init_mode == "standalone":
        if not interactive:
            raise ValueError("standalone mode requires --interactive")
        if not runtime.get("conversation_id"):
            runtime["conversation_id"] = f"conv_{now_iso().replace(':', '-')}"
        user_text_first = ""
        system_prompt = ""  # loaded below from system_prompt_path

    # ── load system prompt ─────────────────────────────────────────
    if init_mode == "file":
        prompt_path = resolve_from_file(runtime["system_prompt_path"], input_file)
        system_prompt = read_text(prompt_path).strip()
    elif init_mode == "standalone":
        sp_path = runtime.get("system_prompt_path", "")
        if not sp_path:
            raise ValueError(
                "standalone interactive mode requires --system_prompt"
            )
        system_prompt = read_text(Path(sp_path)).strip()

    # ── load model / tools / memory ─────────────────────────────────
    fixture_data = None
    tools_file: Path | None = None
    memory_file: Path | None = None
    model_file: Path | None = None
    selected_memory: dict = {}

    if execution_mode == "fixture":
        fixture_data = _load_fixture_inputs(input_file, runtime)
        selected_memory = fixture_data["selected_memory"]
        tools_schema = fixture_data["tools_schema"]
        mode = "fixture"
    else:
        if not tools_config or not memory_config or not model_config:
            raise ValueError("integrated mode requires tools_config, memory_config, and model_config")
        from b3_tool_layer import execute_tool_calls, get_tools_schema
        from b5_memory import load_memory

        tools_file = Path(tools_config).resolve()
        memory_file = Path(memory_config).resolve()
        model_file = Path(model_config).resolve()
        mode = llm_mode or _default_llm_mode(model_file)

        if init_mode == "resume":
            # resume: skip memory loading (already embedded in system prompt)
            selected_memory = {}
        else:
            selected_memory = load_memory(
                str(memory_file),
                runtime.get("selected_memory_ids", []),
                runtime.get("use_global_memory", False),
                user_text_first,
                str(output_dir),
            )
        tools_schema = get_tools_schema(str(tools_file), runtime["toolset"], str(output_dir))

    # ── build system message ────────────────────────────────────────
    if init_mode != "resume":
        memory_context = _memory_context(selected_memory)
        if memory_context:
            system_prompt = f"{system_prompt}\n\n{memory_context}"
    system_msg = {"role": "system", "content": system_prompt}
    _original_system_prompt = system_prompt  # frozen at startup, for /system - reset

    # ── cross-turn accumulators ─────────────────────────────────────
    if init_mode == "resume":
        # start from the loaded conversation history
        messages: list[dict] = list(resume_messages)
        all_messages = messages
    else:
        messages = []
        all_messages = []
    all_turns: list[dict] = []
    all_tool_messages: list[dict] = []
    llm_calls = 0
    tool_rounds_total = 0
    final_answer = ""
    status = "success"
    terminal_error: dict | None = None
    warnings: list[str] = []
    if selected_memory.get("status") in {"partial", "error"}:
        warnings.append("memory selection completed with errors")

    if interactive:
        cid = runtime.get("conversation_id", "?")
        source = "恢复对话" if init_mode == "resume" else "新对话"
        print(f"\n{'='*60}")
        print(f"  交互对话模式  {source}  conversation_id: {cid}")
        print(f"  /compress [N]  压缩对话历史（保留最近 N 轮，默认 3）")
        print(f"  /exit /quit    退出")
        print(f"{'='*60}\n")

    turn_idx = 0
    while True:
        # ---- outer loop: each iteration = one user turn ----

        # --- get user input ---
        if interactive:
            if init_mode == "file" and turn_idx == 0:
                user_text = user_text_first
            elif init_mode == "standalone" and turn_idx == 0:
                # first turn in standalone: prompt from stdin immediately
                try:
                    user_text = input(">>> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if not user_text or user_text in {"/exit", "/quit"}:
                    break
            else:
                try:
                    user_text = input(">>> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if not user_text or user_text in {"/exit", "/quit"}:
                    break
        else:
            user_text = user_text_first
        # ── special commands ──────────────────────────────────────────
        if interactive and user_text.startswith("/compress"):
            parts = user_text.split()
            keep_n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 3
            old_count = len(messages)
            messages = _summarize_history(
                messages,
                keep_last_n=keep_n,
                model_config_path=str(model_file) if model_file else "",
                mode=mode,
                output_dir=output_dir,
                call_index=llm_calls,
            )
            saved = max(old_count - len(messages), 0)
            saved_tokens = _estimate_tokens(
                [{"role": "x", "content": "x" * 2 * saved * 30}]  # rough
            ) if saved else 0
            print(f"[压缩完成] {old_count} 条 → {len(messages)} 条，保留最近 {keep_n} 轮")
            turn_idx += 1
            continue

        if interactive and user_text.startswith("/system"):
            rest = user_text[len("/system"):].strip()
            if not rest:
                # show current
                cur = messages[0]["content"] if messages else ""
                print(f"[当前 system prompt] ({len(cur)} 字符)\n{cur[:200]}{'...' if len(cur)>200 else ''}")
            elif rest == "-":
                # reset to original
                messages[0]["content"] = _original_system_prompt
                print(f"[已重置] 恢复为启动时的 system prompt ({len(_original_system_prompt)} 字符)")
            elif rest.startswith("+"):
                # append
                append_text = rest[1:].strip()
                if append_text:
                    messages[0]["content"] += "\n\n" + append_text
                    print(f"[已追加] +{len(append_text)} 字符 → 当前 {len(messages[0]['content'])} 字符")
                else:
                    print("[忽略] /system + 后没有内容")
            else:
                # replace from file
                sp_file = Path(rest)
                if not sp_file.is_absolute():
                    # relative to prompts/ directory
                    candidates = [sp_file, Path("..") / sp_file]
                    sp_file = next((c for c in candidates if c.exists()), sp_file)
                if sp_file.exists():
                    new_sp = read_text(sp_file).strip()
                    # re-append memory context if available
                    mc = _memory_context(selected_memory)
                    if mc:
                        new_sp = f"{new_sp}\n\n{mc}"
                    messages[0]["content"] = new_sp
                    print(f"[已切换] system prompt → {sp_file} ({len(new_sp)} 字符)")
                else:
                    print(f"[错误] 文件不存在: {rest}")
            turn_idx += 1
            continue

        print(f"user_input: {user_text}")

        # Build or extend message list for this turn
        if init_mode == "resume" and turn_idx == 0:
            # already have full history from resume file; just append user
            messages.append({"role": "user", "content": user_text})
        elif turn_idx == 0:
            messages = [system_msg, {"role": "user", "content": user_text}]
        else:
            messages.append({"role": "user", "content": user_text})

        # Per-turn state (reset each round)
        turns: list[dict] = []
        turn_tool_messages: list[dict] = []
        tool_rounds = 0
        turn_final_answer = ""
        turn_status = "success"
        turn_terminal_error: dict | None = None

        # ======== tool_calls loop (unchanged logic) ========
        while True:
            llm_calls += 1
            turn_start = perf_counter()
            if execution_mode == "fixture":
                if llm_calls > len(fixture_data["ai_messages"]):
                    raise ValueError("fixture AIMessage sequence ended before a final answer")
                ai_message = deepcopy(fixture_data["ai_messages"][llm_calls - 1])
                llm_status = "success"
                llm_error = None
            else:
                llm_result = generate_ai_message(
                    str(model_file),
                    messages,
                    tools_schema,
                    mode,
                    str(output_dir / "llm_calls"),
                    f"llm_call_{llm_calls:03d}",
                )
                if not isinstance(llm_result, dict) or not isinstance(llm_result.get("ai_message"), dict):
                    raise ValueError("B4 result must contain an ai_message object")
                ai_message = llm_result["ai_message"]
                llm_status = llm_result.get("status")
                llm_error = llm_result.get("error")
            messages.append(ai_message)
            turn = {
                "turn_index": llm_calls,
                "ai_message": ai_message,
                "llm_status": llm_status,
                "llm_error": llm_error,
                "tool_messages": [],
                "latency_ms": None,
            }
            if llm_status != "success":
                turn_status = "llm_parse_error"
                turn_terminal_error = {
                    "type": "LLMParseError",
                    "message": "B4 failed to parse the model output as a valid AIMessage JSON object.",
                    "llm_call_index": llm_calls,
                    "cause": llm_error,
                }
                turn["latency_ms"] = round((perf_counter() - turn_start) * 1000, 3)
                turns.append(turn)
                break
            tool_calls = ai_message.get("tool_calls", [])
            if not tool_calls:
                turn_final_answer = ai_message["content"]
                turn["latency_ms"] = round((perf_counter() - turn_start) * 1000, 3)
                turns.append(turn)
                break
            if tool_rounds >= runtime.get("max_turns", 5):
                requested = ", ".join(call.get("name", "unknown") for call in tool_calls)
                turn_final_answer = (
                    "任务因超过最大工具调用轮次而终止，"
                    f"最后一次模型仍请求调用工具：{requested}。"
                )
                turn_status = "max_turns_exceeded"
                turn_terminal_error = {
                    "type": "MaxTurnsExceeded",
                    "message": turn_final_answer,
                    "unexecuted_tool_calls": tool_calls,
                }
                turn["latency_ms"] = round((perf_counter() - turn_start) * 1000, 3)
                turns.append(turn)
                break
            if execution_mode == "fixture":
                tool_messages = _fixture_tool_messages(
                    tool_calls,
                    fixture_data["tool_messages"],
                )
            else:
                tool_messages = execute_tool_calls(
                    tool_calls,
                    str(tools_file),
                    runtime.get("toolset", "basic_tools"),
                    str(output_dir),
                )
            tool_rounds += 1
            messages.extend(tool_messages)
            turn_tool_messages.extend(tool_messages)
            turn["tool_messages"] = tool_messages
            turn["latency_ms"] = round((perf_counter() - turn_start) * 1000, 3)
            turns.append(turn)
        # ======== end tool_calls loop ========

        # Accumulate across turns
        all_messages = messages  # always the latest (carries full history)
        all_turns.extend(turns)
        all_tool_messages.extend(turn_tool_messages)
        tool_rounds_total += tool_rounds
        final_answer = turn_final_answer
        status = turn_status
        terminal_error = turn_terminal_error

        if turn_final_answer:
            print(f"content: {turn_final_answer}")

        if not interactive:
            break

        if turn_status != "success":
            break

        turn_idx += 1

    write_json(all_messages, output_dir / "messages.json")
    if execution_mode == "integrated":
        write_json(all_tool_messages, output_dir / "tool_messages.json")
    write_text(final_answer.strip() + "\n", output_dir / "final_answer.md")
    save_mem = runtime.get("save_memory", "none")
    memory_save = {"requested": save_mem, "status": "not_requested"}
    if status != "success" and save_mem != "none":
        memory_save = {"requested": save_mem, "status": "skipped", "reason": status}
    conv_id = runtime.get("conversation_id", "")
    trace = {
        "conversation_id": conv_id,
        "execution_mode": execution_mode,
        "status": status,
        "toolset": runtime.get("toolset", "basic_tools"),
        "max_turns": runtime.get("max_turns", 5),
        "tool_rounds_used": tool_rounds_total,
        "llm_call_count": llm_calls,
        "turns": all_turns,
        "final_answer_path": "final_answer.md",
        "memory_save": memory_save,
        "warnings": warnings,
        "error": terminal_error,
    }
    write_json(trace, output_dir / "trace.json")

    saved_memory = None
    if execution_mode == "integrated" and save_mem != "none" and trace["status"] == "success" and memory_file:
        try:
            from b5_memory import save_memory

            saved_memory = save_memory(
                str(memory_file),
                conv_id,
                save_mem,
                str(output_dir / "messages.json"),
                str(output_dir / "trace.json"),
                str(output_dir / "final_answer.md"),
                str(output_dir),
            )
            trace["memory_save"] = {"requested": save_mem, "status": "success"}
        except Exception as exc:
            trace["memory_save"] = {
                "requested": save_mem,
                "status": "error",
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
            trace["warnings"].append("memory save failed")
            if trace["status"] == "success":
                trace["status"] = "partial"
        write_json(trace, output_dir / "trace.json")

    result = {
        "conversation_id": conv_id,
        "execution_mode": execution_mode,
        "status": trace["status"],
        "final_answer": final_answer,
        "messages_path": str(output_dir / "messages.json"),
        "trace_path": str(output_dir / "trace.json"),
        "final_answer_path": str(output_dir / "final_answer.md"),
        "selected_memory": selected_memory,
        "saved_memory": saved_memory,
        "elapsed_ms": round((perf_counter() - started) * 1000, 3),
    }
    if execution_mode == "integrated":
        append_jsonl(
            {
                "timestamp": now_iso(),
                "conversation_id": conv_id,
                "execution_mode": execution_mode,
                "status": trace["status"],
                "llm_mode": mode,
                "tool_rounds_used": tool_rounds_total,
                "llm_call_count": llm_calls,
                "elapsed_ms": result["elapsed_ms"],
            },
            output_dir / "runtime_log.jsonl",
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local Agent message and tool loop.")
    parser.add_argument("--input",
                        help="runtime_input.json 路径（普通模式必填，交互模式可选）")
    parser.add_argument("--tools_config")
    parser.add_argument("--memory_config")
    parser.add_argument("--model_config")
    parser.add_argument("--llm_mode", choices=["mock", "prompt_json"], default=None)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="交互对话模式：用户持续输入消息，对话历史跨轮次保留。")
    parser.add_argument("--system_prompt",
                        help="系统提示词文件路径（无 --input 的交互模式必填）")
    parser.add_argument("--conversation_id", default="",
                        help="对话 ID（无 --input 时使用，默认自动生成）")
    parser.add_argument("--toolset", default="basic_tools",
                        help="工具集名称（默认 basic_tools）")
    parser.add_argument("--max_turns", type=int, default=5,
                        help="每轮最大工具调用次数（默认 5）")
    parser.add_argument("--save_memory", choices=["none", "conversation", "global"], default="none",
                        help="记忆保存模式（默认 none）")
    parser.add_argument("--selected_memory_ids", nargs="*", default=[],
                        help="指定加载的记忆 ID 列表")
    parser.add_argument("--use_global_memory", action="store_true",
                        help="启用全局记忆加载")
    parser.add_argument("--resume",
                        help="从 messages.json 恢复对话，跳过初始化直接进入交互模式。")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        # Validate arg combinations
        if args.resume and args.input:
            print("error: --resume and --input are mutually exclusive", file=sys.stderr)
            return 1
        if args.resume and not args.interactive:
            print("error: --resume requires --interactive", file=sys.stderr)
            return 1
        if not args.interactive and not args.input:
            print("error: non-interactive mode requires --input", file=sys.stderr)
            return 1

        result = run_agent(
            input_path=str(resolve_cli_path(args.input)) if args.input else None,
            tools_config=str(resolve_cli_path(args.tools_config)) if args.tools_config else None,
            memory_config=str(resolve_cli_path(args.memory_config)) if args.memory_config else None,
            model_config=str(resolve_cli_path(args.model_config)) if args.model_config else None,
            outdir=str(resolve_cli_path(args.outdir)),
            llm_mode=args.llm_mode,
            interactive=args.interactive,
            resume_path=str(resolve_cli_path(args.resume)) if args.resume else None,
            system_prompt_path=args.system_prompt,
            conversation_id=args.conversation_id,
            toolset=args.toolset,
            max_turns=args.max_turns,
            save_memory=args.save_memory,
            selected_memory_ids=args.selected_memory_ids,
            use_global_memory=args.use_global_memory,
        )
        print(result["final_answer_path"])
        return 0
    except Exception as exc:
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

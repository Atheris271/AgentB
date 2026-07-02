"""B4: local Agent LLM decision module.

B4 receives ``messages`` and ``tools_schema`` from B1/B3, then returns a
standard AIMessage dictionary. It does not execute tools. It only decides
whether the next assistant message should contain final ``content`` or
``tool_calls`` for B3 to execute.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
CODE_ROOT = Path(__file__).resolve().parent
for item in (ROOT, CODE_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))


JsonDict = dict[str, Any]


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, data: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_jsonl(path: str | Path, record: JsonDict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _minimal_yaml_load(text: str) -> JsonDict:
    """Small fallback parser for this project's simple YAML configs."""

    def strip_comment(line: str) -> str:
        in_quote = False
        quote = ""
        for index, char in enumerate(line):
            if char in {"'", '"'}:
                if not in_quote:
                    in_quote = True
                    quote = char
                elif quote == char:
                    in_quote = False
            elif char == "#" and not in_quote:
                return line[:index].rstrip()
        return line.rstrip()

    def scalar(value: str) -> Any:
        value = value.strip()
        if value == "":
            return {}
        if value in {"true", "True", "TRUE"}:
            return True
        if value in {"false", "False", "FALSE"}:
            return False
        if value in {"null", "None", "~"}:
            return None
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            return value[1:-1]
        if value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            return [] if not inner else [scalar(part) for part in inner.split(",")]
        if re.fullmatch(r"-?\d+", value):
            return int(value)
        if re.fullmatch(r"-?\d+\.\d+", value):
            return float(value)
        return value

    lines = text.splitlines()
    root: JsonDict = {}
    stack: list[tuple[int, Any]] = [(-1, root)]
    for index, raw in enumerate(lines):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        line = strip_comment(raw)
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        content = line.strip()
        while indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if content.startswith("- "):
            if not isinstance(parent, list):
                raise ValueError(f"List item outside list: {raw}")
            parent.append(scalar(content[2:]))
            continue
        if ":" not in content:
            raise ValueError(f"Unsupported YAML line: {raw}")
        key, value = content.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value:
            parent[key] = scalar(value)
            continue
        container: Any = {}
        for next_raw in lines[index + 1 :]:
            if not next_raw.strip() or next_raw.lstrip().startswith("#"):
                continue
            next_indent = len(next_raw) - len(next_raw.lstrip(" "))
            next_content = strip_comment(next_raw).strip()
            if next_indent > indent and next_content.startswith("- "):
                container = []
            break
        parent[key] = container
        stack.append((indent, container))
    return root


def _load_config(path: str | Path) -> JsonDict:
    path = Path(path)
    try:
        from common.config_loader import load_yaml

        return load_yaml(path)
    except Exception:
        text = path.read_text(encoding="utf-8")
        stripped = text.strip()
        if stripped.startswith("{"):
            return json.loads(stripped)
        return _minimal_yaml_load(text)


def _load_messages(messages: str | Path | list[JsonDict]) -> list[JsonDict]:
    if isinstance(messages, (str, Path)):
        messages = _read_json(messages)
    if not isinstance(messages, list):
        raise ValueError("messages must be a JSON array")
    return [dict(item) for item in messages]


def _load_tools_schema(tools_schema: str | Path | list[JsonDict]) -> list[JsonDict]:
    if isinstance(tools_schema, (str, Path)):
        tools_schema = _read_json(tools_schema)
    if not isinstance(tools_schema, list):
        raise ValueError("tools_schema must be a JSON array")
    return [dict(item) for item in tools_schema]


def _tool_names(tools_schema: list[JsonDict]) -> set[str]:
    names = set()
    for item in tools_schema:
        fn = item.get("function") if isinstance(item, dict) else None
        if isinstance(fn, dict) and fn.get("name"):
            names.add(str(fn["name"]))
    return names


def _last_user_text(messages: list[JsonDict]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content") or "")
    return ""


def _tool_messages(messages: list[JsonDict]) -> list[JsonDict]:
    return [msg for msg in messages if msg.get("role") == "tool"]


def _first_available(name: str, names: set[str]) -> str | None:
    return name if name in names else None


def _mock_ai_message(messages: list[JsonDict], tools_schema: list[JsonDict]) -> JsonDict:
    """Deterministic demo mode for environments without a local model."""

    tool_msgs = _tool_messages(messages)
    names = _tool_names(tools_schema)
    user_text = _last_user_text(messages)

    if not tool_msgs:
        file_match = re.search(r"([\w./\\-]+\.(?:txt|md|csv|tsv|json))", user_text)
        if file_match and _first_available("file_reader", names):
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_001",
                        "name": "file_reader",
                        "args": {"path": file_match.group(1).replace("\\", "/"), "max_size_kb": 512},
                    }
                ],
            }

        if re.search(r"\d+\s*[-+*/%^]|\bsqrt\b|平方根|计算", user_text) and _first_available("calculator", names):
            expression = "sqrt(256)" if "平方根" in user_text or "square root" in user_text.lower() else user_text
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_001", "name": "calculator", "args": {"expression": expression}}
                ],
            }

        return {
            "role": "assistant",
            "content": "当前问题不需要调用工具，我可以直接回答。请提供更具体的本地文件、表格或计算任务。",
            "tool_calls": [],
        }

    summaries: list[str] = []
    for tool_msg in tool_msgs:
        name = tool_msg.get("name", "tool")
        content = str(tool_msg.get("content") or "")
        try:
            payload = json.loads(content)
            if payload.get("status") == "success":
                output = payload.get("output")
                if isinstance(output, dict) and "result" in output:
                    text = str(output["result"])
                else:
                    text = json.dumps(output, ensure_ascii=False)
                summaries.append(f"{name} 返回成功：{text}")
            else:
                summaries.append(f"{name} 调用失败：{payload.get('error')}")
        except json.JSONDecodeError:
            summaries.append(f"{name} 返回：{content}")
    return {
        "role": "assistant",
        "content": "根据工具返回结果，结论如下：\n" + "\n".join(f"- {item}" for item in summaries),
        "tool_calls": [],
    }


def _message_to_prompt_line(message: JsonDict) -> str:
    role = message.get("role", "unknown")
    payload = {key: value for key, value in message.items() if key != "role"}
    return f"{role}: {json.dumps(payload, ensure_ascii=False)}"


def build_prompt(messages: list[JsonDict], tools_schema: list[JsonDict]) -> str:
    return (
        "You are a local tool-using agent decision module.\n"
        "You must output only one valid JSON object and no markdown fences.\n"
        "Do not output <think>, chain-of-thought, or hidden reasoning.\n"
        "If a tool is needed, return exactly this shape:\n"
        '{"role":"assistant","content":"","tool_calls":[{"id":"call_001","name":"tool_name","args":{}}]}\n'
        "If no tool is needed, return exactly this shape:\n"
        '{"role":"assistant","content":"final answer","tool_calls":[]}\n\n'
        "Available tools_schema:\n"
        f"{json.dumps(tools_schema, ensure_ascii=False, indent=2)}\n\n"
        "Conversation messages:\n"
        + "\n".join(_message_to_prompt_line(message) for message in messages)
        + "\n\nReturn JSON now:"
    )


def _strip_think_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_json_object(text: str) -> JsonDict:
    text = _strip_fences(_strip_think_blocks(text))
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start < 0:
        raise ValueError("model output does not contain a JSON object")
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if escape:
            escape = False
            continue
        if char == "\\" and in_string:
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                parsed = json.loads(text[start : index + 1])
                if not isinstance(parsed, dict):
                    raise ValueError("parsed JSON is not an object")
                return parsed
    raise ValueError("could not find a complete JSON object")


def _normalize_tool_call(call: Any, index: int) -> JsonDict:
    if not isinstance(call, dict):
        raise ValueError(f"tool_calls[{index}] is not an object")
    if isinstance(call.get("function"), dict):
        fn = call["function"]
        name = fn.get("name", "")
        args = fn.get("arguments", {})
    else:
        name = call.get("name") or call.get("tool_name") or ""
        args = call.get("args", call.get("arguments", {}))
    if isinstance(args, str):
        args = json.loads(args)
    if not isinstance(args, dict):
        raise ValueError(f"tool_calls[{index}].args must be an object")
    if not name:
        raise ValueError(f"tool_calls[{index}].name is missing")
    return {
        "id": str(call.get("id") or f"call_{index + 1:03d}"),
        "name": str(name),
        "args": args,
    }


def parse_ai_message(raw_text: str) -> tuple[JsonDict, JsonDict]:
    """Parse raw model text into a standard AIMessage dict."""

    payload = _extract_json_object(raw_text)
    if "choices" in payload and isinstance(payload["choices"], list):
        message = payload["choices"][0].get("message", {})
        if isinstance(message, dict):
            payload = message

    tool_calls_raw = payload.get("tool_calls") or []
    if tool_calls_raw and not isinstance(tool_calls_raw, list):
        raise ValueError("tool_calls must be a list")
    tool_calls = [
        _normalize_tool_call(call, index)
        for index, call in enumerate(tool_calls_raw)
    ]
    content = payload.get("content", "")
    if content is None:
        content = ""
    if tool_calls:
        content = ""
    if not tool_calls and not str(content).strip():
        raise ValueError("AIMessage must contain either content or tool_calls")

    ai_message = {
        "role": "assistant",
        "content": str(content),
        "tool_calls": tool_calls,
    }
    raw_record = {
        "status": "success",
        "raw_text": raw_text,
        "parsed": ai_message,
        "error": None,
    }
    return ai_message, raw_record


def _run_transformers_prompt(prompt: str, model_cfg: JsonDict) -> str:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "prompt_json mode requires torch and transformers. Install dependencies "
            "and make sure the local model path in model.yaml is correct."
        ) from exc

    model_section = model_cfg.get("model") or {}
    generation = model_section.get("generation") or model_cfg.get("generation") or {}
    context = model_section.get("context") or model_cfg.get("context") or {}
    model_path = (
        model_section.get("model_name_or_path")
        or model_section.get("model_name")
        or model_section.get("model_path")
    )
    tokenizer_path = model_section.get("tokenizer_name_or_path") or model_path
    if not model_path:
        raise ValueError("model.yaml must set model.model_name_or_path for prompt_json mode")

    dtype_name = str(model_section.get("torch_dtype") or "auto")
    dtype = "auto"
    if dtype_name == "bfloat16":
        dtype = torch.bfloat16
    elif dtype_name == "float16":
        dtype = torch.float16
    elif dtype_name == "float32":
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=bool(model_section.get("local_files_only", True)),
        trust_remote_code=bool(model_section.get("trust_remote_code", True)),
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=bool(model_section.get("local_files_only", True)),
        trust_remote_code=bool(model_section.get("trust_remote_code", True)),
        torch_dtype=dtype,
        device_map=model_section.get("device_map", "auto"),
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    max_input_tokens = int(context.get("max_input_tokens") or 4096)
    if inputs["input_ids"].shape[-1] > max_input_tokens:
        inputs["input_ids"] = inputs["input_ids"][:, -max_input_tokens:]
        if "attention_mask" in inputs:
            inputs["attention_mask"] = inputs["attention_mask"][:, -max_input_tokens:]
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=int(generation.get("max_new_tokens") or 1024),
            do_sample=bool(generation.get("do_sample", False)),
            temperature=float(generation.get("temperature", 0) or 0),
            top_p=float(generation.get("top_p", 1) or 1),
            pad_token_id=tokenizer.eos_token_id,
        )
    output_ids = generated[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(output_ids, skip_special_tokens=True)


def generate_ai_message(
    model_config: str | Path | JsonDict,
    messages: str | Path | list[JsonDict],
    tools_schema: str | Path | list[JsonDict],
    *,
    mode: str | None = None,
    outdir: str | Path | None = None,
) -> JsonDict:
    """Public B1-facing API: generate one AIMessage dict."""

    model_cfg = _load_config(model_config) if isinstance(model_config, (str, Path)) else dict(model_config)
    message_list = _load_messages(messages)
    schema = _load_tools_schema(tools_schema)
    selected_mode = mode or (model_cfg.get("runtime") or {}).get("default_mode") or "prompt_json"
    out = Path(outdir) if outdir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    prompt_text = build_prompt(message_list, schema)
    raw_record: JsonDict

    try:
        if selected_mode == "mock":
            ai_message = _mock_ai_message(message_list, schema)
            raw_text = json.dumps(ai_message, ensure_ascii=False)
            raw_record = {
                "status": "success",
                "mode": "mock",
                "raw_text": raw_text,
                "parsed": ai_message,
                "error": None,
            }
        elif selected_mode == "prompt_json":
            raw_text = _run_transformers_prompt(prompt_text, model_cfg)
            ai_message, raw_record = parse_ai_message(raw_text)
            raw_record["mode"] = "prompt_json"
        else:
            raise ValueError(f"unsupported B4 mode: {selected_mode}")
    except Exception as exc:
        ai_message = {
            "role": "assistant",
            "content": "",
            "tool_calls": [],
            "status": "error",
            "error": str(exc),
        }
        raw_record = {
            "status": "error",
            "mode": selected_mode,
            "raw_text": "",
            "parsed": None,
            "error": str(exc),
        }

    raw_record["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)

    if out:
        _write_json(out / "raw_model_output.json", raw_record)
        _write_json(out / "ai_message.json", ai_message)
        (out / "prompt_text.txt").write_text(prompt_text, encoding="utf-8")
        _append_jsonl(
            out / "llm_run_log.jsonl",
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "mode": selected_mode,
                "status": raw_record["status"],
                "latency_ms": raw_record["latency_ms"],
                "message_count": len(message_list),
                "tool_count": len(schema),
                "error": raw_record.get("error"),
            },
        )
    return ai_message


def main() -> None:
    parser = argparse.ArgumentParser(description="B4 local Agent LLM decision module")
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--messages", required=True)
    parser.add_argument("--tools_schema", required=True)
    parser.add_argument("--mode", default=None, choices=["mock", "prompt_json"])
    parser.add_argument("--outdir", default="../outputs/B4_llm/demo")
    args = parser.parse_args()

    ai_message = generate_ai_message(
        args.model_config,
        args.messages,
        args.tools_schema,
        mode=args.mode,
        outdir=args.outdir,
    )
    status = ai_message.get("status", "success")
    print(json.dumps({"status": status, "ai_message": ai_message}, ensure_ascii=False))


if __name__ == "__main__":
    main()

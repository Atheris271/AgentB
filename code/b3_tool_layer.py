"""B3: tool schema generation and tool-call execution layer.

This module bridges B2 skills and B4 LLM decisions:

- read ``tools.yaml`` and generate OpenAI-style ``tools_schema``;
- receive ``AIMessage.tool_calls`` from B4;
- validate tool names and arguments;
- execute the matching B2 skill;
- return standard ToolMessage dictionaries for B1.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import re
import sys
import time
import types
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, get_args, get_origin

ROOT = Path(__file__).resolve().parent.parent
CODE_ROOT = Path(__file__).resolve().parent
for item in (ROOT, CODE_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

try:
    from common.io_utils import read_yaml as load_yaml
except ImportError:
    import yaml

    def load_yaml(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}


JsonDict = dict[str, Any]


class _SimpleTool:
    """Small runtime shim used only when langchain_core is not installed."""

    def __init__(self, func: Callable[..., Any]):
        self.func = func
        self.name = func.__name__
        self.description = inspect.getdoc(func) or ""

    def invoke(self, args: JsonDict) -> Any:
        return self.func(**args)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)


def _install_langchain_tool_shim_if_needed() -> None:
    try:
        import langchain_core.tools  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    langchain_core = sys.modules.setdefault("langchain_core", types.ModuleType("langchain_core"))
    tools_module = types.ModuleType("langchain_core.tools")

    def tool(func: Callable[..., Any] | None = None, **_: Any) -> Any:
        if func is None:
            return lambda real_func: _SimpleTool(real_func)
        return _SimpleTool(func)

    tools_module.BaseTool = _SimpleTool
    tools_module.tool = tool
    sys.modules["langchain_core.tools"] = tools_module
    setattr(langchain_core, "tools", tools_module)


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


def _load_tools_config(tools_config: str | Path) -> tuple[JsonDict, Path]:
    path = Path(tools_config).resolve()
    data = load_yaml(path)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return data, path


def _tool_names(config: JsonDict, toolset: str | None) -> tuple[str, list[str]]:
    if "toolsets" in config:
        selected = toolset or config.get("default_toolset") or "basic_tools"
        names = config.get("toolsets", {}).get(selected)
        if names is None:
            raise ValueError(f"Unknown toolset: {selected}")
        return selected, [str(name) for name in names]

    tools_section = config.get("tools") or {}
    enabled = tools_section.get("enabled") or []
    selected = toolset or "basic_tools"
    return selected, [str(name) for name in enabled]


def _settings(config: JsonDict) -> JsonDict:
    if "settings" in config:
        return config.get("settings") or {}
    return (config.get("tools") or {}).get("settings") or {}


def _tool_config(config: JsonDict, name: str) -> JsonDict:
    explicit = (config.get("tools") or {}).get(name)
    if isinstance(explicit, dict):
        return explicit
    return {
        "module": f"skills.{name}",
        "function": name,
    }


def _json_type(annotation: Any) -> str:
    if annotation is inspect._empty:
        return "string"
    if isinstance(annotation, str):
        lowered = annotation.lower()
        if "bool" in lowered:
            return "boolean"
        if "int" in lowered:
            return "integer"
        if "float" in lowered:
            return "number"
        if "list" in lowered or "tuple" in lowered or "set" in lowered:
            return "array"
        if "dict" in lowered:
            return "object"
        return "string"
    origin = get_origin(annotation)
    if origin is not None and str(origin).endswith("Annotated"):
        return _json_type(get_args(annotation)[0])
    if origin in {list, tuple, set}:
        return "array"
    if origin is dict:
        return "object"
    if origin is not None:
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(args) == 1:
            return _json_type(args[0])
    if annotation in {str, "str"}:
        return "string"
    if annotation in {int, "int"}:
        return "integer"
    if annotation in {float, "float"}:
        return "number"
    if annotation in {bool, "bool"}:
        return "boolean"
    return "string"


def _annotation_description(annotation: Any) -> str:
    if isinstance(annotation, str):
        matches = re.findall(r"['\"]([^'\"]+)['\"]", annotation, flags=re.DOTALL)
        return " ".join(matches)
    origin = get_origin(annotation)
    if origin is not None and str(origin).endswith("Annotated"):
        extras = get_args(annotation)[1:]
        text = " ".join(str(item) for item in extras if isinstance(item, str))
        return text
    return ""


def _schema_from_signature(func: Callable[..., Any]) -> tuple[JsonDict, list[str]]:
    properties: JsonDict = {}
    required: list[str] = []
    signature = inspect.signature(func)
    for name, parameter in signature.parameters.items():
        annotation = parameter.annotation
        properties[name] = {
            "type": _json_type(annotation),
            "description": _annotation_description(annotation) or f"Argument `{name}`.",
        }
        if parameter.default is inspect._empty:
            required.append(name)
    return properties, required


def _schema_from_langchain_tool(tool_obj: Any) -> tuple[JsonDict, list[str]]:
    args_schema = getattr(tool_obj, "args_schema", None)
    if args_schema is not None:
        if hasattr(args_schema, "model_json_schema"):
            schema = args_schema.model_json_schema()
        elif hasattr(args_schema, "schema"):
            schema = args_schema.schema()
        else:
            schema = {}
        properties = schema.get("properties") or {}
        required = schema.get("required") or []
        return properties, [str(item) for item in required]

    func = getattr(tool_obj, "func", None)
    if callable(func):
        return _schema_from_signature(func)
    if callable(tool_obj):
        return _schema_from_signature(tool_obj)
    return {}, []


def _load_skill(tool_cfg: JsonDict, global_settings: JsonDict | None = None) -> Any:
    _install_langchain_tool_shim_if_needed()
    module_name = str(tool_cfg.get("module") or "")
    function_name = str(tool_cfg.get("function") or "")
    if not module_name or not function_name:
        raise ValueError("tool config must contain module and function")

    module = importlib.import_module(module_name)
    if global_settings:
        workspace_root = global_settings.get("workspace_root") or global_settings.get("data_root")
        if workspace_root and hasattr(module, "set_workspace_root"):
            root_path = Path(workspace_root)
            if not root_path.is_absolute():
                root_path = (ROOT / root_path).resolve()
            module.set_workspace_root(root_path)

    skill = getattr(module, function_name)
    if not callable(skill) and not hasattr(skill, "invoke"):
        raise TypeError(f"{module_name}.{function_name} is not callable")
    return skill


def _property_schema(value: Any) -> JsonDict:
    if isinstance(value, dict):
        result = {
            key: deepcopy(value[key])
            for key in ("type", "description", "enum", "items", "default", "minimum", "maximum")
            if key in value
        }
        result.setdefault("type", "string")
        result.setdefault("description", "")
        return result
    return {"type": "string", "description": str(value)}


def _required_from_config(tool_cfg: JsonDict, parameters: JsonDict) -> list[str]:
    required: list[str] = []
    if isinstance(tool_cfg.get("required"), list):
        required.extend(str(item) for item in tool_cfg["required"])
    for name, spec in parameters.items():
        if isinstance(spec, dict) and spec.get("required") is True:
            required.append(name)
    return sorted(set(required))


def build_tools_schema(
    tools_config: str | Path,
    toolset: str | None = None,
    *,
    auto_schema: bool = True,
) -> tuple[list[JsonDict], JsonDict]:
    config, config_path = _load_tools_config(tools_config)
    selected_toolset, names = _tool_names(config, toolset)
    settings = _settings(config)

    schema: list[JsonDict] = []
    report: JsonDict = {
        "status": "success",
        "tools_config": str(config_path),
        "toolset": selected_toolset,
        "tool_count": 0,
        "tools": [],
        "warnings": [],
    }

    for name in names:
        tool_cfg = _tool_config(config, name)
        description = str(tool_cfg.get("description") or "")
        parameters = deepcopy(tool_cfg.get("parameters") or {})
        required = _required_from_config(tool_cfg, parameters)
        tool_report: JsonDict = {"name": name, "available": True, "warnings": []}

        if auto_schema:
            try:
                skill = _load_skill(tool_cfg, settings)
                description = description or str(getattr(skill, "description", "") or inspect.getdoc(skill) or "")
                inferred_properties, inferred_required = _schema_from_langchain_tool(skill)
                if not parameters:
                    parameters = inferred_properties
                    required = inferred_required
            except Exception as exc:
                tool_report["available"] = False
                tool_report["warnings"].append(str(exc))
                report["warnings"].append(f"{name}: {exc}")

        properties = {key: _property_schema(value) for key, value in parameters.items()}
        schema_item: JsonDict = {
            "type": "function",
            "function": {
                "name": name,
                "description": description or f"Call skill `{name}`.",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            },
        }
        if "returns" in tool_cfg:
            schema_item["function"]["x-returns"] = deepcopy(tool_cfg["returns"])
        schema.append(schema_item)
        tool_report["parameters"] = list(properties)
        tool_report["required"] = required
        report["tools"].append(tool_report)

    report["tool_count"] = len(schema)
    return schema, report


def get_tools_schema(
    tools_config: str | Path,
    toolset: str | None = None,
    outdir: str | Path | None = None,
    auto_schema: bool = True,
) -> list[JsonDict]:
    """Public B1-facing API: return tools_schema and optionally save it."""

    schema, report = build_tools_schema(tools_config, toolset, auto_schema=auto_schema)
    if outdir:
        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)
        _write_json(out / "tools_schema.json", schema)
        _write_json(out / "tool_schema_report.json", report)
    return schema


def _schema_index(schema: list[JsonDict]) -> dict[str, JsonDict]:
    result: dict[str, JsonDict] = {}
    for item in schema:
        fn = item.get("function") if isinstance(item, dict) else None
        if isinstance(fn, dict) and fn.get("name"):
            result[str(fn["name"])] = fn
    return result


def _normalize_tool_calls(payload: Any) -> list[JsonDict]:
    raw_calls = payload.get("tool_calls", []) if isinstance(payload, dict) else payload
    if not isinstance(raw_calls, list):
        raise ValueError("tool_calls must be a list or an AIMessage object")

    calls: list[JsonDict] = []
    for index, call in enumerate(raw_calls, start=1):
        if not isinstance(call, dict):
            calls.append({"id": f"call_{index:03d}", "name": "", "args": {}, "raw": call})
            continue
        if isinstance(call.get("function"), dict):
            fn = call["function"]
            name = fn.get("name") or ""
            args = fn.get("arguments") or {}
        else:
            name = call.get("name") or call.get("tool_name") or ""
            args = call.get("args", call.get("arguments", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"_raw_arguments": args}
        calls.append(
            {
                "id": str(call.get("id") or f"call_{index:03d}"),
                "name": str(name),
                "args": args if isinstance(args, dict) else {"value": args},
                "raw": call,
            }
        )
    return calls


def _type_ok(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    return True


def _validate_args(name: str, args: JsonDict, fn_schema: JsonDict) -> list[str]:
    parameters = fn_schema.get("parameters") or {}
    properties = parameters.get("properties") or {}
    required = parameters.get("required") or []
    errors: list[str] = []

    for key in required:
        if key not in args:
            errors.append(f"missing required argument `{key}`")
    for key, value in args.items():
        if key not in properties:
            if parameters.get("additionalProperties") is False:
                errors.append(f"unexpected argument `{key}`")
            continue
        expected = str(properties[key].get("type") or "")
        if expected and not _type_ok(value, expected):
            errors.append(f"argument `{key}` for {name} should be {expected}, got {type(value).__name__}")
    return errors


def _skill_error(name: str, args: JsonDict, error: str, latency_ms: float = 0.0) -> JsonDict:
    return {
        "skill_name": name,
        "status": "error",
        "input": args,
        "output": None,
        "error": error,
        "latency_ms": round(latency_ms, 3),
    }


def _wrap_result(name: str, args: JsonDict, result: Any, latency_ms: float) -> JsonDict:
    if isinstance(result, dict) and {"status", "output"}.issubset(result):
        wrapped = deepcopy(result)
        wrapped.setdefault("skill_name", name)
        wrapped.setdefault("input", args)
        wrapped.setdefault("error", None)
        wrapped["latency_ms"] = round(float(wrapped.get("latency_ms", latency_ms)), 3)
        return wrapped
    return {
        "skill_name": name,
        "status": "success",
        "input": args,
        "output": {"result": result},
        "error": None,
        "latency_ms": round(latency_ms, 3),
    }


def _invoke_skill(skill: Any, name: str, args: JsonDict) -> JsonDict:
    start = time.perf_counter()
    try:
        if hasattr(skill, "invoke"):
            result = skill.invoke(args)
        else:
            result = skill(**args)
        return _wrap_result(name, args, result, (time.perf_counter() - start) * 1000)
    except Exception as exc:
        return _skill_error(name, args, str(exc), (time.perf_counter() - start) * 1000)


def _cache_key(name: str, args: JsonDict) -> str:
    text = json.dumps({"name": name, "args": args}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_cache(path: Path) -> JsonDict:
    if not path.exists():
        return {}
    try:
        data = _read_json(path)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _tool_message(call: JsonDict, skill_result: JsonDict) -> JsonDict:
    return {
        "role": "tool",
        "tool_call_id": call["id"],
        "name": call["name"],
        "content": json.dumps(skill_result, ensure_ascii=False),
        "status": skill_result.get("status"),
    }


def _write_stats(outdir: Path, records: list[JsonDict]) -> None:
    stats: dict[str, JsonDict] = {}
    for record in records:
        name = record.get("name") or "unknown"
        item = stats.setdefault(name, {"calls": 0, "success": 0, "error": 0, "latencies": []})
        item["calls"] += 1
        status = (record.get("skill_result") or {}).get("status")
        if status == "success":
            item["success"] += 1
        else:
            item["error"] += 1
        item["latencies"].append(float((record.get("skill_result") or {}).get("latency_ms") or 0))
    for item in stats.values():
        latencies = item.pop("latencies")
        item["avg_latency_ms"] = round(sum(latencies) / len(latencies), 3) if latencies else 0
        item["failure_rate"] = round(item["error"] / item["calls"], 4) if item["calls"] else 0
    _write_json(outdir / "tool_stats.json", stats)


def execute_tool_calls(
    ai_message_or_tool_calls: Any,
    tools_config: str | Path,
    toolset: str | None = None,
    outdir: str | Path | None = None,
    retry: int = 0,
    use_cache: bool = False,
) -> list[JsonDict]:
    """Public B1-facing API: execute B4 tool_calls and return ToolMessages."""

    config, config_path = _load_tools_config(tools_config)
    selected_toolset, names = _tool_names(config, toolset)
    settings = _settings(config)
    schema, _ = build_tools_schema(config_path, selected_toolset)
    schema_by_name = _schema_index(schema)
    calls = _normalize_tool_calls(ai_message_or_tool_calls)

    out = Path(outdir) if outdir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    cache_path = out / "tool_cache.json" if out else None
    cache = _read_cache(cache_path) if use_cache and cache_path else {}

    messages: list[JsonDict] = []
    records: list[JsonDict] = []
    for call in calls:
        name = call["name"]
        args = call["args"]
        cached = False

        if name not in names:
            skill_result = _skill_error(name, args, f"tool `{name}` is not in toolset `{selected_toolset}`")
        elif name not in schema_by_name:
            skill_result = _skill_error(name, args, f"tool `{name}` has no generated schema")
        else:
            errors = _validate_args(name, args, schema_by_name[name])
            if errors:
                skill_result = _skill_error(name, args, "; ".join(errors))
            else:
                key = _cache_key(name, args)
                if use_cache and key in cache:
                    skill_result = deepcopy(cache[key])
                    skill_result["cached"] = True
                    cached = True
                else:
                    skill = _load_skill(_tool_config(config, name), settings)
                    attempts = max(0, retry) + 1
                    skill_result = _skill_error(name, args, "not executed")
                    for attempt in range(1, attempts + 1):
                        skill_result = _invoke_skill(skill, name, args)
                        skill_result["attempt"] = attempt
                        if skill_result.get("status") == "success":
                            break
                    if use_cache and cache_path and skill_result.get("status") == "success":
                        cache[key] = deepcopy(skill_result)

        message = _tool_message(call, skill_result)
        messages.append(message)
        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "toolset": selected_toolset,
            "id": call["id"],
            "name": name,
            "args": args,
            "cached": cached,
            "skill_result": skill_result,
            "tool_message": message,
        }
        records.append(record)
        if out:
            _append_jsonl(out / "tool_call_log.jsonl", record)

    if out:
        _write_json(out / "tool_messages.json", messages)
        _write_stats(out, records)
        if use_cache and cache_path:
            _write_json(cache_path, cache)
    return messages


def main() -> None:
    parser = argparse.ArgumentParser(description="B3 tool layer")
    parser.add_argument("--tools_config", required=True)
    parser.add_argument("--toolset", default=None)
    parser.add_argument("--export_schema", action="store_true")
    parser.add_argument("--tool_calls", default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--outdir", default="../outputs/B3_tools")
    parser.add_argument("--retry", type=int, default=0)
    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument("--no_auto_schema", action="store_true")
    args = parser.parse_args()

    if args.export_schema or not args.execute:
        schema = get_tools_schema(
            args.tools_config,
            args.toolset,
            outdir=args.outdir,
            auto_schema=not args.no_auto_schema,
        )
        print(json.dumps({"status": "success", "tools": len(schema)}, ensure_ascii=False))

    if args.execute:
        if not args.tool_calls:
            raise SystemExit("--tool_calls is required with --execute")
        payload = _read_json(args.tool_calls)
        messages = execute_tool_calls(
            payload,
            args.tools_config,
            args.toolset,
            outdir=args.outdir,
            retry=args.retry,
            use_cache=args.use_cache,
        )
        print(json.dumps({"status": "success", "tool_messages": len(messages)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

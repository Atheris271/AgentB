# B1 + B2 对话流程：函数级执行详解

> 本文档以 `integrated`（集成）模式为例，追踪一次完整的 Agent 对话流程中 B1（Agent 运行时）与 B2（Skill 调度器）的每一个函数调用。

---

## 架构概览

```
CLI
 └─ B1:main()
      └─ B1:run_agent()           ← 核心编排循环
           ├─ B4:generate_ai_message()   LLM 推理
           ├─ B3:execute_tool_calls()    工具调用分发
           │    └─ B2:run_skill()        Skill 反射调度
           │         └─ skills/*.py      Skill 具体实现
           └─ B5:load_memory() / save_memory()   记忆管理
```

**调用链深度：** B1 → B3 → B2 → Skill 实现（四层）

---

## 阶段 0：启动与初始化

### `B1:main()` — 入口

**文件：** `code/b1_agent_runtime.py:352`

```python
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)     # 解析 CLI 参数
    result = run_agent(                         # 进入核心逻辑
        str(resolve_cli_path(args.input)),
        str(resolve_cli_path(args.tools_config)) if args.tools_config else None,
        str(resolve_cli_path(args.memory_config)) if args.memory_config else None,
        str(resolve_cli_path(args.model_config)) if args.model_config else None,
        str(resolve_cli_path(args.outdir)),
        args.llm_mode,
    )
    print(result["final_answer_path"])
    return 0
```

CLI 参数：

| 参数 | 说明 |
|------|------|
| `--input` | 运行时输入 JSON 路径（对话配置） |
| `--tools_config` | 工具配置文件（`configs/tools.yaml`） |
| `--memory_config` | 记忆配置文件（`configs/memory.yaml`） |
| `--model_config` | 模型配置文件（`configs/model.yaml`） |
| `--llm_mode` | LLM 模式：`mock` 或 `prompt_json` |
| `--outdir` | 输出目录 |

---

### `B1:run_agent()` — 核心编排

**文件：** `code/b1_agent_runtime.py:119`

**签名：**
```python
def run_agent(
    input_path: str,
    tools_config: str | None,
    memory_config: str | None,
    model_config: str | None,
    outdir: str,
    llm_mode: str | None = None,
) -> dict:
```

#### 步骤 0.1：初始化环境

```python
started = perf_counter()                              # 开始计时
input_file = Path(input_path).resolve()               # 解析输入文件路径
output_dir = Path(outdir).resolve()                   # 解析输出目录路径
output_dir.mkdir(parents=True, exist_ok=True)         # 创建输出目录
```

#### 步骤 0.2：加载并校验运行时输入

```python
runtime = _validate_runtime_input(read_json(input_file))
```

调用链：

```
read_json(input_file)                     common/io_utils.py — 读取 JSON 文件
  └─ _validate_runtime_input(payload)     b1_agent_runtime.py:15
```

**`_validate_runtime_input()` 校验项：**

| 字段 | 约束 |
|------|------|
| `execution_mode` | 必须是 `"integrated"` 或 `"fixture"`，默认 `"integrated"` |
| `conversation_id` | 非空字符串 |
| `user_input` | 非空字符串 |
| `system_prompt_path` | 必须存在 |
| `max_turns` | 正整数（≥1） |
| `save_memory` | `"none"` / `"conversation"` / `"global"` |
| `toolset` | 工具集名称 |
| `[fixture 模式] fixtures` | 必须含 4 个预设文件路径 |
| `[integrated 模式] selected_memory_ids` | 字符串列表；`use_global_memory` 必须为 bool |

返回值：校验后的 `runtime` dict（已填充默认值）。

#### 步骤 0.3：分支 — integrated vs fixture

```python
execution_mode = runtime["execution_mode"]
```

**Integrated 路径：**
```python
# 懒加载 B3、B5（仅在 integrated 模式导入）
from b3_tool_layer import execute_tool_calls, get_tools_schema
from b5_memory import load_memory

# 加载记忆文档
selected_memory = load_memory(
    str(memory_file),
    runtime["selected_memory_ids"],
    runtime["use_global_memory"],
    runtime["user_input"],
    str(output_dir),
)

# 生成 OpenAI function-calling 工具 schema
tools_schema = get_tools_schema(
    str(tools_file),
    runtime["toolset"],
    str(output_dir),
)

# 确定 LLM 模式
mode = llm_mode or _default_llm_mode(model_file)
```

**`_default_llm_mode()`**（`:68`）：
```python
def _default_llm_mode(model_config: Path) -> str:
    config = read_yaml(model_config)
    return config.get("runtime", {}).get("default_mode", "mock")
```
从 `configs/model.yaml` 的 `runtime.default_mode` 读取，若未配置则回退到 `"mock"`。

#### 步骤 0.4：构建系统提示词 + 记忆上下文

```python
system_prompt = read_text(prompt_path).strip()

# 将记忆文档拼接为 XML 块
memory_context = _memory_context(selected_memory)
if memory_context:
    system_prompt = f"{system_prompt}\n\n{memory_context}"
```

**`_memory_context()`**（`:58`）：
```python
def _memory_context(selected_memory: dict) -> str:
    sections = []
    for document in selected_memory.get("selected_memory_docs", []):
        sections.append(
            f'<memory id="{document["memory_id"]}" type="{document["memory_type"]}">\n'
            f'{document["content"].strip()}\n</memory>'
        )
    return "\n\n".join(sections)
```

#### 步骤 0.5：初始化对话消息

```python
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": runtime["user_input"]},
]
```

#### 步骤 0.6：初始化循环状态

```python
tool_rounds = 0       # 已执行的工具调用轮次
llm_calls = 0          # LLM 调用次数
turns = []              # 每轮详细信息
all_tool_messages = []  # 所有工具消息
final_answer = ""       # 最终回复文本
status = "success"      # 全局状态
terminal_error = None   # 终止错误
warnings = []           # 警告列表
```

---

## 阶段 1：主循环 — 每轮迭代

### `while True:` — 无限循环，内部条件 break

**文件：** `code/b1_agent_runtime.py:179`

```
while True:
    llm_calls += 1          ← 计数器递增
    turn_start = perf_counter()  ← 计时本轮
```

---

### 1a. 调用 LLM 生成回复

```python
ai_message = ...

# Integrated 路径：
llm_result = generate_ai_message(
    str(model_file),
    messages,
    tools_schema,
    mode,
    str(output_dir / "llm_calls"),
    f"llm_call_{llm_calls:03d}",
)
```

**`generate_ai_message()`**（`:73`）是一个懒加载代理：
```python
def generate_ai_message(*args, **kwargs) -> dict:
    """Lazy B4 proxy retained as the integrated-mode injection point."""
    from b4_local_agent_llm import generate_ai_message as b4_generate_ai_message
    return b4_generate_ai_message(*args, **kwargs)
```

**B4 内部执行路径（`code/b4_local_agent_llm.py`）：**

```
generate_ai_message(model_config_path, messages, tools_schema, mode, ...)
  │
  ├─ [mock 模式] _mock_generate(messages, tools_schema)
  │     └─ 规则匹配：在用户输入中找工具名 → 生成假的 tool_calls
  │        若未匹配到工具 → 返回假的文本回复
  │
  └─ [prompt_json 模式] _prompt_json_generate(...)
       ├─ _load_model_config(model_config_path)     加载 model.yaml
       ├─ _model_cache_key(...)                      计算缓存键
       ├─ _load_model_bundle(config, device)         加载 tokenizer + model
       │     └─ AutoTokenizer.from_pretrained() / AutoModelForCausalLM.from_pretrained()
       │        带 torch_dtype / device_map 配置
       ├─ _build_prompt_messages(messages, tools_schema)
       │     └─ tokenizer.apply_chat_template()      应用 Qwen chat template
       ├─ model.generate(**gen_kwargs)               实际推理
       │     max_new_tokens=4096, temperature=0.0, do_sample=False
       └─ _parse_model_output(raw_text)
            ├─ _extract_tool_result(raw)                尝试直接 JSON.parse
            ├─ _parse_tool_calls_fragment(raw)          正则匹配 tool_calls 片段
            ├─ _parse_json_with_backtick_tail(raw)      去除尾部 ``` 后再解析
            └─ _candidate_to_message(candidate)         构造标准 AIMessage dict
```

**返回值格式：**
```python
{
    "ai_message": {
        "role": "assistant",
        "content": "...",         # 最终回复时有内容，工具调用时为空
        "tool_calls": [...]       # 工具调用列表
    },
    "status": "success" | "parse_error",
    "error": None | {...}
}
```

---

### 1b. 判断 LLM 输出并分支

```python
messages.append(ai_message)              # 将 AI 回复加入对话历史

turn = {
    "turn_index": llm_calls,
    "ai_message": ai_message,
    "llm_status": llm_status,
    "llm_error": llm_error,
    "tool_messages": [],
    "latency_ms": None,
}
```

#### 分支 1：LLM 解析失败 → 终止

```python
if llm_status != "success":
    status = "llm_parse_error"
    terminal_error = {
        "type": "LLMParseError",
        "message": "B4 failed to parse the model output as a valid AIMessage JSON object.",
        "llm_call_index": llm_calls,
        "cause": llm_error,
    }
    turns.append(turn)
    break                                     # ← 退出循环
```

#### 分支 2：无工具调用 → 最终回复

```python
tool_calls = ai_message.get("tool_calls", [])

if not tool_calls:
    final_answer = ai_message["content"]      # 这就是最终回复
    turns.append(turn)
    break                                     # ← 退出循环
```

#### 分支 3：超过最大轮次 → 终止

```python
if tool_rounds >= runtime["max_turns"]:
    final_answer = (
        "任务因超过最大工具调用轮次而终止，"
        f"最后一次模型仍请求调用工具：{requested}。"
    )
    status = "max_turns_exceeded"
    turns.append(turn)
    break                                     # ← 退出循环
```

---

### 1c. 执行工具调用 ← B2 在此介入

```python
# Integrated 路径：
tool_messages = execute_tool_calls(
    tool_calls,
    str(tools_file),
    runtime["toolset"],
    str(output_dir),
)
```

这进入 **B3 → B2 → Skill** 的三层调用链。

#### B3 层：`execute_tool_calls()`

**文件：** `code/b3_tool_layer.py`

```
execute_tool_calls(tool_calls, tools_config_path, toolset, output_dir)
  │
  ├─ _load_tools_config(tools_config_path)
  │     └─ read_yaml("configs/tools.yaml")    读取工具声明配置
  │
  ├─ _resolve_toolset(all_tools, toolset)
  │     └─ 按 toolset 字段过滤，只保留启用的工具
  │
  └─ for each tool_call in tool_calls:
       │
       ├─ tool_def = tools_by_name.get(tool_call["name"])
       │
       ├─ _validate_args(tool_call["args"], tool_def["parameters"])
       │     ├─ 检查必填参数是否存在
       │     ├─ 检查参数类型是否匹配（string / number / integer / boolean / object / array）
       │     └─ 校验失败 → _error_result() → SkillResult(error)
       │
       └─ run_skill(skill_name, args, data_root, output_dir)
            │                                            ← B3 → B2 桥接点
            └─ [进入 B2] ───────────────────────────────┐
```

#### B2 层：`run_skill()` — Skill 反射调度器

**文件：** `code/b2_run_skill.py:28`

**签名：**
```python
def run_skill(
    skill_name: str,
    input_data: dict,
    data_root: str | None = None,
    output_dir: str | None = None,
) -> dict:
```

**完整执行流程：**

```python
# 步骤 1：校验 skill 名称
if skill_name not in SKILL_MODULES:
    raise ValueError(f"unknown skill: {skill_name}")
# SKILL_MODULES = {
#     "calculator":        "skills.calculator",
#     "file_reader":       "skills.file_reader",
#     "local_file_search": "skills.local_file_search",
#     "table_analyzer":    "skills.table_analyzer",
#     "format_converter":  "skills.format_converter",
# }
```

```python
# 步骤 2：校验输入
if not isinstance(input_data, dict):
    raise ValueError("skill input must be a JSON object")
```

```python
# 步骤 3：动态导入模块
module = importlib.import_module(SKILL_MODULES[skill_name])
# 例：skill_name="calculator" → import skills.calculator
```

```python
# 步骤 4：获取函数对象（约定：函数名 = skill 名）
function = getattr(module, skill_name)
# 例：getattr(skills.calculator, "calculator") → calculator 函数
```

```python
# 步骤 5：准备参数（按需注入 data_root 和 output_dir）
kwargs = dict(input_data)
signature = inspect.signature(function)

if "data_root" in signature.parameters:
    kwargs["data_root"] = data_root or str(DEFAULT_DATA_ROOT)

if "output_dir" in signature.parameters:
    kwargs["output_dir"] = output_dir
```

```python
# 步骤 6：执行 Skill 函数
start = perf_counter()
try:
    output = function(**kwargs)
    status = "success"
    error = None
except Exception as exc:
    output = None
    status = "error"
    error = {"type": type(exc).__name__, "message": str(exc)}
latency_ms = round((perf_counter() - start) * 1000, 3)
```

```python
# 步骤 7：构造标准化 SkillResult
return make_skill_result(skill_name, status, input_data, output, error, latency_ms)
```

**`make_skill_result()`**（`common/schemas.py`）返回值：
```python
{
    "skill_name": "calculator",
    "status": "success",        # "success" | "error"
    "input": {"expression": "23 * 17 + 9"},
    "output": {"result": 400},  # 成功时有值
    "error": null,              # 失败时含 type 和 message
    "latency_ms": 0.5
}
```

#### Skill 实现层：5 个内置 Skill

**文件：** `skills/*.py`

| Skill | 入口函数 | 核心逻辑 | 特殊依赖 |
|-------|---------|---------|---------|
| `calculator` | `calculator(expression)` | `_evaluate()` → `ast.parse()` + `ast.literal_eval()` 安全求值 | 无 |
| `file_reader` | `file_reader(path, max_chars)` | `resolve_data_path()` + `open().read()` 截断读取 | `data_root` |
| `local_file_search` | `local_file_search(path, keyword, context_chars)` | `resolve_data_path()` + 分词匹配 + `_snippet()` 截图 + 计分排序 | `data_root` |
| `table_analyzer` | `table_analyzer(path)` | `resolve_data_path()` + `csv.DictReader()` 解析 CSV/TSV + 描述性统计 | `data_root` |
| `format_converter` | `format_converter(content, output_format, output_dir)` | `_parse_key_value_lines()` → `_safe_output_path()` → `_write_output_file()` 写 Markdown/JSON | `output_dir` |

**执行示例 — `calculator._evaluate()`：**

```python
def _evaluate(expression: str) -> float:
    tree = ast.parse(expression, mode="eval")
    return ast.literal_eval(tree.body)   # 安全的算术求值，不执行任意代码
```

#### B3 层：包装 SkillResult → ToolMessage

回到 B3，将 SkillResult 包装为标准的 ToolMessage：

```python
{
    "role": "tool",
    "tool_call_id": "call_001",          # 关联 AIMessage 中的 tool_call
    "name": "calculator",                # 技能名称
    "content": '{"skill_name":"calculator","status":"success",...}',  # SkillResult JSON 字符串
    "status": "success"
}
```

---

### 1d. 回环 — 更新状态并继续

```python
# 累加状态
tool_rounds += 1
messages.extend(tool_messages)         # 工具结果加入对话历史
all_tool_messages.extend(tool_messages)

# 记录本轮
turn["tool_messages"] = tool_messages
turn["latency_ms"] = round((perf_counter() - turn_start) * 1000, 3)
turns.append(turn)

# → 回到 while True 顶部
#   下一轮 LLM 将看到 system + user + ai(tool_calls) + tool(results)
#   然后决定：继续调工具 / 输出最终回复
```

---

## 阶段 2：收尾 — 输出与持久化

循环结束后（任一 break 触发）：

### 写入产物

```python
# 完整对话历史（所有轮次的 messages）
write_json(messages, output_dir / "messages.json")

# 仅工具消息（integrated 模式）
if execution_mode == "integrated":
    write_json(all_tool_messages, output_dir / "tool_messages.json")

# 最终回复
write_text(final_answer.strip() + "\n", output_dir / "final_answer.md")
```

### 构造执行追踪

```python
trace = {
    "conversation_id": runtime["conversation_id"],
    "execution_mode": execution_mode,
    "status": status,                    # success | llm_parse_error | max_turns_exceeded | partial
    "toolset": runtime["toolset"],
    "max_turns": runtime["max_turns"],
    "tool_rounds_used": tool_rounds,
    "llm_call_count": llm_calls,
    "turns": turns,                     # 每轮的完整信息
    "final_answer_path": "final_answer.md",
    "memory_save": memory_save,
    "warnings": warnings,
    "error": terminal_error,
}
write_json(trace, output_dir / "trace.json")
```

### 记忆持久化

```python
if execution_mode == "integrated"
   and runtime["save_memory"] != "none"
   and trace["status"] == "success":

    saved_memory = save_memory(
        str(memory_file),
        runtime["conversation_id"],
        runtime["save_memory"],     # "conversation" | "global"
        str(output_dir / "messages.json"),
        str(output_dir / "trace.json"),
        str(output_dir / "final_answer.md"),
        str(output_dir),
    )
```

### 写入运行日志

```python
append_jsonl(
    {
        "timestamp": now_iso(),
        "conversation_id": runtime["conversation_id"],
        "execution_mode": execution_mode,
        "status": trace["status"],
        "llm_mode": mode,
        "tool_rounds_used": tool_rounds,
        "llm_call_count": llm_calls,
        "elapsed_ms": result["elapsed_ms"],
    },
    output_dir / "runtime_log.jsonl",
)
```

### 返回值

```python
return {
    "conversation_id": runtime["conversation_id"],
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
```

---

## B2 独立 CLI 路径

B2 也可以**独立运行**（不经过 B1/B3），用于单独测试某个 Skill：

**文件：** `code/b2_run_skill.py:63`

```
B2:main()
  ├─ build_parser()                        解析 --skill / --input / --outdir / --data_root
  ├─ resolve_cli_path(args.input)          解析输入文件路径
  ├─ read_json(input_path)                 读取工具调用参数的 JSON 文件
  ├─ run_skill(skill_name, input_data, data_root, output_dir)
  │     └─ [反射调度，同上]
  ├─ write_json(result, f"{skill_name}_result.json")  写入结果
  ├─ append_jsonl(log_entry, "skill_run_log.jsonl")   追加日志
  └─ print(result_path)
```

**调用示例：**
```bash
cd agent/code
python b2_run_skill.py \
  --skill calculator \
  --input ../data/tool_inputs/tool_input_calculator.json \
  --outdir ../outputs/test_calc
```

---

## 数据流总览

```
                            runtime_input.json
                                   │
                                   ▼
              ┌──────────────────────────────────────┐
              │           B1: run_agent()             │
              │                                      │
              │  ┌─ _validate_runtime_input()        │
              │  ├─ B5.load_memory()                 │
              │  ├─ B3.get_tools_schema()            │
              │  ├─ _memory_context()                │
              │  ├─ _default_llm_mode()              │
              │  │                                   │
              │  └─ while True: ─────────────────┐   │
              │       │                           │   │
              │       ▼                           │   │
              │  ┌─ B4.generate_ai_message()      │   │
              │  │    → AIMessage                 │   │
              │  ├─ tool_calls? ────No──→ break   │   │
              │  │    │                           │   │
              │  │   Yes                          │   │
              │  │    ▼                           │   │
              │  ├─ B3.execute_tool_calls()       │   │
              │  │    │                            │   │
              │  │    ├─ _validate_args()          │   │
              │  │    └─ B2.run_skill() ──────┐   │   │
              │  │         │                   │   │   │
              │  │         ├─ import_module()  │   │   │
              │  │         ├─ getattr()        │   │   │
              │  │         ├─ function()       │   │   │
              │  │         └─ make_skill_res() │   │   │
              │  │              → SkillResult  │   │   │
              │  │                             │   │   │
              │  ├─ ← ToolMessage ─────────────┘   │   │
              │  ├─ tool_rounds += 1               │   │
              │  └─ ──→ loop back ─────────────────┘   │
              │                                      │
              │  ┌─ write_json(messages.json)        │
              │  ├─ write_text(final_answer.md)      │
              │  ├─ write_json(trace.json)           │
              │  ├─ B5.save_memory()                 │
              │  └─ append_jsonl(runtime_log.jsonl)  │
              └──────────────────────────────────────┘
                                   │
                                   ▼
                          outputs/<outdir>/
                          ├── messages.json
                          ├── tool_messages.json
                          ├── final_answer.md
                          ├── trace.json
                          └── runtime_log.jsonl
```

---

## B2 设计亮点

`run_skill()`（仅 23 行）是整个 Skill 系统的**反射式调度核心**：

| 设计 | 实现 | 效果 |
|------|------|------|
| 动态导入 | `importlib.import_module(SKILL_MODULES[name])` | 按字符串名加载模块，新增 Skill 无需改 B2 代码 |
| 函数约定 | `getattr(module, skill_name)` | 模块内函数名 = skill 名，零配置 |
| 签名探测 | `inspect.signature(function)` | 按需注入 `data_root` / `output_dir`，Skill 可按需声明 |
| 统一结果 | `make_skill_result(...)` | 无论成功或异常，返回相同结构，消费者无需分支处理 |
| 独立可测 | B2 有独立 CLI | 可脱离 B1/B3 单独测试任意 Skill |

---

## 调用关系矩阵

```
调用者                          被调用者                        边类型
─────────────────────────────────────────────────────────────────────
B1:main()                 →    B1:run_agent()                  calls
B1:run_agent()            →    B1:_validate_runtime_input()    calls
B1:run_agent()            →    B1:_memory_context()            calls
B1:run_agent()            →    B1:_default_llm_mode()          calls
B1:run_agent()            →    B5:load_memory()                calls
B1:run_agent()            →    B5:save_memory()                calls
B1:run_agent()            →    B3:get_tools_schema()           calls
B1:run_agent()            →    B3:execute_tool_calls()         calls
B1:run_agent()            →    B4:generate_ai_message()        calls
B3:execute_tool_calls()   →    B2:run_skill()                  calls
B3:execute_tool_calls()   →    B3:_validate_args()             calls
B2:run_skill()            →    skills.*:<skill_function>()     calls
B1:run_agent()            →    common/io_utils:*               calls
B1:run_agent()            →    common/schemas:*                calls
```

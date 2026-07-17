# 个人模块 README — B1 Agent 运行时总控 + B2 Skill 反射调度

---

## 1. 模块概述

### 1.1 模块名称

`B1 (Agent Runtime)` + `B2 (Skill Runner)` — Agent 运行时总控与 Skill 反射调度器

### 1.2 模块说明

B1 是整个 Agent 框架的**中央编排器**，负责协调 B3/B4/B5 各模块的执行时序、管理多轮对话状态、提供交互式命令行界面。B2 是工具执行链的**适配枢纽**，通过反射机制将工具调用请求动态映射到 Skill 函数，统一处理错误分类、参数注入和结果标准化。

```text
B1 核心能力：
- 三种启动模式：file（JSON 配置）/ standalone（CLI 参数）/ resume（checkpoint 恢复）
- 交互式 REPL：多轮对话，上下文跨轮次累积
- 内置命令系统：/compress（历史压缩）/ /system（提示词切换） / /exit
- 批量任务：从 JSONL 文件读取多任务，支持断点续跑
- 自动 checkpoint：每轮结束后保存状态，Ctrl+C 不丢数据

B2 核心能力：
- 反射调度：importlib 动态导入 + inspect 签名探测，新增 Skill 无需改任何调度代码
- 标准化错误码：15 个错误码，异常类型 + 消息内容双维度匹配
- 复合 Skill：read_and_analyze（读取+统计）、search_and_read（搜索+读取）
- 4 个新增 Skill：list_ops、text_stats 及两个复合 Skill
```

### 1.3 完成情况概览

| 类型 | 完成情况 |
|---|---|
| 基础要求 | 已完成：B1 单次执行 tool_calls 循环，B2 反射调度 5 个基础 Skill，B3/B4/B5 集成联调 |
| 进阶要求 | 已完成：多轮交互对话、历史压缩、运行时提示词切换、断点续跑、批量任务、复合 Skill 流水线、标准化错误码 |
| 可独立运行的演示 | B1 交互模式 `python b1_agent_runtime.py --interactive --tools_config ...`；B2 独立 CLI `python b2_run_skill.py --skill list_ops --input ...` |
| 与团队系统集成情况 | B1 调用 B3/B4/B5；B3 调用 B2；B2 调用 Skill 实现。全链路通过 CLI 参数和 `configs/tools.yaml` 配置集成 |

---

## 2. 环境、模型与数据依赖

### 2.1 运行环境

| 项目 | 要求 |
|---|---|
| Python 版本 | `3.10` |
| 必要依赖 | PyYAML, transformers, torch, accelerate（完整列表见 `requirements.txt`） |
| 是否需要模型 | 是，Qwen3.5-4B（prompt_json 模式）；mock 模式不需要 |
| 是否需要 GPU | prompt_json 模式需要 CUDA GPU；mock 模式不需要 |
| 是否需要外部数据集 | 否，使用项目自带 `data/` 目录下的测试固件 |

### 2.2 模型依赖

| 模型 | 来源 | 项目内相对路径 | 用途 |
|---|---|---|---|
| Qwen3.5-4B | 公共服务器路径或 HuggingFace | `configs/model.yaml` 中 `model_name_or_path` 指定 | LLM 推理（prompt_json 模式） |

```bash
# 在 configs/model.yaml 中修改 model_name_or_path 指向模型目录
```

### 2.3 数据集或样例数据依赖

| 数据或文件 | 来源 | 项目内相对路径 | 用途 |
|---|---|---|---|
| runtime_input.json | 项目自带 | `data/runtime_input.json` | B1 单次执行输入配置 |
| b1_fixtures | 项目自带 | `data/b1_fixtures/` | B1 fixture 模式测试固件 |
| tool_inputs | 项目自带 | `data/tool_inputs/` | B2 各 Skill 独立测试输入 |
| docs | 项目自带 | `data/docs/` | file_reader / local_file_search 测试文档 |

### 2.4 安装步骤

```bash
# 创建环境
conda create -n agent python=3.10 -y
conda activate agent

# 安装依赖
pip install -r requirements.txt

# 验证 B2 可独立运行
cd code
python b2_run_skill.py --skill calculator --input ../data/tool_inputs/tool_input_calculator.json --outdir ../outputs/test
```

---

## 3. 文件结构与接口边界

### 3.1 文件结构

```text
agent/
├── code/
│   ├── b1_agent_runtime.py       # B1 核心：run_agent()、交互循环、批量任务、checkpoint
│   ├── b2_run_skill.py            # B2 核心：run_skill()、SKILL_MODULES 注册表
│   ├── common/
│   │   ├── error_codes.py         # 错误码枚举 + classify_error()（新增）
│   │   ├── schemas.py             # 数据契约（修改：content 自动类型转换）
│   │   ├── io_utils.py            # 文件读写工具
│   │   ├── logging_utils.py       # 日志工具
│   │   └── path_utils.py          # 路径解析工具
│   └── test_errors.py             # 错误码测试脚本（新增）
├── skills/
│   ├── __init__.py                # resolve_data_path() 路径安全校验
│   ├── calculator.py              # 安全算术求值
│   ├── file_reader.py             # 文件读取
│   ├── local_file_search.py       # 本地文件搜索
│   ├── table_analyzer.py          # CSV/TSV 表格分析
│   ├── format_converter.py        # 文本格式转换
│   ├── list_ops.py                # 列表操作（新增）
│   ├── text_stats.py              # 文本统计（新增）
│   ├── read_and_analyze.py        # 复合 Skill：读取+统计（新增）
│   └── search_and_read.py         # 复合 Skill：搜索+读取（新增）
├── configs/
│   └── tools.yaml                 # 工具注册表（修改：新增 4 个 Skill 定义）
├── prompts/
│   └── local_tool_agent.txt       # 系统提示词（修改：新增工具选择规则）
├── data/                          # 测试固件与样例数据
├── outputs/                       # 输出目录
└── README.md
```

### 3.2 接口边界

| 类型 | 来源 / 去向 | 数据格式 | 说明 |
|---|---|---|---|
| 输入 | 用户（CLI 参数） | `--input` / `--interactive` / `--batch` 等 | B1 启动方式选择 |
| 输入 | B5 | `{"selected_memory_docs": [...]}` | 记忆文档 |
| 输入 | B3 | `[{"type":"function", "function": {...}}]` | OpenAI function-calling 工具 schema |
| 输入 | B4 | `{"ai_message": dict, "status": str}` | LLM 生成的 AIMessage |
| 输入 | B3 → B2 | `("calculator", {"expression": "1+2"})` | 工具名 + 参数字典 |
| 输出 | B1 → 文件 | `messages.json`, `trace.json`, `final_answer.md`, `checkpoint.json` | 对话产物 |
| 输出 | B2 → B3 | `{"skill_name": str, "status": str, "output": dict, "error": dict}` | 标准化 SkillResult |

---

## 4. 基础要求实现与演示

### 4.1 基础功能说明

基础版本实现了一个完整的"单次用户输入 → tool_calls 循环 → 最终回复" Agent 执行流程：

- B1 读取 `runtime_input.json` 配置，加载系统提示词和记忆，构建初始 messages
- 进入 while 循环：调用 LLM → 判断 tool_calls → 调用 B3 执行工具 → 结果回注 → 继续循环
- 循环结束条件：LLM 输出纯文本（最终回复）或超过最大工具调用轮次
- 写入 `messages.json`、`trace.json`、`final_answer.md` 产物

B2 通过反射机制动态调度 5 个基础 Skill：calculator、file_reader、local_file_search、table_analyzer、format_converter。

### 4.2 基础功能实现路径

| 文件 / 函数 / 脚本 | 作用 |
|---|---|
| `b1_agent_runtime.py:run_agent()` | 核心编排引擎，单次执行入口 |
| `b1_agent_runtime.py:_validate_runtime_input()` | 校验 runtime_input.json 配置 |
| `b1_agent_runtime.py:_memory_context()` | 将记忆文档拼为 XML 块注入 system prompt |
| `b2_run_skill.py:run_skill()` | 反射调度：动态导入→签名探测→参数注入→执行→结果标准化 |
| `skills/__init__.py:resolve_data_path()` | 路径安全校验，防止目录穿越攻击 |
| 5 个 `skills/*.py` | 基础 Skill 实现 |

```text
runtime_input.json → _validate_runtime_input → load_memory → get_tools_schema
→ [system + user] messages → while (tool_calls 循环):
    generate_ai_message(B4) → AIMessage
    → if tool_calls: execute_tool_calls(B3) → run_skill(B2) → Skill函数
    → ToolMessage 回注 → 继续循环
    → if no tool_calls: final_answer → break
→ write messages.json / trace.json / final_answer.md
```

### 4.3 基础功能输入格式与样例

| 字段 / 输入文件 | 类型 / 格式 | 是否必需 | 说明 |
|---|---|---|---|
| `conversation_id` | string | 是 | 对话唯一标识 |
| `user_input` | string | 是 | 用户输入文本 |
| `system_prompt_path` | string | 是 | 系统提示词文件路径 |
| `toolset` | string | 是 | 启用的工具集名称 |
| `max_turns` | integer | 是 | 最大工具调用轮次 |
| `save_memory` | string | 是 | none / conversation / global |
| `execution_mode` | string | 否 | integrated（默认）或 fixture |

样例输入：

| 样例文件 | 用途 |
|---|---|
| `data/runtime_input.json` | 基础对话（读文件并总结要点） |
| `data/runtime_input_0.json` | 无工具场景（纯文本回复） |
| `data/runtime_input_2.json` | 计算器 Skill 调用 |
| `data/runtime_input_4.json` | CSV 数据分析 |
| `data/tool_inputs/tool_input_calculator.json` | B2 独立测试：calculator |
| `data/tool_inputs/tool_input_file_reader.json` | B2 独立测试：file_reader |

### 4.4 基础功能演示命令

```bash
# === B1 单次执行（mock 模式） ===
cd code
python b1_agent_runtime.py \
  --input ../data/runtime_input.json \
  --tools_config ../configs/tools.yaml \
  --memory_config ../configs/memory_small_limit.yaml \
  --model_config ../configs/model.yaml \
  --outdir ../outputs/basic_demo \
  --llm_mode mock

# === B1 单次执行（真实模型） ===
python b1_agent_runtime.py \
  --input ../data/runtime_input.json \
  --tools_config ../configs/tools.yaml \
  --memory_config ../configs/memory_small_limit.yaml \
  --model_config ../configs/model.yaml \
  --outdir ../outputs/basic_demo \
  --llm_mode prompt_json

# === B2 独立运行 ===
python b2_run_skill.py \
  --skill calculator \
  --input ../data/tool_inputs/tool_input_calculator.json \
  --outdir ../outputs/b2_demo
```

观察：
- `messages.json` 包含完整对话历史（system → user → assistant → tool → assistant）
- `final_answer.md` 包含最终中文回复（三点摘要）
- B2 独立运行输出 `calculator_result.json` 包含 `status: "success"` 和计算结果

### 4.5 基础功能输出格式

| 输出文件 / 返回字段 | 格式 | 说明 |
|---|---|---|
| `messages.json` | JSON 数组 | 完整对话历史，每个元素含 role/content/tool_calls |
| `trace.json` | JSON 对象 | 执行追踪：status/turns/tool_rounds_used/llm_call_count |
| `final_answer.md` | Markdown 文本 | 最终回复 |
| `{skill_name}_result.json` | JSON 对象 | SkillResult: skill_name/status/output/error/latency_ms |

### 4.6 基础功能结果截图

```text
[在此处插入基础功能运行截图]
[在此处插入关键输出文件截图]
```

![基础功能演示占位](docs/images/basic_feature_placeholder.png)

---

## 5. 进阶要求实现与演示

### 5.1 选择的进阶要求

| 进阶要求 | 是否完成 | 对应文件 / 函数 | 简要说明 |
|---|---|---|---|
| 多轮交互对话 | 是 | `b1_agent_runtime.py:run_agent()` 外层循环 | REPL 模式，messages 跨轮次累积 |
| 对话历史压缩 | 是 | `b1_agent_runtime.py:_summarize_history()` | `/compress` 命令，LLM 摘要旧消息 |
| 运行时提示词切换 | 是 | `b1_agent_runtime.py` `/system` 分支 | /system FILE/+TEXT/- 三种操作 |
| 断点续跑 | 是 | `b1_agent_runtime.py` checkpoint 机制 | 每轮自动保存，`--resume` 恢复 |
| 批量任务执行 | 是 | `b1_agent_runtime.py:_run_batch_tasks()` | JSONL 输入 + `batch_progress.json` 进度 |
| 标准化错误码 | 是 | `common/error_codes.py` | 15 个错误码，异常类型+消息双匹配 |
| 复合 Skill 流水线 | 是 | `skills/read_and_analyze.py`, `skills/search_and_read.py` | 内部串联 2 个 Skill，LLM 只需 1 次调用 |
| 新增实用 Skill | 是 | `skills/list_ops.py`, `skills/text_stats.py` | 列表操作（12 种 op）、文本统计（CJK 感知） |

### 5.2 进阶功能 1：多轮交互对话与命令系统

#### 功能说明

在原有单次执行的 tool_calls 循环外包装交互 REPL 循环，用户可在命令行持续输入消息，每轮独立执行 tool_calls 循环，对话历史跨轮次累积。内置命令系统支持运行时对话管理。

```text
解决了基础版本的三大不足：
1. 每次对话需要重写 runtime_input.json，无法连续提问
2. 长对话上下文膨胀，无法压缩
3. 系统提示词写死，无法根据对话需求动态调整
```

#### 实现路径

| 文件 / 函数 / 脚本 | 作用 |
|---|---|
| `b1_agent_runtime.py:run_agent()` 外层 while 循环 | 交互轮次循环，获取输入→tool_calls→输出→checkpoint |
| `b1_agent_runtime.py:_summarize_history()` | 切分轮次→调 LLM 生成摘要→替换旧消息 |
| 命令分支：`/compress` `/system` `/exit` | 内联处理，不进入 tool_calls 循环 |

```text
用户输入 → /compress? → _summarize_history → 更新 messages + checkpoint → continue
         → /system?  → 切换/追加/重置 system prompt → continue
         → /exit?    → break
         → 普通消息  → 追加 user → tool_calls 循环 → 累积 → checkpoint → 下一轮
```

#### 输入格式与样例

| 字段 / 输入文件 / 配置项 | 类型 / 格式 | 是否必需 | 说明 |
|---|---|---|---|
| `--interactive` / `-i` | CLI flag | 是 | 启用交互模式 |
| `--system_prompt` | 文件路径 | standalone 模式必填 | 系统提示词文件 |
| `/compress [N]` | 交互命令 | 否 | 压缩历史，保留最近 N 轮 |
| `/system +TEXT` | 交互命令 | 否 | 追加指令到 system prompt |
| `/exit` / `/quit` | 交互命令 | 否 | 退出对话 |

#### 演示命令

```bash
# 方式 A：从 runtime_input.json 启动（首轮用 JSON 的 user_input）
cd code
python b1_agent_runtime.py \
  --input ../data/runtime_input.json \
  --tools_config ../configs/tools.yaml \
  --memory_config ../configs/memory_small_limit.yaml \
  --model_config ../configs/model.yaml \
  --outdir ../outputs/chat \
  --llm_mode prompt_json \
  --interactive

# 方式 B：无 JSON 启动（首轮从 stdin 输入）
python b1_agent_runtime.py \
  --tools_config ../configs/tools.yaml \
  --memory_config ../configs/memory_small_limit.yaml \
  --model_config ../configs/model.yaml \
  --system_prompt ../prompts/local_tool_agent.txt \
  --outdir ../outputs/chat \
  --llm_mode prompt_json \
  --interactive

# 方式 C：从 checkpoint 恢复
python b1_agent_runtime.py \
  --tools_config ../configs/tools.yaml \
  --memory_config ../configs/memory_small_limit.yaml \
  --model_config ../configs/model.yaml \
  --resume ../outputs/chat/conv_001_2026-07-12T09-30-00/checkpoint.json \
  --interactive
```

交互过程：
```
>>> 帮我读取 docs/agent_intro.txt
[LLM 调 file_reader → 返回内容 → 输出三点摘要]

>>> /compress 2
[压缩完成] 8 条 -> 4 条，保留最近 2 轮

>>> /system +用中文回答，不超过50字
[已追加] +12 字符

>>> 总结一下
[LLM 看到压缩后的历史 + 新 system prompt → 精简回复]

>>> /exit
```

#### 输出格式

| 输出文件 / 返回字段 | 格式 | 说明 |
|---|---|---|
| `checkpoint.json` | JSON 对象 | 每轮更新：version/messages/runtime/original_system_prompt |
| `messages.json` | JSON 数组 | 完整对话历史（压缩后同步更新） |
| `{conv_id}_{timestamp}/` | 目录 | 交互模式下自动创建的时间戳子目录 |

#### 示例图片

```text
[在此处插入交互模式运行截图]
[在此处插入 /compress、/system 命令执行截图]
```

![进阶功能演示占位](docs/images/advanced_feature_placeholder.png)

### 5.3 进阶功能 2：批量任务执行与断点续跑

#### 功能说明

从 JSONL 文件读取多个任务，顺序执行并通过 B1 的非交互路径处理每个任务。每个任务独立输出到子目录。执行过程中实时写 `batch_progress.json`，中断后重新运行自动跳过已完成任务。

#### 实现路径

| 文件 / 函数 / 脚本 | 作用 |
|---|---|
| `b1_agent_runtime.py:_run_batch_tasks()` | 批量任务调度器 |
| `batch_progress.json` | 进度追踪文件：{"completed":[0,1,3],"total":5} |

```text
tasks.jsonl → _run_batch_tasks() → for each task:
  写临时 runtime.json → run_agent(非交互) → write_json(task_outdir/*)
  → 更新 batch_progress.json
→ batch_summary.json
```

#### 演示命令

```bash
# 准备任务文件
cat > tasks.jsonl << EOF
{"user_input": "计算 3*5+2"}
{"user_input": "总结 docs/agent_intro.txt"}
{"user_input": "用 list_ops 排序 [5, 2, 8, 1]", "max_turns": 2}
EOF

# 执行批量任务
cd code
python b1_agent_runtime.py \
  --batch tasks.jsonl \
  --system_prompt ../prompts/local_tool_agent.txt \
  --tools_config ../configs/tools.yaml \
  --memory_config ../configs/memory_small_limit.yaml \
  --model_config ../configs/model.yaml \
  --outdir ../outputs/batch_run \
  --llm_mode prompt_json

# 中断后重新运行同一命令即可续跑
```

### 5.4 进阶功能 3：标准化错误码系统

#### 功能说明

为所有 Skill 异常定义 15 个机器可读的错误码（`E_FILE_NOT_FOUND`、`E_PATH_ESCAPE`、`E_EXEC_TIMEOUT` 等），通过异常类型 + 消息内容双维度匹配自动分类。LLM 可根据 `code` 字段精确判断错误类型并调整策略。

#### 实现路径

| 文件 / 函数 / 脚本 | 作用 |
|---|---|
| `common/error_codes.py:SkillErrorCode` | 错误码枚举（15 个） |
| `common/error_codes.py:classify_error()` | 异常→错误码分类函数 |
| `b2_run_skill.py:run_skill()` | 调用 classify_error() 替代裸异常 |
| `b3_tool_layer.py:_error_result()` | 同上 |

#### 演示命令

```bash
cd code && python test_errors.py
```

输出：
```
PASS | file_reader          | code=E_FILE_NOT_FOUND
PASS | file_reader          | code=E_PATH_ESCAPE
PASS | calculator           | code=E_INVALID_INPUT
PASS | calculator           | code=E_MISSING_PARAM
PASS | format_converter     | code=E_INVALID_INPUT
PASS | nonexistent          | code=E_INVALID_INPUT

6/6 passed
```

### 5.5 进阶功能 4：复合 Skill

#### 功能说明

`read_and_analyze` 内部串联 file_reader → text_stats，LLM 只需传一个 `path` 参数。`search_and_read` 内部串联 local_file_search → file_reader，LLM 只需传一个 `query` 参数。每个复合 Skill 约 20 行代码，完全复用已有 Skill 逻辑，不重复实现。

#### 演示命令

```bash
# 独立测试复合 Skill
cd code
python b2_run_skill.py \
  --skill read_and_analyze \
  --input <(echo '{"path": "docs/agent_intro.txt"}') \
  --outdir ../outputs/composite_test
```

---

## 6. 与团队系统的集成说明

B1 作为系统入口，通过 `main()` 接受 CLI 参数后进入 `run_agent()`：

- **B1 调用 B4**：通过 `generate_ai_message()` 懒加载代理函数调用 `b4_local_agent_llm.generate_ai_message()`，传入 model_config 路径、messages 列表、tools_schema 列表、mode 和 artifact 路径
- **B1 调用 B3**：通过 `from b3_tool_layer import execute_tool_calls, get_tools_schema` 懒加载，传入 tools_config 路径和 toolset 名称
- **B1 调用 B5**：通过 `from b5_memory import load_memory, save_memory` 懒加载，传入 memory_config 路径和相关配置
- **B3 调用 B2**：B3 的 `execute_tool_calls()` 通过 `from b2_run_skill import run_skill` 调用，传入 skill_name 和 args 字典

配置文件 `configs/tools.yaml` 是核心集成点——工具定义同时在 B3（生成 schema）和 B2（动态导入）中读取，保持同步。

联调过程中遇到的主要问题及解决：
- **JSON 解析失败**：Qwen 输出含未转义换行符和尾部垃圾，通过在 B4 的 `_parse_model_output()` 中增加 `_repair_newlines_in_json()` 预修复和放宽 `_parse_json_with_backtick_tail()` 后缀检查解决
- **content 非字符串**：Qwen 偶尔输出 `content: [3, 5]` 而非字符串，在 `validate_ai_message()` 中增加自动 json.dumps 转换
- **参数类型不匹配**：LLM 将 `max_chars` 输出为字符串 `"2000"` 而非整数，在 B3 `_validate_args()` 中增加 integer/number 字符串自动转型

---

## 7. 已知问题与后续改进

| 问题 | 当前原因 | 后续改进 |
|---|---|---|
| Qwen3.5-4B 小模型 function-calling 不稳定 | 4B 参数能力上限，复杂 tool schema 易输出格式错误 | 精简 tools_schema 注入（只传工具名+一行描述+参数名列表） |
| 大文件读取受限 | file_reader 有 max_chars 上限，LLM 需多次调用 | 增加 `read_file_full` Skill 自动分块读取并拼接 |
| 批量任务无并发 | `_run_batch_tasks()` 顺序执行 | 支持 `--batch-workers N` 多任务并发 |
| 无对话历史自动保存到 memory | `/compress` 压缩后的摘要未存为 memory 文档 | 压缩完成后自动调用 B5 save_memory() |
| mock 模式行为固定 | `_mock_generate()` 硬编码 file_reader 调用 | 根据用户输入关键词匹配不同工具的 mock 行为 |

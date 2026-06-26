"""Full demo: B1 runtime + B2 skills + config-driven agent.

Usage::

    python code/run_full_demo.py

Requires ``ANTHROPIC_API_KEY`` in your environment.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure the project root is on sys.path.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langchain_anthropic import ChatAnthropic

from code.b1_agent_runtime import Agent, AgentConfig
from code.b2_run_skill import load_skills
from code.common.config_loader import config_path, load_text, load_yaml


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Check prerequisites
    # ------------------------------------------------------------------
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("⚠️  ANTHROPIC_API_KEY is not set. Set it and re-run:")
        print("   export ANTHROPIC_API_KEY=sk-ant-...")
        return

    # ------------------------------------------------------------------
    # 2. Load configuration from YAML files
    # ------------------------------------------------------------------
    model_cfg = load_yaml(config_path("model.yaml"))["model"]
    tools_cfg = load_yaml(config_path("tools.yaml"))["tools"]
    memory_cfg = load_yaml(config_path("memory.yaml"))["memory"]

    print(f"📋 Model:  {model_cfg['provider']}/{model_cfg['model_name']}")
    print(f"🔧 Tools:  {tools_cfg['enabled']}")
    print(f"🧠 Memory: {memory_cfg['type']} (max {memory_cfg['max_messages']} msgs)")

    # ------------------------------------------------------------------
    # 3. Load system prompt from prompts/
    # ------------------------------------------------------------------
    prompt_path = ROOT / "prompts" / "local_tool_agent.txt"
    system_prompt = load_text(prompt_path)

    # ------------------------------------------------------------------
    # 4. Create the LLM
    # ------------------------------------------------------------------
    model = ChatAnthropic(
        model=model_cfg["model_name"],
        temperature=model_cfg.get("temperature", 0),
        max_tokens=model_cfg.get("max_tokens", 4096),
    )

    # ------------------------------------------------------------------
    # 5. Load skills via B2
    # ------------------------------------------------------------------
    tools = load_skills(tools_cfg["enabled"])
    print(f"✅ Loaded {len(tools)} tools: {[t.name for t in tools]}\n")

    # Apply tool settings (workspace root for file tools)
    workspace = tools_cfg.get("settings", {}).get("workspace_root", "./workspace")
    from skills.file_reader import set_workspace_root
    set_workspace_root(workspace)

    # ------------------------------------------------------------------
    # 6. Create the agent (B1)
    # ------------------------------------------------------------------
    config = AgentConfig(
        model=model,
        system_prompt=system_prompt,
        tools=tools,
        max_iterations=10,
        verbose=True,
    )
    agent = Agent(config)

    # ------------------------------------------------------------------
    # 7. Run a query
    # ------------------------------------------------------------------
    query = "What is the square root of 256 plus 100, all divided by 3?"

    print(f"👤 User: {query}\n")
    answer = agent.run(query)
    print(f"\n🤖 Agent: {answer}\n")

    # ------------------------------------------------------------------
    # 8. Show message history
    # ------------------------------------------------------------------
    print("=" * 60)
    print("📜 Full message history:")
    print("=" * 60)
    for i, msg in enumerate(agent.store.get_history()):
        role = type(msg).__name__
        content_preview = (
            str(msg.content)[:80].replace("\n", " ")
            if msg.content
            else "(tool_calls)"
        )
        print(f"  [{i}] {role}: {content_preview}")
    print("=" * 60)


if __name__ == "__main__":
    main()

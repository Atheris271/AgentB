"""B1: Agent运行与消息管理模块 — runtime loop and message management.

Drives one complete agent task from user input to final answer:

    SystemMessage → HumanMessage → [AIMessage(tool_calls) → ToolMessage]* → AIMessage(final)

Responsibilities:
    - AgentConfig:  holds model, system prompt, tools, and loop settings.
    - MessageStore: maintains the ordered message sequence.
    - Agent:         the main execution loop (LLM → tool calls → LLM → answer).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool


# =========================================================================
# AgentConfig
# =========================================================================


@dataclass
class AgentConfig:
    """Configuration for an Agent instance.

    Attributes:
        model: The LLM to use (e.g. ``ChatAnthropic``, ``ChatOpenAI``).
        system_prompt: System-level instruction injected at the start of
            every conversation.
        tools: Skill tools available to the agent.
        max_iterations: Hard limit on LLM→Tool→LLM loops per ``run()`` call.
            Defaults to 10.
        verbose: When True, prints each step of the agent loop to stdout.
    """

    model: BaseChatModel
    system_prompt: str
    tools: list[BaseTool] = field(default_factory=list)
    max_iterations: int = 10
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")


# =========================================================================
# MessageStore
# =========================================================================


class MessageStore:
    """Ordered container for LangChain message objects.

    Maintains the canonical message order and provides typed helpers
    for appending each message kind.

    Usage::

        store = MessageStore()
        store.set_system("You are a helpful assistant.")
        store.add_human("What is 2+2?")
        store.add_ai(ai_message)
        store.add_tool("4", tool_call_id="call_123")
    """

    def __init__(self) -> None:
        self._messages: list[BaseMessage] = []

    # -- System prompt ---------------------------------------------------

    def set_system(self, content: str) -> None:
        """Set (or replace) the system message."""
        self._messages = [m for m in self._messages if not isinstance(m, SystemMessage)]
        self._messages.insert(0, SystemMessage(content=content))

    @property
    def system_prompt(self) -> str | None:
        """The current system prompt text, if any."""
        for m in self._messages:
            if isinstance(m, SystemMessage):
                return str(m.content)
        return None

    # -- Append helpers --------------------------------------------------

    def append(self, message: BaseMessage) -> None:
        """Append any message to the history."""
        self._messages.append(message)

    def add_human(self, content: str) -> HumanMessage:
        """Append a HumanMessage and return it."""
        msg = HumanMessage(content=content)
        self._messages.append(msg)
        return msg

    def add_ai(self, message: AIMessage) -> AIMessage:
        """Append an AIMessage (possibly with tool_calls) and return it."""
        self._messages.append(message)
        return message

    def add_tool(self, content: str, *, tool_call_id: str, name: str = "") -> ToolMessage:
        """Append a ToolMessage and return it."""
        msg = ToolMessage(content=content, tool_call_id=tool_call_id, name=name)
        self._messages.append(msg)
        return msg

    # -- Query -----------------------------------------------------------

    def get_history(self) -> list[BaseMessage]:
        """Return the full message list (for passing to an LLM)."""
        return list(self._messages)

    @property
    def last_message(self) -> BaseMessage | None:
        """The most recent message, or None if empty."""
        return self._messages[-1] if self._messages else None

    @property
    def is_empty(self) -> bool:
        return len(self._messages) == 0

    def clear(self) -> None:
        """Remove all messages (system prompt included)."""
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)

    def __repr__(self) -> str:
        types = [type(m).__name__ for m in self._messages]
        return f"MessageStore({types})"


# =========================================================================
# Agent
# =========================================================================


class Agent:
    """A single conversational agent backed by an LLM and a set of tools.

    Usage::

        from langchain_anthropic import ChatAnthropic
        from code.b1_agent_runtime import Agent, AgentConfig
        from code.b2_run_skill import load_skills

        tools = load_skills(["calculator", "file_reader"])
        config = AgentConfig(
            model=ChatAnthropic(model="claude-sonnet-4-6"),
            system_prompt="You are a helpful assistant.",
            tools=tools,
        )
        agent = Agent(config)
        answer = agent.run("What is 15 * 23?")
        print(answer)
    """

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.store = MessageStore()
        self._tool_map: dict[str, BaseTool] = {t.name: t for t in config.tools}
        self._model: BaseChatModel = (
            config.model.bind_tools(config.tools) if config.tools else config.model
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, user_input: str) -> str:
        """Execute one agent task end-to-end.

        Args:
            user_input: The user's natural-language request.

        Returns:
            The agent's final text answer.
        """
        self.store.clear()
        self.store.set_system(self.config.system_prompt)
        self.store.add_human(user_input)

        for i in range(self.config.max_iterations):
            if self.config.verbose:
                print(f"\n--- Iteration {i + 1} ---")

            messages = self.store.get_history()
            response: AIMessage = self._model.invoke(messages)

            if self.config.verbose:
                self._log_response(response)

            self.store.add_ai(response)

            if not response.tool_calls:
                return self._extract_text(response)

            for tc in response.tool_calls:
                result = self._execute_tool(
                    name=tc["name"],
                    args=tc["args"],
                    tool_call_id=tc.get("id", ""),
                )
                self.store.add_tool(
                    content=result,
                    tool_call_id=tc.get("id", ""),
                    name=tc["name"],
                )

        last = self.store.last_message
        if isinstance(last, AIMessage):
            return self._extract_text(last)
        return "Agent stopped: maximum iterations reached."

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _execute_tool(self, name: str, args: dict[str, Any], tool_call_id: str) -> str:
        """Look up a tool by name and invoke it, returning the string result."""
        tool = self._tool_map.get(name)
        if tool is None:
            return (
                f"Error: tool {name!r} is not available. "
                f"Available: {list(self._tool_map)}"
            )
        try:
            return str(tool.invoke(args))
        except Exception as exc:
            return f"Tool {name} raised {type(exc).__name__}: {exc}"

    @staticmethod
    def _extract_text(message: AIMessage) -> str:
        """Safely extract text content from an AIMessage."""
        content = message.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            return "\n".join(parts)
        return str(content)

    @staticmethod
    def _log_response(response: AIMessage) -> None:
        """Print a concise summary of the LLM response."""
        if response.tool_calls:
            for tc in response.tool_calls:
                print(f"  🔧 tool_call: {tc['name']}({tc['args']})")
        else:
            preview = (
                str(response.content)[:120] + "..."
                if len(str(response.content)) > 120
                else str(response.content)
            )
            print(f"  💬 text: {preview}")

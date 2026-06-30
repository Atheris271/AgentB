#!/usr/bin/env python3
"""
B5: 记忆文档存储与查找模块
功能：记忆的查找和保存
"""

import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
import yaml


class MemoryManager:
    """B5 记忆管理类 - 负责记忆的查找和保存"""

    def __init__(self, config_path: str):
        """初始化，读取配置并加载索引"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        memory_config = config['memory']
        self.root_dir = Path(memory_config['root_dir'])
        self.global_dir = self.root_dir / memory_config['global_memory_dir']
        self.conversation_dir = self.root_dir / memory_config['conversation_memory_dir']
        self.index_path = self.root_dir / memory_config['index_path']
        self.max_chars = memory_config['max_memory_chars']

        # 确保目录存在
        self.global_dir.mkdir(parents=True, exist_ok=True)
        self.conversation_dir.mkdir(parents=True, exist_ok=True)

        # 加载或创建索引
        self.index = self._load_index()

    def _load_index(self) -> Dict:
        """加载 memory_index.json，如果不存在则创建空索引"""
        if self.index_path.exists():
            with open(self.index_path, 'r') as f:
                return json.load(f)
        return {"memories": []}

    def _save_index(self):
        """保存索引到 memory_index.json"""
        with open(self.index_path, 'w') as f:
            json.dump(self.index, f, indent=2, ensure_ascii=False)

    def _generate_memory_id(self, mem_type: str) -> str:
        """生成唯一的 memory_id"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"mem_{mem_type}_{timestamp}"

    def _find_memory_by_id(self, memory_id: str) -> Optional[Dict]:
        """根据 memory_id 查找具体的记忆文档"""
        for mem_info in self.index.get("memories", []):
            if mem_info.get("memory_id") == memory_id:
                file_path = self.root_dir / mem_info.get("path", "")
                if file_path.exists():
                    with open(file_path, 'r') as f:
                        content = f.read()
                    return {
                        "id": memory_id,
                        "content": content,
                        "type": mem_info.get("type", "unknown"),
                        "title": mem_info.get("title", ""),
                        "path": str(file_path)
                    }
        return None

    def _load_global_memory(self) -> str:
        """加载所有全局记忆文档，合并为一个字符串"""
        global_files = list(self.global_dir.glob("*.md"))
        if not global_files:
            return ""

        contents = []
        for file_path in global_files:
            with open(file_path, 'r') as f:
                content = f.read()
                contents.append(f"## {file_path.stem}\n\n{content}")

        return "\n\n---\n\n".join(contents)

    def load_memory(
        self,
        selected_memory_ids: Optional[List[str]] = None,
        use_global_memory: bool = False,
        query: str = ""
    ) -> Dict[str, Any]:
        """
        查找并返回记忆内容

        Args:
            selected_memory_ids: 用户选择的记忆ID列表
            use_global_memory: 是否加载全局记忆
            query: 用户查询（用于日志）

        Returns:
            包含 selected_memories, global_memory, total_chars, truncated, errors
        """
        result = {
            "selected_memories": [],
            "global_memory": "",
            "total_chars": 0,
            "truncated": False,
            "errors": []
        }

        # 1. 查找用户选中的记忆
        if selected_memory_ids:
            for mem_id in selected_memory_ids:
                mem_content = self._find_memory_by_id(mem_id)
                if mem_content:
                    result["selected_memories"].append(mem_content)
                    result["total_chars"] += len(mem_content.get("content", ""))
                else:
                    result["errors"].append(f"Memory ID '{mem_id}' not found")

        # 2. 加载全局记忆
        if use_global_memory:
            global_content = self._load_global_memory()
            if global_content:
                result["global_memory"] = global_content
                result["total_chars"] += len(global_content)

        # 3. 截断超长内容
        if result["total_chars"] > self.max_chars:
            result["truncated"] = True
            excess = result["total_chars"] - self.max_chars

            # 优先保留全局记忆，从选中的记忆中截断
            for mem in result["selected_memories"]:
                if excess <= 0:
                    break
                content_len = len(mem["content"])
                if content_len > excess:
                    mem["content"] = mem["content"][:content_len - excess] + "\n...[truncated]"
                    excess = 0
                else:
                    excess -= content_len
                    mem["content"] = "[truncated]"

            # 如果还不够，从全局记忆截断
            if excess > 0 and result["global_memory"]:
                g_len = len(result["global_memory"])
                if g_len > excess:
                    result["global_memory"] = result["global_memory"][:g_len - excess] + "\n...[truncated]"

        return result

    def _generate_title(self, messages: List[Dict]) -> str:
        """从消息中生成标题"""
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                title = content[:20].strip()
                if len(content) > 20:
                    title += "..."
                return title
        return "未命名对话"

    def _build_memory_document(
        self,
        memory_id: str,
        conversation_id: str,
        title: str,
        messages: List[Dict],
        trace: Dict,
        answer: str,
        save_type: str
    ) -> str:
        """构建记忆文档的 Markdown 内容"""
        doc = f"""# {title}

## 元信息
- **Memory ID**: {memory_id}
- **对话 ID**: {conversation_id}
- **类型**: {save_type}
- **创建时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 对话记录

"""
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if msg.get("tool_calls"):
                tool_calls = msg.get("tool_calls", [])
                tool_names = [t.get('name') for t in tool_calls]
                content += f"\n[工具调用: {tool_names}]"
            doc += f"### {role.upper()}\n{content}\n\n"

        doc += f"""## 运行轨迹
```json
{json.dumps(trace, indent=2, ensure_ascii=False)}

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from common.io_utils import append_jsonl, read_json, read_text, read_yaml, write_json, write_text
from common.logging_utils import now_iso
from common.path_utils import resolve_cli_path, resolve_from_file


def _memory_paths(config_path: str | Path) -> dict[str, Path | int]:
    path = Path(config_path).resolve()
    config = read_yaml(path)
    if not isinstance(config, dict) or not isinstance(config.get("memory"), dict):
        raise ValueError("memory.yaml must define a memory object")
    memory = config["memory"]
    required = ["root_dir", "global_memory_dir", "conversation_memory_dir", "index_path", "max_memory_chars"]
    missing = [name for name in required if name not in memory]
    if missing:
        raise ValueError(f"memory.yaml missing: {', '.join(missing)}")
    root = resolve_from_file(memory["root_dir"], path)
    max_chars = memory["max_memory_chars"]
    if not isinstance(max_chars, int) or isinstance(max_chars, bool) or max_chars <= 0:
        raise ValueError("max_memory_chars must be a positive integer")
    return {
        "root": root,
        "global": root / memory["global_memory_dir"],
        "conversations": root / memory["conversation_memory_dir"],
        "index": root / memory["index_path"],
        "max_chars": max_chars,
    }


def _read_index(index_path: Path) -> dict:
    if not index_path.exists():
        return {}
    index = read_json(index_path)
    if not isinstance(index, dict):
        raise ValueError("memory_index.json must be an object")
    return index


# ========== 新增辅助函数，供关键词检索和向量检索共用 ==========
def _get_all_memories(config_path: str, use_global_memory: bool) -> list[dict]:
    """获取所有可用的记忆文档列表"""
    paths = _memory_paths(config_path)
    index = _read_index(paths["index"])
    docs = []
    for memory_id, metadata in index.items():
        if not use_global_memory and metadata.get("memory_type") == "global":
            continue
        doc_path = paths["root"] / metadata.get("path", "")
        if doc_path.exists():
            content = read_text(doc_path)
            docs.append({
                "memory_id": memory_id,
                "content": content,
                "title": metadata.get("title", memory_id),
                "memory_type": metadata.get("memory_type"),
                "path": metadata.get("path"),
            })
    return docs
# ====================


def load_memory(
    config_path: str,
    selected_memory_ids: list[str],
    use_global_memory: bool,
    query: str | None = None,
    outdir: str | None = None,
) -> dict:
    if not isinstance(selected_memory_ids, list) or not all(isinstance(item, str) for item in selected_memory_ids):
        raise ValueError("selected_memory_ids must be a list of strings")
    paths = _memory_paths(config_path)
    index = _read_index(paths["index"])
    ordered_ids = []
    if use_global_memory:
        ordered_ids.extend(sorted(key for key, item in index.items() if item.get("memory_type") == "global"))
    ordered_ids.extend(selected_memory_ids)
    ordered_ids = list(dict.fromkeys(ordered_ids))

    docs = []
    errors = []
    remaining = int(paths["max_chars"])
    any_truncated = False
    for memory_id in ordered_ids:
        metadata = index.get(memory_id)
        if not isinstance(metadata, dict):
            errors.append({"memory_id": memory_id, "type": "MemoryNotFound", "message": "memory_id does not exist"})
            continue
        relative_path = metadata.get("path")
        if not isinstance(relative_path, str):
            errors.append({"memory_id": memory_id, "type": "InvalidMetadata", "message": "memory path is missing"})
            continue
        document_path = (paths["root"] / relative_path).resolve()
        try:
            document_path.relative_to(paths["root"].resolve())
        except ValueError:
            errors.append({"memory_id": memory_id, "type": "InvalidPath", "message": "memory path escapes root"})
            continue
        if not document_path.is_file():
            errors.append({"memory_id": memory_id, "type": "FileNotFoundError", "message": f"memory file not found: {relative_path}"})
            continue
        original = read_text(document_path)
        included = original[:remaining] if remaining > 0 else ""
        truncated = len(included) < len(original)
        any_truncated = any_truncated or truncated
        if included:
            docs.append(
                {
                    "memory_id": memory_id,
                    "memory_type": metadata.get("memory_type"),
                    "title": metadata.get("title", memory_id),
                    "path": relative_path,
                    "content": included,
                    "original_chars": len(original),
                    "included_chars": len(included),
                    "truncated": truncated,
                }
            )
            remaining -= len(included)
    if errors and docs:
        status = "partial"
    elif errors:
        status = "error"
    else:
        status = "success"
    result = {
        "status": status,
        "query": query,
        "selected_memory_docs": docs,
        "max_memory_chars": paths["max_chars"],
        "total_chars": sum(item["included_chars"] for item in docs),
        "truncated": any_truncated,
        "errors": errors,
    }
    if outdir:
        output_dir = Path(outdir)
        write_json(result, output_dir / "selected_memory.json")
        append_jsonl(
            {
                "timestamp": now_iso(),
                "operation": "load",
                "status": status,
                "selected_ids": [item["memory_id"] for item in docs],
                "errors": errors,
            },
            output_dir / "memory_log.jsonl",
        )
    return result


def _safe_conversation_id(conversation_id: str) -> str:
    if not isinstance(conversation_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", conversation_id):
        raise ValueError("conversation_id may only contain letters, numbers, dot, underscore, and hyphen")
    return conversation_id


def save_memory(
    config_path: str,
    conversation_id: str,
    save_type: str,
    messages_path: str,
    trace_path: str,
    answer_path: str,
    outdir: str | None = None,
    # ========== 新增参数（冲突处理，是否生成摘要，摘要最大字符） ==========
    update_strategy: str = "merge",
    use_summary: bool = False,
    max_summary_chars: int = 200,
    # ====================
) -> dict:
    conversation_id = _safe_conversation_id(conversation_id)
    if save_type not in {"conversation", "global"}:
        raise ValueError("save_type must be conversation or global")
    # ========== 校验新增参数，确保冲突处理策略合法 ==========
    if update_strategy not in {"merge", "overwrite"}:
        raise ValueError("update_strategy must be merge or overwrite")
    # ====================
    paths = _memory_paths(config_path)
    messages = read_json(messages_path)
    trace = read_json(trace_path)
    answer = read_text(answer_path).strip()
    if not isinstance(messages, list) or not isinstance(trace, dict):
        raise ValueError("messages must be an array and trace must be an object")
    now = now_iso()
    memory_id = f"mem_{save_type}_{conversation_id}"
    target_dir = paths["conversations"] if save_type == "conversation" else paths["global"]
    relative_dir = "conversations" if save_type == "conversation" else "global"
    target_path = Path(target_dir) / f"{conversation_id}.md"
    relative_path = f"{relative_dir}/{conversation_id}.md"
    title = f"{save_type.title()} {conversation_id}"
    # ========== 摘要生成逻辑 ==========
    if use_summary:
        summary = answer[:max_summary_chars] if answer else "No answer provided"
        # 这里可以扩展调用 LLM，暂时用 answer 截断
    else:
        summary = answer[:200]
    # ====================
    markdown = (
        f"# {title}\n\n"
        f"- memory_id: `{memory_id}`\n"
        f"- conversation_id: `{conversation_id}`\n"
        f"- created_or_updated_at: `{now}`\n"
        # ========== 在markdown文档中记录更新策略 ==========
        f"- update_strategy: `{update_strategy}`\n"
        # ========== 改动5结束 ==========
        "## Final Answer\n\n"
        f"{answer}\n\n"
        "## Messages\n\n```json\n"
        f"{json.dumps(messages, ensure_ascii=False, indent=2)}\n```\n\n"
        "## Trace\n\n```json\n"
        f"{json.dumps(trace, ensure_ascii=False, indent=2)}\n```\n"
    )
    # ========== 冲突管理（更新策略） ==========
    if target_path.exists() and update_strategy == "merge":
        old_content = read_text(target_path)
        # 简单合并：保留旧版，追加新版差异
        merged = (
            f"# {title} - MERGED at {now}\n\n"
            f"## ⚠️ 检测到冲突，已自动合并\n\n"
            f"### 新版本\n\n{markdown}\n\n"
            f"---\n\n"
            f"### 旧版本（已合并）\n\n{old_content}\n\n"
        )
        write_text(merged, target_path)
    else:
        write_text(markdown, target_path)
    # ====================
    index = _read_index(paths["index"])
    existing = index.get(memory_id, {})
    created_at = existing.get("created_at", now)
    index[memory_id] = {
        "memory_id": memory_id,
        "memory_type": save_type,
        "title": title,
        "summary": summary,
        "path": relative_path,
        "conversation_id": conversation_id,
        "created_at": created_at,
        "updated_at": now,
        # ========== 记录摘要和更新策略 ==========
        "update_strategy": update_strategy,
        # ====================
    }
    write_json(index, paths["index"])
    result = {
        "status": "success",
        "memory_id": memory_id,
        "memory_type": save_type,
        "conversation_id": conversation_id,
        "title": title,
        "summary": summary,
        "path": relative_path,
        "index_path": Path(paths["index"]).name,
        "created_at": created_at,
        "updated_at": now,
        # ========== 返回更新策略 ==========
        "update_strategy": update_strategy,
        # ====================
        "source_paths": {
            "messages": str(messages_path),
            "trace": str(trace_path),
            "answer": str(answer_path),
        },
    }
    if outdir:
        output_dir = Path(outdir)
        write_json(result, output_dir / "saved_memory.json")
        append_jsonl(
            {"timestamp": now, "operation": "save", "status": "success", "memory_id": memory_id},
            output_dir / "memory_log.jsonl",
        )
    return result


# ========== 新增关键词检索函数 ==========
def search_memory_by_keyword(
    config_path: str,
    query: str,
    top_k: int = 5,
    use_global_memory: bool = True,
    outdir: str | None = None,
) -> dict:
    """按关键词检索记忆，返回最相关的 top_k 个文档"""
    try:
        import jieba
    except ImportError:
        return {"status": "error", "message": "jieba not installed. Run: pip install jieba"}
    from collections import Counter
    import math

    docs = _get_all_memories(config_path, use_global_memory)
    if not docs:
        return {"status": "error", "message": "No documents found", "results": []}

    def tokenize(text: str) -> list[str]:
        return [w for w in jieba.cut(text) if len(w.strip()) > 1]

    query_tokens = tokenize(query)
    doc_tokens_list = [tokenize(d["content"]) for d in docs]

    doc_freq = Counter()
    for tokens in doc_tokens_list:
        for t in set(tokens):
            doc_freq[t] += 1

    total = len(docs)
    idf = {t: math.log((total + 1) / (f + 1)) + 1 for t, f in doc_freq.items()}

    def tfidf(tokens: list[str]) -> dict:
        tf = Counter(tokens)
        return {t: tf[t] * idf.get(t, 0) for t in set(tokens)}

    q_vec = tfidf(query_tokens)

    def cosine(v1: dict, v2: dict) -> float:
        if not v1 or not v2:
            return 0.0
        inter = set(v1.keys()) & set(v2.keys())
        dot = sum(v1[t] * v2[t] for t in inter)
        n1 = math.sqrt(sum(v ** 2 for v in v1.values()))
        n2 = math.sqrt(sum(v ** 2 for v in v2.values()))
        return dot / (n1 * n2) if n1 and n2 else 0.0

    for d, tokens in zip(docs, doc_tokens_list):
        d["score"] = cosine(q_vec, tfidf(tokens))

    sorted_docs = sorted(docs, key=lambda x: x["score"], reverse=True)[:top_k]
    result = {
        "status": "success",
        "query": query,
        "top_k": top_k,
        "total_docs": len(docs),
        "results": [
            {
                "memory_id": d["memory_id"],
                "title": d["title"],
                "memory_type": d["memory_type"],
                "content_preview": d["content"][:300],
                "score": round(d["score"], 4)
            }
            for d in sorted_docs
        ]
    }
    if outdir:
        output_dir = Path(outdir)
        write_json(result, output_dir / "keyword_search_result.json")
    return result
# ====================


# ========== 新增向量检索函数 ==========
def vector_search_memory(
    config_path: str,
    query: str,
    top_k: int = 5,
    model_name: str = "BAAI/bge-small-zh-v1.5",
    use_global_memory: bool = True,
    outdir: str | None = None,
) -> dict:
    """使用向量检索搜索记忆"""
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        return {
            "status": "error",
            "message": "sentence-transformers not installed. Run: pip install sentence-transformers numpy"
        }

    docs = _get_all_memories(config_path, use_global_memory)
    if not docs:
        return {"status": "error", "message": "No documents found", "results": []}

    try:
        model = SentenceTransformer(model_name)
        texts = [d["content"][:1000] for d in docs]
        embeddings = model.encode(texts, normalize_embeddings=True)
        q_emb = model.encode([query], normalize_embeddings=True)

        import numpy as np
        similarities = np.dot(embeddings, q_emb.T).flatten()
        indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in indices:
            results.append({
                "memory_id": docs[idx]["memory_id"],
                "title": docs[idx]["title"],
                "memory_type": docs[idx]["memory_type"],
                "content_preview": docs[idx]["content"][:300],
                "score": round(float(similarities[idx]), 4)
            })

        result = {
            "status": "success",
            "query": query,
            "top_k": top_k,
            "total_docs": len(docs),
            "model": model_name,
            "results": results
        }
    except Exception as e:
        return {"status": "error", "message": f"Vector search failed: {str(e)}"}

    if outdir:
        output_dir = Path(outdir)
        write_json(result, output_dir / "vector_search_result.json")
    return result
# ====================


def parse_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select or save local memory documents.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--select_memory_ids", nargs="*")
    parser.add_argument("--use_global_memory", type=parse_bool)
    parser.add_argument("--query")
    parser.add_argument("--save_type", choices=["conversation", "global"])
    parser.add_argument("--save_input_path")
    parser.add_argument("--outdir", required=True)
    # ========== 新增进阶功能参数 ==========
    parser.add_argument("--search_keyword", help="关键词检索")
    parser.add_argument("--top_k", type=int, default=5, help="Top-K 检索数量")
    parser.add_argument("--vector_search", action="store_true", help="使用向量检索")
    parser.add_argument("--embedding_model", default="BAAI/bge-small-zh-v1.5", help="Embedding 模型名称")
    parser.add_argument("--update_strategy", choices=["merge", "overwrite"], default="merge", help="更新策略")
    parser.add_argument("--use_summary", type=parse_bool, default=False, help="是否使用摘要")
    parser.add_argument("--max_summary_chars", type=int, default=200, help="摘要最大字符数")
    # ====================
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config_path = resolve_cli_path(args.config)
        outdir = resolve_cli_path(args.outdir)
        # ========== 新增关键词检索分支 ==========
        if args.search_keyword:
            result = search_memory_by_keyword(
                str(config_path),
                args.search_keyword,
                args.top_k,
                bool(args.use_global_memory or True),
                str(outdir),
            )
            print(f"keyword search completed: {outdir / 'keyword_search_result.json'}")
            return 0
        # ====================

        # ========== 新增向量检索分支 ==========
        if args.vector_search:
            if not args.query:
                raise ValueError("--query is required for vector search")
            result = vector_search_memory(
                str(config_path),
                args.query,
                args.top_k,
                args.embedding_model,
                bool(args.use_global_memory or True),
                str(outdir),
            )
            print(f"vector search completed: {outdir / 'vector_search_result.json'}")
            return 0
        # ====================

        if args.save_type or args.save_input_path:
            if not args.save_type or not args.save_input_path:
                raise ValueError("--save_type and --save_input_path must be provided together")
            input_path = resolve_cli_path(args.save_input_path)
            payload = read_json(input_path)
            if payload.get("save_type") != args.save_type:
                raise ValueError("CLI save_type must match memory_save_input.json")
            base = input_path.parent
            # ========== 传递新增参数 ==========
            result = save_memory(
                str(config_path),
                payload["conversation_id"],
                args.save_type,
                str((base / payload["messages_path"]).resolve()),
                str((base / payload["trace_path"]).resolve()),
                str((base / payload["answer_path"]).resolve()),
                str(outdir),
                update_strategy=args.update_strategy,
                use_summary=bool(args.use_summary),
                max_summary_chars=args.max_summary_chars,
            )
            # ====================
            print(outdir / "saved_memory.json")
        else:
            if args.select_memory_ids is None and args.use_global_memory is None:
                raise ValueError("select mode requires --select_memory_ids or --use_global_memory")
            result = load_memory(
                str(config_path),
                args.select_memory_ids or [],
                bool(args.use_global_memory),
                args.query,
                str(outdir),
            )
            print(outdir / "selected_memory.json")
        return 0
    except Exception as exc:
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

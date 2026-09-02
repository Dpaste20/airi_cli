import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .config import MemoryConfig
from .storage import MemoryStorage
from .vectors import MemoryVectors

MEMORY_TOOLS_GUIDE = """<memory-tools-guide>
## Memory tools usage guide

If the injected memories above are not enough to answer the user's question, you may actively call:
- airi_memory_search: search structured memories (L1) - preferences, historical events, rules.
- airi_conversation_search: search raw conversations (L0) - exact message text, timeline details.

Limit: at most 3 combined calls per turn. If no results after 3 searches, answer from what you know.
</memory-tools-guide>"""


@dataclass
class RecallResult:
    prepend_context: Optional[str] = None
    recalled_memories: List[Dict[str, Any]] = field(default_factory=list)
    persona_loaded: bool = False
    scenes_loaded: bool = False


def _build_fts_query(query: str) -> str:
    terms = re.findall(r"[\w\u4e00-\u9fff]+", query.lower())
    if not terms:
        return ""
    if len(query.strip()) <= 60 and " " not in query.strip():
        return '"' + query.strip().replace('"', "") + '"'
    return " OR ".join(f'"{t}"' for t in terms[:8])


def _rrf_fuse(
    fts_results: List[Dict[str, Any]],
    vec_results: List[Dict[str, Any]],
    k: int,
    limit: int,
) -> List[Dict[str, Any]]:
    scores: Dict[str, float] = {}
    items: Dict[str, Dict[str, Any]] = {}

    for rank, row in enumerate(fts_results):
        key = str(row.get("record_id") or row["id"])
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        items[key] = row

    for rank, row in enumerate(vec_results):
        payload = row["payload"] or {}
        key = str(payload.get("record_id") or row["id"])
        row["_vec_score"] = row["score"]
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        if key not in items:
            items[key] = {
                "id": -1,
                "record_id": key,
                "content": payload.get("content", ""),
                "type": payload.get("type", "unknown"),
                "priority": payload.get("priority", 0),
                "scene_name": payload.get("scene_name", ""),
                "session_id": payload.get("session_id", ""),
                "metadata": "{}",
            }

    ranked = sorted(items.items(), key=lambda kv: scores[kv[0]], reverse=True)
    return [row for _, row in ranked[:limit]]


def _format_memory_line(m: Dict[str, Any]) -> str:
    line = f"- [{m.get('type', 'unknown')}|{m.get('priority', 0)}] {m.get('content', '')}"
    scene = m.get("scene_name") or ""
    if scene:
        line += f" (scene: {scene})"
    return line


def search_l1_hybrid(
    storage: MemoryStorage,
    vectors: MemoryVectors,
    query: str,
    config: MemoryConfig,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    limit = limit or config.recall.max_results
    k = config.recall.rrf_k

    fts_query = _build_fts_query(query)
    fts_results = (
        storage.search_memories_fts(fts_query, limit * 4) if fts_query else []
    )
    vec_results = vectors.search_l1(
        query, limit * 4, config.recall.score_threshold
    )
    return _rrf_fuse(fts_results, vec_results, k, limit)


def _apply_char_budget(memories: List[Dict[str, Any]], max_chars: int) -> List[Dict[str, Any]]:
    if max_chars <= 0:
        return memories
    budget = max_chars
    kept: List[Dict[str, Any]] = []
    for m in memories:
        if budget <= 0:
            break
        line = _format_memory_line(m)
        kept.append(m)
        budget -= len(line)
    return kept


def perform_recall(
    storage: MemoryStorage,
    vectors: MemoryVectors,
    config: MemoryConfig,
    user_text: str,
    max_chars: Optional[int] = None,
) -> RecallResult:
    if not storage or not config.enabled:
        return RecallResult()

    max_chars = max_chars or config.recall.max_total_recall_chars
    memories = search_l1_hybrid(storage, vectors, user_text, config)
    memories = _apply_char_budget(memories, max_chars)

    persona_loaded = False
    scenes_loaded = False
    stable_parts: List[str] = []

    persona_text = ""
    try:
        with open(config.persona_path, "r", encoding="utf-8") as f:
            persona_text = f.read().strip()
    except OSError:
        persona_text = ""
    if persona_text:
        persona_loaded = True
        stable_parts.append(f"<user-persona>\n{persona_text}\n</user-persona>")

    scenes = storage.list_scenes()
    if scenes:
        scenes_loaded = True
        nav_lines = [
            "- {name} | summary: {summary} | file: {file}".format(
                name=s["name"], summary=s["summary"], file=s["filename"]
            )
            for s in scenes[:10]
        ]
        stable_parts.append(
            "<scene-navigation>\nAvailable scene blocks (drill down by asking "
            "the file_search tool or airi_memory_search):\n"
            + "\n".join(nav_lines)
            + "\n</scene-navigation>"
        )

    prepend_parts: List[str] = []
    if memories:
        lines = [_format_memory_line(m) for m in memories]
        prepend_parts.append(
            "<relevant-memories>\n"
            "Here are memories recalled for the current conversation. They are "
            "references only, not part of the current task state:\n\n"
            + "\n".join(lines)
            + "\n</relevant-memories>"
        )
    if stable_parts:
        stable_parts.append(MEMORY_TOOLS_GUIDE)
        prepend_parts.append("\n\n".join(stable_parts))

    return RecallResult(
        prepend_context="\n\n".join(prepend_parts) if prepend_parts else None,
        recalled_memories=memories,
        persona_loaded=persona_loaded,
        scenes_loaded=scenes_loaded,
    )


def search_conversations_hybrid(
    storage: MemoryStorage,
    vectors: MemoryVectors,
    config: MemoryConfig,
    query: str,
    limit: int = 10,
    session_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    k = config.recall.rrf_k
    limit = limit or config.recall.tool_max_results

    fts_query = _build_fts_query(query)
    fts_results = (
        storage.search_conversations_fts(fts_query, limit * 4, session_id)
        if fts_query
        else []
    )
    vec_results = vectors.search_l0(query, limit * 4, config.recall.score_threshold)

    scores: Dict[str, float] = {}
    items: Dict[str, Dict[str, Any]] = {}
    for rank, row in enumerate(fts_results):
        key = str(row.get("message_id") or row["id"])
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        items[key] = row
    for rank, row in enumerate(vec_results):
        payload = row["payload"] or {}
        key = str(payload.get("message_id") or row["id"])
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        if key not in items:
            items[key] = {
                "id": payload.get("message_id", ""),
                "session_id": payload.get("session_id", ""),
                "role": payload.get("role", ""),
                "content": payload.get("content", ""),
                "ts": payload.get("ts", 0),
            }

    ranked = sorted(items.items(), key=lambda kv: scores[kv[0]], reverse=True)
    return [row for _, row in ranked[:limit]]
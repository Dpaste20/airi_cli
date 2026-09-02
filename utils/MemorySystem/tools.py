from typing import Optional

from agno.tools import tool

from . import _system, delete_memory, wipe_memory
from .recall import search_conversations_hybrid, search_l1_hybrid


@tool
def airi_memory_search(query: str, limit: int = 5) -> str:
    """Search Airi's structured long-term memories (preferences, past events,
    instructions). Use when you need facts about the user that you may have
    forgotten or that were extracted from earlier conversations.
    """
    system = _system()
    if not system:
        return "Error: Memory system not initialized."
    try:
        results = search_l1_hybrid(
            system.storage,
            system.vectors,
            query,
            system.config,
            limit=min(limit, system.config.recall.tool_max_results),
        )
        if not results:
            return "No memories found for this query."
        lines = [
            f"- [{m.get('type', 'unknown')}|{m.get('priority', 0)}] {m.get('content', '')}"
            for m in results
        ]
        return "\n".join(lines)
    except Exception as e:
        return f"Error searching memories: {str(e)}"


@tool
def airi_conversation_search(
    query: str, session_id: Optional[str] = None, limit: int = 5
) -> str:
    """Search Airi's raw conversation history (exact message text, timeline).
    Use to recall exactly what was said before, or to verify details of past turns.
    """
    system = _system()
    if not system:
        return "Error: Memory system not initialized."
    try:
        results = search_conversations_hybrid(
            system.storage,
            system.vectors,
            system.config,
            query,
            limit=min(limit, system.config.recall.tool_max_results),
            session_id=session_id,
        )
        if not results:
            return "No conversations found for this query."
        lines = []
        for r in results:
            role = r.get("role", "?")
            content = (r.get("content") or "").strip().replace("\n", " ")[:300]
            lines.append(f"[{role}] ({r.get('session_id', '?')}) {content}")
        return "\n".join(lines)
    except Exception as e:
        return f"Error searching conversations: {str(e)}"


@tool
def airi_delete_memory(memory_query: str) -> str:
    """Delete memories from Airi's long-term memory that match the given
    description. Use only when the user explicitly asks to forget or delete
    specific memories. Returns what was removed.
    """
    try:
        result = delete_memory(memory_query, limit=3)
    except Exception as e:
        return f"Error deleting memory: {e}"
    if not result.get("ok"):
        return f"Delete failed: {result.get('error', 'unknown error')}"
    deleted = result.get("deleted", [])
    if not deleted:
        return "No matching memories found to delete."
    lines = [f"- {d['content']}" for d in deleted]
    return f"Deleted {len(deleted)} memory/memories:\n" + "\n".join(lines)


@tool
def airi_wipe_memory(confirm: bool = False) -> str:
    """WIPE ALL of Airi's memory: raw conversations, extracted memories,
    scenes, persona, and vector index. Irreversible. The `confirm` parameter
    MUST be true to proceed.
    """
    if not confirm:
        return (
            "Wipe requires confirmation. Call again with confirm=true if you "
            "really want to erase all memory."
        )
    try:
        result = wipe_memory()
    except Exception as e:
        return f"Error wiping memory: {e}"
    if not result.get("ok"):
        return f"Wipe failed: {result.get('error', 'unknown error')}"
    counts = result.get("wiped", {})
    return (
        "All memory wiped. "
        f"Conversations: {counts.get('l0_messages', 0)}, "
        f"Memories: {counts.get('l1_memories', 0)}, "
        f"Scenes: {counts.get('scenes', 0)}. "
        "Persona and scene files removed."
    )
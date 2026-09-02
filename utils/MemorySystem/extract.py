import re
import time
import uuid
from typing import Any, Dict, List, Optional

from .llm import MemoryLlm
from .prompts import EXTRACT_MEMORIES_SYSTEM_PROMPT, format_extraction_user_prompt
from .storage import MEMORY_TYPES, MemoryStorage
from .vectors import MemoryVectors

DUPLICATE_SIMILARITY = 0.97


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", text.lower())


def _deduplicate(
    candidate: Dict[str, Any],
    storage: MemoryStorage,
    vectors: MemoryVectors,
) -> bool:
    normalized = _normalize(candidate.get("content", ""))
    if len(normalized) < 20:
        return True

    for existing in storage.search_memories_fts(f'"{normalized[:40]}"', 5):
        if _normalize(existing["content"]) == normalized:
            return True

    vec = vectors.get_embedding_or_none(candidate["content"])
    if not vec:
        return False
    try:
        for p in vectors.search_l1_with_vector(vec, 1, DUPLICATE_SIMILARITY):
            payload = p["payload"]
            if payload.get("type") == candidate.get("type") and p["score"] >= DUPLICATE_SIMILARITY:
                return True
    except Exception as e:
        print(f"Warning: dedup vector search failed: {e}")
    return False


def extract_l1_memories(
    storage: MemoryStorage,
    vectors: MemoryVectors,
    llm: MemoryLlm,
    session_id: str,
    batch_limit: int,
    previous_scene_name: str = "",
) -> Dict[str, Any]:
    new_messages = storage.get_unextracted_messages(session_id, batch_limit)
    if not new_messages:
        return {"extracted": 0, "stored": 0, "new_scene": previous_scene_name}

    background = new_messages[-6:] if len(new_messages) > 6 else []
    user_prompt = format_extraction_user_prompt(
        new_messages, background, previous_scene_name
    )
    result = llm.chat_json(EXTRACT_MEMORIES_SYSTEM_PROMPT, user_prompt)
    if result is None:
        return {"extracted": 0, "stored": 0, "new_scene": previous_scene_name}

    if isinstance(result, dict) and "scenes" in result:
        scenes = result["scenes"]
    elif isinstance(result, list):
        scenes = result
    else:
        scenes = []

    stored = 0
    last_scene = previous_scene_name
    processed_ids = set()

    for scene in scenes:
        if not isinstance(scene, dict):
            continue
        scene_name = str(scene.get("scene_name") or "").strip()
        if scene_name:
            last_scene = scene_name

        for memory in scene.get("memories", []) or []:
            if not isinstance(memory, dict):
                continue
            content = str(memory.get("content") or "").strip()
            mem_type = str(memory.get("type") or "").strip()
            if not content or mem_type not in MEMORY_TYPES:
                continue

            priority = memory.get("priority")
            try:
                priority = int(priority)
            except (TypeError, ValueError):
                priority = 50

            source_ids = [
                int(i)
                for i in (memory.get("source_message_ids") or [])
                if str(i).isdigit()
            ]
            if not source_ids:
                source_ids = [m["id"] for m in new_messages]

            if _deduplicate(memory, storage, vectors):
                continue

            record_id = f"l1_{uuid.uuid4().hex[:12]}"
            storage.insert_memory(
                record_id=record_id,
                content=content,
                mem_type=mem_type,
                priority=priority,
                scene_name=scene_name,
                session_id=session_id,
                source_msg_ids=source_ids,
                metadata=memory.get("metadata") or {},
            )
            vectors.upsert_l1(
                point_id=record_id,
                text=content,
                payload={
                    "content": content,
                    "type": mem_type,
                    "priority": priority,
                    "scene_name": scene_name,
                    "session_id": session_id,
                    "record_id": record_id,
                },
            )
            stored += 1
            processed_ids.update(source_ids)

    storage.mark_messages_extracted([m["id"] for m in new_messages])
    return {
        "extracted": len(new_messages),
        "stored": stored,
        "new_scene": last_scene,
    }
import asyncio
import time
from typing import Dict, List, Optional

from .config import MemoryConfig, load_memory_config
from .llm import MemoryLlm
from .pipeline import MemoryPipeline
from .recall import RecallResult, perform_recall
from .storage import MemoryStorage
from .vectors import MemoryVectors

_memory_system: Optional["MemorySystem"] = None


class MemorySystem:
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.storage = MemoryStorage(config)
        self.vectors = MemoryVectors(config)
        self.llm = MemoryLlm(config)
        self.pipeline = MemoryPipeline(config, self.storage, self.vectors, self.llm)

    def start(self) -> None:
        if not self.config.enabled:
            print("[memory] Disabled by config")
            return
        self.pipeline.start()
        print(
            f"[memory] System initialized: db={self.config.db_path}, "
            f"llm={self.config.llm.id}, embeddings={self.config.embedding.id}"
        )

    async def shutdown(self) -> None:
        if not self.config.enabled:
            return
        try:
            await self.pipeline.shutdown()
        finally:
            self.storage.close()
        print("[memory] System shut down")


def _system() -> Optional[MemorySystem]:
    return _memory_system


def initialize(config_path: str = "config.toml") -> Optional[MemorySystem]:
    global _memory_system
    config = load_memory_config(config_path)
    _memory_system = MemorySystem(config)
    _memory_system.start()
    return _memory_system


async def ashutdown() -> None:
    global _memory_system
    if _memory_system:
        await _memory_system.shutdown()
        _memory_system = None


def capture_turn(session_id: str, user_text: str, assistant_text: str) -> None:
    system = _system()
    if not system or not system.config.enabled:
        return
    user_text = (user_text or "").strip()
    assistant_text = (assistant_text or "").strip()
    if not user_text:
        return
    ts = int(time.time() * 1000)
    stamp = f"{session_id}_{ts}"
    message_id = f"l0_{stamp}_u"
    system.storage.insert_message(message_id, session_id, "user", user_text)
    if assistant_text:
        assistant_id = f"l0_{stamp}_a"
        system.storage.insert_message(assistant_id, session_id, "assistant", assistant_text)
        system.vectors.upsert_l0(
            assistant_id,
            assistant_text,
            {
                "message_id": assistant_id,
                "session_id": session_id,
                "role": "assistant",
                "content": assistant_text,
                "ts": ts,
            },
        )


def notify_turn(session_id: str) -> None:
    system = _system()
    if not system or not system.config.enabled:
        return
    system.pipeline.submit_turn(session_id)


async def recall_for_prompt(user_text: str) -> Optional[str]:
    system = _system()
    if not system or not system.config.enabled or not user_text:
        return None
    try:
        result: RecallResult = await asyncio.to_thread(
            perform_recall,
            system.storage,
            system.vectors,
            system.config,
            user_text,
        )
    except Exception as e:
        print(f"[memory] recall failed (non-blocking): {e}")
        return None
    return result.prepend_context


def capture_and_notify(session_id: str, user_text: str, assistant_text: str) -> None:
    capture_turn(session_id, user_text, assistant_text)
    notify_turn(session_id)


def delete_memory(query: str, limit: int = 3) -> Dict:
    """Delete memories whose content best matches `query`. Returns what was removed."""
    system = _system()
    if not system:
        return {"ok": False, "error": "Memory system not initialized."}
    from .recall import search_l1_hybrid

    matches = search_l1_hybrid(
        system.storage, system.vectors, query, system.config, limit=limit
    )
    if not matches:
        return {"ok": False, "deleted": [], "error": "No matching memories found."}

    deleted = []
    for m in matches:
        record_id = m.get("record_id") or ""
        if system.storage.delete_memory(record_id):
            system.vectors.delete_l1(record_id)
            deleted.append({"record_id": record_id, "content": m.get("content", "")})
    return {"ok": True, "deleted": deleted}


async def adelete_memory(query: str, limit: int = 3) -> Dict:
    return await asyncio.to_thread(delete_memory, query, limit)


def wipe_memory() -> Dict:
    """Wipe ALL memory: conversations, L1 memories, scenes, persona, vectors, pipeline state."""
    system = _system()
    if not system:
        return {"ok": False, "error": "Memory system not initialized."}
    import os
    import shutil

    counts = system.storage.wipe_all()
    system.vectors.wipe_l0()
    system.vectors.wipe_l1()
    for path in (system.config.persona_path, system.config.scenes_dir):
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                try:
                    os.remove(path)
                except OSError:
                    pass
    return {"ok": True, "wiped": counts}


async def awipe_memory() -> Dict:
    return await asyncio.to_thread(wipe_memory)
import asyncio
import os
import time
from typing import Dict, Optional, Set

from .config import MemoryConfig
from .extract import extract_l1_memories
from .llm import MemoryLlm
from .persona import generate_persona
from .scenes import consolidate_scenes
from .storage import MemoryStorage
from .vectors import MemoryVectors

WORKER_INTERVAL_SECONDS = 60


class MemoryPipeline:
    def __init__(
        self,
        config: MemoryConfig,
        storage: MemoryStorage,
        vectors: MemoryVectors,
        llm: MemoryLlm,
    ):
        self.config = config
        self.storage = storage
        self.vectors = vectors
        self.llm = llm
        self._session_locks: Dict[str, asyncio.Lock] = {}
        self._idle_timers: Dict[str, asyncio.Task] = {}
        self._l1_rerun: Set[str] = set()
        self._tasks: Set[asyncio.Task] = set()
        self._worker: Optional[asyncio.Task] = None
        self._pending_start = False
        self._shutdown = False

    # ── lifecycle ──

    def start(self) -> None:
        if self._worker is not None:
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            self._pending_start = True
            return
        self._worker = asyncio.create_task(self._worker_loop())
        self._tasks.add(self._worker)

    @property
    def is_running(self) -> bool:
        return self._worker is not None

    async def shutdown(self) -> None:
        self._shutdown = True
        for timer in list(self._idle_timers.values()):
            timer.cancel()
        self._idle_timers.clear()
        if self._worker:
            self._worker.cancel()
        for task in list(self._tasks):
            if task is not self._worker:
                task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        self._worker = None

    # ── turn submission ──

    def submit_turn(self, session_id: str) -> None:
        if not self.config.enabled:
            return
        if self._pending_start and self._worker is None:
            self._pending_start = False
            self.start()
        state = self.storage.get_session_state(session_id)
        count = state["conversation_count"] + 1
        threshold = state["warmup_threshold"]
        self.storage.update_session_state(session_id, conversation_count=count)

        self._reset_idle_timer(session_id)

        if count >= threshold:
            self._schedule_l1(session_id)

    # ── scheduling ──

    def _reset_idle_timer(self, session_id: str) -> None:
        existing = self._idle_timers.pop(session_id, None)
        if existing:
            existing.cancel()

        timeout = self.config.pipeline.l1_idle_timeout_seconds

        async def _idle_fire():
            try:
                await asyncio.sleep(timeout)
            except asyncio.CancelledError:
                return
            self._idle_timers.pop(session_id, None)
            self._schedule_l1(session_id)

        self._idle_timers[session_id] = asyncio.create_task(_idle_fire())

    def _schedule_l1(self, session_id: str) -> None:
        task = asyncio.create_task(self._run_l1(session_id))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _schedule_l2(self, session_id: str) -> None:
        task = asyncio.create_task(self._run_l2(session_id))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _schedule_l3(self) -> None:
        task = asyncio.create_task(self._run_l3())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _lock_for(self, session_id: str) -> asyncio.Lock:
        lock = self._session_locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._session_locks[session_id] = lock
        return lock

    # ── pipeline stages ──

    async def _run_l1(self, session_id: str) -> None:
        lock = self._lock_for(session_id)
        if lock.locked():
            self._l1_rerun.add(session_id)
            return
        async with lock:
            try:
                state = self.storage.get_session_state(session_id)
                batch_limit = self.config.pipeline.l1_batch_limit
                result = await asyncio.to_thread(
                    extract_l1_memories,
                    self.storage,
                    self.vectors,
                    self.llm,
                    session_id,
                    batch_limit,
                    state["last_scene_name"],
                )

                now_ms = int(time.time() * 1000)
                prev_count = state["conversation_count"]
                new_threshold = state["warmup_threshold"]
                if self.config.pipeline.enable_warmup:
                    new_threshold = min(
                        new_threshold * 2, self.config.pipeline.every_n_conversations
                    )

                self.storage.update_session_state(
                    session_id,
                    conversation_count=max(prev_count - result.get("extracted", 0), 0),
                    warmup_threshold=new_threshold,
                    last_l1_at=now_ms,
                    last_scene_name=result.get("new_scene", state["last_scene_name"]),
                    l2_pending=1,
                )
                print(
                    f"[memory] L1 extract: session={session_id} "
                    f"msgs={result.get('extracted', 0)} stored={result.get('stored', 0)}"
                )
            finally:
                if session_id in self._l1_rerun:
                    self._l1_rerun.discard(session_id)
                    self._schedule_l1(session_id)
                    return

        self._maybe_run_l2(session_id)

    def _maybe_run_l2(self, session_id: str) -> None:
        state = self.storage.get_session_state(session_id)
        now_ms = int(time.time() * 1000)
        min_interval = self.config.pipeline.l2_min_interval_seconds * 1000
        elapsed = now_ms - state["last_l2_at"]
        if elapsed >= min_interval:
            self._schedule_l2(session_id)
            self.storage.update_session_state(session_id, l2_pending=0)
        else:
            self.storage.update_session_state(session_id, l2_pending=1)

    async def _run_l2(self, session_id: str) -> None:
        lock = self._lock_for(session_id)
        async with lock:
            state = self.storage.get_session_state(session_id)
            cursor = state["last_l2_cursor"]
            result = await asyncio.to_thread(
                consolidate_scenes,
                self.storage,
                self.llm,
                self.config,
                since_memory_id=cursor,
            )
            latest_id = self.storage.latest_memory_id()
            self.storage.update_session_state(
                session_id,
                last_l2_at=int(time.time() * 1000),
                last_l2_cursor=latest_id,
                l2_pending=0,
            )
            print(
                f"[memory] L2 scenes: session={session_id} "
                f"processed={result.get('processed', 0)} written={result.get('written', 0)}"
            )

        self._maybe_run_l3()

    def _maybe_run_l3(self) -> None:
        total = self.storage.count_memories()
        session_states = [self.storage.get_session_state(s) for s in self.storage.list_active_sessions()]
        latest = max(
            (s["last_persona_count"] for s in session_states),
            default=0,
        )
        persona_exists = os.path.exists(self.config.persona_path)
        if (not persona_exists and total > 0) or (
            total - latest >= self.config.persona.trigger_every_n_memories
        ):
            self._schedule_l3()

    def _on_l3_done(self, total_memories: int) -> None:
        for session_id in self.storage.list_active_sessions():
            self.storage.update_session_state(
                session_id, l3_pending=0, last_persona_count=total_memories
            )

    async def _run_l3(self) -> None:
        result = await asyncio.to_thread(
            generate_persona, self.storage, self.llm, self.config,
            self.config.persona.max_persona_chars,
        )
        self._on_l3_done(self.storage.count_memories())
        print(
            f"[memory] L3 persona: generated={result.get('generated', False)} "
            f"chars={result.get('chars', 0)}"
        )

    # ── worker loop for pending L2 ──

    async def _worker_loop(self) -> None:
        while not self._shutdown:
            try:
                now_ms = int(time.time() * 1000)
                min_interval = self.config.pipeline.l2_min_interval_seconds * 1000
                for session_id in self.storage.list_active_sessions():
                    state = self.storage.get_session_state(session_id)
                    if state["l2_pending"] and state["last_l1_at"] > 0:
                        elapsed = now_ms - state["last_l2_at"]
                        if elapsed >= min_interval:
                            self._schedule_l2(session_id)
                            self.storage.update_session_state(session_id, l2_pending=0)
            except Exception as e:
                print(f"[memory] worker loop error: {e}")
            await asyncio.sleep(WORKER_INTERVAL_SECONDS)
import json
import os
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional

from .config import MemoryConfig

SCHEMA = """
CREATE TABLE IF NOT EXISTS l0_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id TEXT UNIQUE,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    ts INTEGER NOT NULL,
    recorded_at TEXT NOT NULL,
    l1_extracted INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_l0_session ON l0_messages(session_id, ts);
CREATE INDEX IF NOT EXISTS idx_l0_extracted ON l0_messages(l1_extracted);

CREATE VIRTUAL TABLE IF NOT EXISTS l0_messages_fts USING fts5(
    content,
    content='l0_messages',
    content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS l0_fts_ai AFTER INSERT ON l0_messages BEGIN
    INSERT INTO l0_messages_fts(rowid, content) VALUES (new.id, new.content);
END;
CREATE TRIGGER IF NOT EXISTS l0_fts_ad AFTER DELETE ON l0_messages BEGIN
    INSERT INTO l0_messages_fts(l0_messages_fts, rowid, content)
    VALUES ('delete', old.id, old.content);
END;

CREATE TABLE IF NOT EXISTS l1_memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id TEXT UNIQUE,
    content TEXT NOT NULL,
    type TEXT NOT NULL,
    priority INTEGER NOT NULL DEFAULT 0,
    scene_name TEXT NOT NULL DEFAULT '',
    session_id TEXT DEFAULT '',
    source_msg_ids TEXT NOT NULL DEFAULT '[]',
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_l1_type ON l1_memories(type);
CREATE INDEX IF NOT EXISTS idx_l1_scene ON l1_memories(scene_name);
CREATE INDEX IF NOT EXISTS idx_l1_session ON l1_memories(session_id);

CREATE VIRTUAL TABLE IF NOT EXISTS l1_memories_fts USING fts5(
    content,
    content='l1_memories',
    content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS l1_fts_ai AFTER INSERT ON l1_memories BEGIN
    INSERT INTO l1_memories_fts(rowid, content) VALUES (new.id, new.content);
END;
CREATE TRIGGER IF NOT EXISTS l1_fts_ad AFTER DELETE ON l1_memories BEGIN
    INSERT INTO l1_memories_fts(l1_memories_fts, rowid, content)
    VALUES ('delete', old.id, old.content);
END;
CREATE TRIGGER IF NOT EXISTS l1_fts_au AFTER UPDATE ON l1_memories BEGIN
    INSERT INTO l1_memories_fts(l1_memories_fts, rowid, content)
    VALUES ('delete', old.id, old.content);
    INSERT INTO l1_memories_fts(rowid, content) VALUES (new.id, new.content);
END;

CREATE TABLE IF NOT EXISTS scenes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    filename TEXT UNIQUE NOT NULL,
    summary TEXT NOT NULL DEFAULT '',
    heat INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pipeline_state (
    session_id TEXT PRIMARY KEY,
    conversation_count INTEGER NOT NULL DEFAULT 0,
    warmup_threshold INTEGER NOT NULL DEFAULT 1,
    last_l1_at INTEGER NOT NULL DEFAULT 0,
    last_l2_at INTEGER NOT NULL DEFAULT 0,
    last_scene_name TEXT NOT NULL DEFAULT '',
    memory_count INTEGER NOT NULL DEFAULT 0,
    l2_pending INTEGER NOT NULL DEFAULT 0,
    l3_pending INTEGER NOT NULL DEFAULT 0,
    last_persona_count INTEGER NOT NULL DEFAULT 0,
    last_l2_cursor INTEGER NOT NULL DEFAULT 0
);
"""

MEMORY_TYPES = ("persona", "episodic", "instruction")


class MemoryStorage:
    def __init__(self, config: MemoryConfig):
        self.config = config
        os.makedirs(os.path.dirname(config.db_path), exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(config.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur

    def _query(self, sql: str, params: tuple = ()) -> List[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def _query_one(self, sql: str, params: tuple = ()) -> Optional[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchone()

    # ── L0 ──

    def insert_message(
        self, message_id: str, session_id: str, role: str, content: str
    ) -> None:
        now = time.time()
        ts = int(now * 1000)
        self._execute(
            "INSERT OR IGNORE INTO l0_messages "
            "(message_id, session_id, role, content, ts, recorded_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                message_id,
                session_id,
                role,
                content,
                ts,
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
            ),
        )

    def get_unextracted_messages(self, session_id: str, limit: int) -> List[Dict[str, Any]]:
        rows = self._query(
            "SELECT id, message_id, role, content, ts FROM l0_messages "
            "WHERE session_id = ? AND l1_extracted = 0 ORDER BY ts ASC LIMIT ?",
            (session_id, limit),
        )
        return [dict(r) for r in rows]

    def mark_messages_extracted(self, message_ids: List[int]) -> None:
        if not message_ids:
            return
        placeholders = ",".join("?" for _ in message_ids)
        self._execute(
            f"UPDATE l0_messages SET l1_extracted = 1 WHERE id IN ({placeholders})",
            tuple(message_ids),
        )

    def search_conversations_fts(
        self, fts_query: str, limit: int, session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        sql = (
            "SELECT l0.id, l0.message_id, l0.session_id, l0.role, l0.content, l0.ts, "
            "bm25(l0_messages_fts) AS rank FROM l0_messages_fts "
            "JOIN l0_messages l0 ON l0_messages_fts.rowid = l0.id "
            "WHERE l0_messages_fts MATCH ?"
        )
        params: tuple = (fts_query,)
        if session_id:
            sql += " AND l0.session_id = ?"
            params = (fts_query, session_id)
        sql += " ORDER BY rank ASC LIMIT ?"
        params = params + (limit,)
        return [dict(r) for r in self._query(sql, params)]

    # ── L1 ──

    def insert_memory(
        self,
        record_id: str,
        content: str,
        mem_type: str,
        priority: int,
        scene_name: str,
        session_id: str,
        source_msg_ids: List[int],
        metadata: Dict[str, Any],
    ) -> None:
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        self._execute(
            "INSERT OR IGNORE INTO l1_memories "
            "(record_id, content, type, priority, scene_name, session_id, "
            " source_msg_ids, metadata, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record_id,
                content,
                mem_type,
                priority,
                scene_name,
                session_id,
                json.dumps(source_msg_ids),
                json.dumps(metadata),
                now,
                now,
            ),
        )

    def get_memory_by_record_id(self, record_id: str) -> Optional[Dict[str, Any]]:
        row = self._query_one(
            "SELECT * FROM l1_memories WHERE record_id = ?", (record_id,)
        )
        return dict(row) if row else None

    def get_memory_by_content(self, content: str) -> Optional[Dict[str, Any]]:
        row = self._query_one(
            "SELECT * FROM l1_memories WHERE content = ?", (content,)
        )
        return dict(row) if row else None

    def delete_memory(self, record_id: str) -> bool:
        cur = self._execute("DELETE FROM l1_memories WHERE record_id = ?", (record_id,))
        return cur.rowcount > 0

    def delete_memory_by_id(self, memory_id: int) -> bool:
        cur = self._execute("DELETE FROM l1_memories WHERE id = ?", (memory_id,))
        return cur.rowcount > 0

    def wipe_memories(self) -> int:
        row = self._query_one("SELECT COUNT(*) AS c FROM l1_memories")
        count = int(row["c"]) if row else 0
        self._execute("DELETE FROM l1_memories")
        return count

    def wipe_all(self) -> Dict[str, int]:
        counts = {}
        for table in ("l0_messages", "l1_memories", "scenes"):
            row = self._query_one(f"SELECT COUNT(*) AS c FROM {table}")
            counts[table] = int(row["c"]) if row else 0
            self._execute(f"DELETE FROM {table}")
        self._execute("DELETE FROM pipeline_state")
        try:
            self._conn.execute("VACUUM")
            self._conn.commit()
        except Exception:
            pass
        return counts

    def list_memories(
        self,
        limit: int = 50,
        offset: int = 0,
        since_id: int = 0,
    ) -> List[Dict[str, Any]]:
        rows = self._query(
            "SELECT * FROM l1_memories WHERE id > ? ORDER BY id ASC LIMIT ? OFFSET ?",
            (since_id, limit, offset),
        )
        return [dict(r) for r in rows]

    def count_memories(self) -> int:
        row = self._query_one("SELECT COUNT(*) AS c FROM l1_memories")
        return int(row["c"]) if row else 0

    def search_memories_fts(self, fts_query: str, limit: int) -> List[Dict[str, Any]]:
        rows = self._query(
            "SELECT m.*, bm25(l1_memories_fts) AS rank FROM l1_memories_fts "
            "JOIN l1_memories m ON l1_memories_fts.rowid = m.id "
            "WHERE l1_memories_fts MATCH ? ORDER BY rank ASC LIMIT ?",
            (fts_query, limit),
        )
        return [dict(r) for r in rows]

    def search_memories_by_scene(self, scene_name: str, limit: int) -> List[Dict[str, Any]]:
        rows = self._query(
            "SELECT * FROM l1_memories WHERE scene_name = ? ORDER BY priority DESC LIMIT ?",
            (scene_name, limit),
        )
        return [dict(r) for r in rows]

    def distinct_scene_names(self) -> List[str]:
        rows = self._query(
            "SELECT DISTINCT scene_name FROM l1_memories "
            "WHERE scene_name != '' ORDER BY scene_name"
        )
        return [r["scene_name"] for r in rows]

    def latest_memory_id(self) -> int:
        row = self._query_one("SELECT MAX(id) AS max_id FROM l1_memories")
        return int(row["max_id"]) if row and row["max_id"] is not None else 0

    def last_scene_name(self) -> str:
        row = self._query_one(
            "SELECT scene_name FROM l1_memories WHERE scene_name != '' "
            "ORDER BY id DESC LIMIT 1"
        )
        return row["scene_name"] if row else ""

    # ── Scenes (L2) ──

    def upsert_scene(
        self,
        name: str,
        filename: str,
        summary: str,
        heat: int,
        created_at: Optional[str] = None,
    ) -> None:
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        existing = self._query_one("SELECT id FROM scenes WHERE filename = ?", (filename,))
        if existing:
            self._execute(
                "UPDATE scenes SET name = ?, summary = ?, heat = ?, updated_at = ? "
                "WHERE filename = ?",
                (name, summary, heat, now, filename),
            )
        else:
            self._execute(
                "INSERT INTO scenes (name, filename, summary, heat, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (name, filename, summary, heat, created_at or now, now),
            )

    def delete_scene(self, filename: str) -> None:
        self._execute("DELETE FROM scenes WHERE filename = ?", (filename,))

    def list_scenes(self) -> List[Dict[str, Any]]:
        rows = self._query(
            "SELECT * FROM scenes ORDER BY heat DESC, updated_at DESC"
        )
        return [dict(r) for r in rows]

    # ── Pipeline state ──

    def get_session_state(self, session_id: str) -> Dict[str, Any]:
        row = self._query_one(
            "SELECT * FROM pipeline_state WHERE session_id = ?", (session_id,)
        )
        return dict(row) if row else {
            "session_id": session_id,
            "conversation_count": 0,
            "warmup_threshold": 1,
            "last_l1_at": 0,
            "last_l2_at": 0,
            "last_scene_name": "",
            "memory_count": 0,
            "l2_pending": 0,
            "l3_pending": 0,
            "last_persona_count": 0,
            "last_l2_cursor": 0,
        }

    def update_session_state(self, session_id: str, **fields: Any) -> None:
        allowed = {
            "conversation_count",
            "warmup_threshold",
            "last_l1_at",
            "last_l2_at",
            "last_scene_name",
            "memory_count",
            "l2_pending",
            "l3_pending",
            "last_persona_count",
            "last_l2_cursor",
        }
        if not fields:
            return
        sets = []
        values: List[Any] = []
        for key, value in fields.items():
            if key not in allowed:
                continue
            sets.append(f"{key} = ?")
            values.append(value)
        if not sets:
            return
        self._execute(
            f"INSERT INTO pipeline_state (session_id) VALUES (?) "
            f"ON CONFLICT(session_id) DO UPDATE SET {', '.join(sets)}",
            (session_id, *values),
        )

    def list_active_sessions(self) -> List[str]:
        rows = self._query(
            "SELECT session_id FROM pipeline_state WHERE conversation_count > 0"
        )
        return [r["session_id"] for r in rows]
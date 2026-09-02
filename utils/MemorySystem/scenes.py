import json
import os
import re
import time
from typing import Any, Dict, List

from .config import MemoryConfig
from .llm import MemoryLlm
from .prompts import SCENE_CONSOLIDATION_SYSTEM_PROMPT, SCENE_USER_PROMPT_TEMPLATE
from .storage import MemoryStorage

META_RE = re.compile(
    r"-----META-START-----(.*?)-----META-END-----", re.DOTALL
)
SUMMARY_RE = re.compile(r"summary:\s*(.+)", re.IGNORECASE)
HEAT_RE = re.compile(r"heat:\s*(\d+)", re.IGNORECASE)
CREATED_RE = re.compile(r"created:\s*(.+)", re.IGNORECASE)

SAFE_FILENAME_RE = re.compile(r"[^a-zA-Z0-9\u4e00-\u9fff._-]+")


def normalize_filename(name: str) -> str:
    cleaned = SAFE_FILENAME_RE.sub("-", name).strip("-")
    if not cleaned.endswith(".md"):
        cleaned = f"{cleaned}.md"
    return cleaned


def parse_meta(content: str) -> Dict[str, Any]:
    match = META_RE.search(content)
    if not match:
        return {}
    block = match.group(1)
    meta: Dict[str, Any] = {}
    summary = SUMMARY_RE.search(block)
    heat = HEAT_RE.search(block)
    created = CREATED_RE.search(block)
    if summary:
        meta["summary"] = summary.group(1).strip()
    if heat:
        try:
            meta["heat"] = int(heat.group(1))
        except ValueError:
            pass
    if created:
        meta["created_at"] = created.group(1).strip()
    return meta


def consolidate_scenes(
    storage: MemoryStorage,
    llm: MemoryLlm,
    config: MemoryConfig,
    since_memory_id: int = 0,
) -> Dict[str, Any]:
    os.makedirs(config.scenes_dir, exist_ok=True)

    new_memories = storage.list_memories(limit=60, since_id=since_memory_id)
    if not new_memories:
        return {"processed": 0, "written": 0}

    memories_json = json.dumps(
        [
            {
                "id": m["id"],
                "content": m["content"],
                "type": m["type"],
                "priority": m["priority"],
                "scene_name": m["scene_name"],
                "created_at": m["created_at"],
            }
            for m in new_memories
        ],
        ensure_ascii=False,
        indent=1,
    )

    existing = storage.list_scenes()
    scene_summary_lines = [f"scene count: {len(existing)}"]
    for s in existing:
        scene_summary_lines.append(
            f"- filename={s['filename']} | name={s['name']} | heat={s['heat']} "
            f"| summary={s['summary']}"
        )
    scene_summary = "\n".join(scene_summary_lines) or "None"

    file_list = "\n".join(f"- `{s['filename']}`" for s in existing) or "(none)"

    user_prompt = SCENE_USER_PROMPT_TEMPLATE.format(
        memories_json=memories_json,
        scene_summary=scene_summary,
        current_time=time.strftime("%Y-%m-%d %H:%M:%S"),
        file_list=file_list,
    )
    result = llm.chat_json(SCENE_CONSOLIDATION_SYSTEM_PROMPT, user_prompt)
    if not isinstance(result, dict):
        return {"processed": len(new_memories), "written": 0}

    written = 0
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    seen_files = {s["filename"] for s in existing}

    for scene in result.get("scenes", []) or []:
        if not isinstance(scene, dict):
            continue
        action = str(scene.get("action") or "update").strip()
        raw_filename = str(scene.get("filename") or "").strip().strip("`")
        content = str(scene.get("content") or "").strip()
        if not raw_filename or not content:
            continue

        filename = normalize_filename(raw_filename)
        if action == "create" and filename in seen_files:
            action = "update"

        meta = parse_meta(content)
        heat = meta.get("heat", 1)
        if action == "update":
            old = next((s for s in existing if s["filename"] == filename), None)
            if old and not meta.get("heat"):
                heat = old["heat"] + 1

        file_path = os.path.join(config.scenes_dir, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        name = raw_filename.removesuffix(".md").replace("-", " ").strip() or filename
        storage.upsert_scene(
            name=name[:200],
            filename=filename,
            summary=meta.get("summary", "")[:500],
            heat=heat,
            created_at=meta.get("created_at", now),
        )
        seen_files.add(filename)
        written += 1

    return {"processed": len(new_memories), "written": written}
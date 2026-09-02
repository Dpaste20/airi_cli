import os
import re
import time
from typing import Any, Dict, Optional

from .config import MemoryConfig
from .llm import MemoryLlm
from .prompts import PERSONA_GENERATION_SYSTEM_PROMPT, format_persona_user_prompt
from .storage import MemoryStorage


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    fenced = re.search(r"```(?:markdown|md)?\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    return text


def generate_persona(
    storage: MemoryStorage,
    llm: MemoryLlm,
    config: MemoryConfig,
    max_chars: int = 2500,
) -> Dict[str, Any]:
    os.makedirs(config.data_dir, exist_ok=True)

    scenes = storage.list_scenes()
    scene_parts = []
    for s in scenes[:6]:
        file_path = os.path.join(config.scenes_dir, s["filename"])
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except OSError:
            content = ""
        scene_parts.append(f"### Scene: {s['name']} (heat {s['heat']})\n{content[:3000]}")
    scenes_summary = "\n\n".join(scene_parts) or "None"

    existing_persona = ""
    if os.path.exists(config.persona_path):
        try:
            with open(config.persona_path, "r", encoding="utf-8") as f:
                existing_persona = f.read()
        except OSError:
            pass

    user_prompt = format_persona_user_prompt(
        existing_persona=existing_persona,
        scenes_summary=scenes_summary,
        current_time=time.strftime("%Y-%m-%d %H:%M:%S"),
    )

    content = llm.chat(PERSONA_GENERATION_SYSTEM_PROMPT, user_prompt)
    if not content.strip():
        return {"generated": False, "chars": 0}

    persona = _strip_code_fences(content)
    if len(persona) > max_chars:
        persona = persona[:max_chars]
    if len(persona) < 50:
        return {"generated": False, "chars": 0}

    with open(config.persona_path, "w", encoding="utf-8") as f:
        f.write(persona)

    return {"generated": True, "chars": len(persona)}
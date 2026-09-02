import json
import re
from typing import Any, Dict, List, Optional

from ollama import Client as OllamaClient

from .config import MemoryConfig


class MemoryLlm:
    def __init__(self, config: MemoryConfig):
        self.config = config
        self._client = OllamaClient(host="http://localhost:11434")

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self._client.chat(
                model=self.config.llm.id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                options={"temperature": self.config.llm.temperature},
            )
            return response.message.content
        except Exception as e:
            print(f"Warning: memory LLM call failed: {e}")
            return ""

    def chat_json(self, system_prompt: str, user_prompt: str) -> Optional[Any]:
        content = self.chat(system_prompt, user_prompt)
        if not content:
            return None
        try:
            return json.loads(self._strip_json_fences(content))
        except json.JSONDecodeError as e:
            print(f"Warning: LLM returned invalid JSON: {e}")
            return None

    @staticmethod
    def _strip_json_fences(text: str) -> str:
        fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if fenced:
            return fenced.group(1)
        start = text.find("{")
        end = text.rfind("}")
        array_start = text.find("[")
        array_end = text.rfind("]")
        candidates = []
        if start != -1 and end != -1:
            candidates.append((end - start, text[start : end + 1]))
        if array_start != -1 and array_end != -1:
            candidates.append((array_end - array_start, text[array_start : array_end + 1]))
        if not candidates:
            return text
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
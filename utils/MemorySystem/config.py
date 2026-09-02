import tomllib
from dataclasses import dataclass, field
from typing import Any, Dict

CONFIG_PATH = "config.toml"

DEFAULTS: Dict[str, Any] = {
    "memory": {
        "enabled": True,
        "db_path": "tmp/memory.db",
        "data_dir": "UserProfile/memory",
        "llm": {"id": "gpt-oss:20b-cloud", "temperature": 0.2},
        "embedding": {"id": "qwen3-embedding:0.6b", "dimensions": 1024},
        "pipeline": {
            "every_n_conversations": 5,
            "enable_warmup": True,
            "l1_idle_timeout_seconds": 600,
            "l2_delay_after_l1_seconds": 10,
            "l2_min_interval_seconds": 900,
            "l2_max_interval_seconds": 3600,
            "l1_batch_limit": 40,
        },
        "recall": {
            "max_results": 5,
            "score_threshold": 0.3,
            "max_total_recall_chars": 2000,
            "rrf_k": 60,
            "tool_max_results": 10,
        },
        "persona": {
            "trigger_every_n_memories": 20,
            "max_scenes": 15,
            "max_persona_chars": 2500,
        },
        "qdrant_url": "http://localhost:6333",
        "l0_collection": "airi_mem_l0",
        "l1_collection": "airi_mem_l1",
    }
}


@dataclass
class MemoryLlmConfig:
    id: str = "gpt-oss:20b-cloud"
    temperature: float = 0.2


@dataclass
class MemoryEmbeddingConfig:
    id: str = "embeddinggemma:latest"
    dimensions: int = 768


@dataclass
class MemoryPipelineConfig:
    every_n_conversations: int = 5
    enable_warmup: bool = True
    l1_idle_timeout_seconds: int = 600
    l2_delay_after_l1_seconds: int = 10
    l2_min_interval_seconds: int = 900
    l2_max_interval_seconds: int = 3600
    l1_batch_limit: int = 40


@dataclass
class MemoryRecallConfig:
    max_results: int = 5
    score_threshold: float = 0.3
    max_total_recall_chars: int = 2000
    rrf_k: int = 60
    tool_max_results: int = 10


@dataclass
class MemoryPersonaConfig:
    trigger_every_n_memories: int = 20
    max_scenes: int = 15
    max_persona_chars: int = 2500


@dataclass
class MemoryConfig:
    enabled: bool = True
    db_path: str = "tmp/memory.db"
    data_dir: str = "UserProfile/memory"
    llm: MemoryLlmConfig = field(default_factory=MemoryLlmConfig)
    embedding: MemoryEmbeddingConfig = field(default_factory=MemoryEmbeddingConfig)
    pipeline: MemoryPipelineConfig = field(default_factory=MemoryPipelineConfig)
    recall: MemoryRecallConfig = field(default_factory=MemoryRecallConfig)
    persona: MemoryPersonaConfig = field(default_factory=MemoryPersonaConfig)
    qdrant_url: str = "http://localhost:6333"
    l0_collection: str = "airi_mem_l0"
    l1_collection: str = "airi_mem_l1"

    @property
    def scenes_dir(self) -> str:
        return f"{self.data_dir}/scenes"

    @property
    def persona_path(self) -> str:
        return f"{self.data_dir}/persona.md"


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_memory_config(config_path: str = CONFIG_PATH) -> MemoryConfig:
    raw: Dict[str, Any] = {}
    try:
        with open(config_path, "rb") as file:
            file_config = tomllib.load(file)
            if isinstance(file_config.get("memory"), dict):
                raw = file_config["memory"]
    except (OSError, tomllib.TOMLDecodeError) as e:
        print(f"Warning: Could not load memory config ({e}). Using defaults.")

    merged = _deep_merge(DEFAULTS["memory"], raw or {})

    return MemoryConfig(
        enabled=merged.get("enabled", True),
        db_path=merged.get("db_path", "tmp/memory.db"),
        data_dir=merged.get("data_dir", "UserProfile/memory"),
        llm=MemoryLlmConfig(**(merged.get("llm") or {})),
        embedding=MemoryEmbeddingConfig(**(merged.get("embedding") or {})),
        pipeline=MemoryPipelineConfig(**(merged.get("pipeline") or {})),
        recall=MemoryRecallConfig(**(merged.get("recall") or {})),
        persona=MemoryPersonaConfig(**(merged.get("persona") or {})),
        qdrant_url=merged.get("qdrant_url", "http://localhost:6333"),
        l0_collection=merged.get("l0_collection", "airi_mem_l0"),
        l1_collection=merged.get("l1_collection", "airi_mem_l1"),
    )
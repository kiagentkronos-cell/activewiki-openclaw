#!/usr/bin/env python3
"""ActiveWiki configuration loader.

Reads `activewiki.json` and provides typed access to all settings.
Search order:
  1. Explicit path (--config argument or ACTIVEWIKI_CONFIG env var)
  2. ./activewiki.json relative to the script's directory
  3. ./activewiki.json relative to the current working directory

All scripts in this project should import from this module instead of
hardcoding paths, model names, or API endpoints.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


def _find_config() -> Path:
    """Locate the activewiki.json config file using the search order."""
    # 1. Explicit env var
    explicit = os.environ.get("ACTIVEWIKI_CONFIG")
    if explicit:
        p = Path(explicit).resolve()
        if p.exists():
            return p

    # 2. Next to the script itself (scripts/../activewiki.json)
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir.parent / "activewiki.json",
        script_dir / "activewiki.json",
        Path.cwd() / "activewiki.json",
    ]
    for c in candidates:
        if c.exists():
            return c

    # 3. Return the most likely default so the error message is useful
    return script_dir.parent / "activewiki.json"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and return the activewiki.json configuration as a dict.

    Args:
        path: Explicit config path. If None, uses search order.

    Raises:
        FileNotFoundError: If no config file is found.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    if path is not None:
        cfg_path = Path(path).resolve()
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"ActiveWiki config not found: {cfg_path}\n"
                f"Set ACTIVEWIKI_CONFIG or place activewiki.json next to your scripts."
            )
    else:
        cfg_path = _find_config()
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"ActiveWiki config not found: {cfg_path}\n"
                f"Searched:\n"
                f"  - ACTIVEWIKI_CONFIG env var\n"
                f"  - Next to scripts/ directory\n"
                f"  - Current working directory\n\n"
                f"Copy activewiki.example.json to activewiki.json and adjust paths."
            )

    with open(cfg_path, encoding="utf-8") as f:
        return json.load(f)


def get(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Access nested config values using dot notation.

    Example:
        get(config, "embeddings.ollama_url")  # → config["embeddings"]["ollama_url"]
        get(config, "ocr.venv_path", "/usr/bin")  # fallback if missing
    """
    keys = dotted_key.split(".")
    val: Any = config
    for k in keys:
        if isinstance(val, dict):
            val = val.get(k)
        else:
            return default
        if val is None:
            return default
    return val


# ── Convenience helpers ──────────────────────────────────────────────────────

def wikis_root(config: dict[str, Any]) -> Path:
    """Return the resolved wikis_root path."""
    return Path(get(config, "wikis_root", "")).resolve()


def scopes(config: dict[str, Any]) -> tuple[str, ...]:
    """Return enabled scopes as a tuple."""
    enabled = get(config, "scopes.enabled", ["private", "family", "public"])
    return tuple(enabled)


def scopes_config_path(config: dict[str, Any]) -> Path:
    """Return the resolved path to scopes.json."""
    raw = get(config, "scopes.scopes_config", "")
    if raw:
        return Path(raw).resolve()
    # Fallback: next to wikis_root
    return wikis_root(config) / "config" / "scopes.json"


def ollama_url(config: dict[str, Any], section: str = "embeddings") -> str:
    """Return Ollama base URL from the specified section."""
    return get(config, f"{section}.ollama_url", "http://localhost:11434")


def embed_model(config: dict[str, Any]) -> str:
    return get(config, "embeddings.model", "bge-m3")


def embed_dims(config: dict[str, Any]) -> int:
    return get(config, "embeddings.embed_dim", 1024)


def db_path(config: dict[str, Any]) -> Path:
    """Return the resolved SQLite index path."""
    rel = get(config, "embeddings.index_path", "vectordb/index.sqlite")
    return wikis_root(config) / rel


def docling_venv(config: dict[str, Any]) -> str:
    """Return the Docling venv path (or empty string for global install)."""
    return get(config, "ocr.venv_path", "")


def llm_backend(config: dict[str, Any]) -> str:
    return get(config, "llm.backend", "ollama")


def llm_model(config: dict[str, Any]) -> str:
    return get(config, "llm.model", "qwen3.6-fp8")


def llm_url(config: dict[str, Any]) -> str:
    """Return LLM API URL (OpenAI-compatible endpoint)."""
    return get(config, "llm.url", "http://127.0.0.1:8000/v1").rstrip("/")


def llm_temperature(config: dict[str, Any]) -> float:
    return float(get(config, "llm.temperature", 0.5))


def llm_max_tokens(config: dict[str, Any]) -> int:
    return int(get(config, "llm.max_tokens", 4096))


def graph_incremental(config: dict[str, Any]) -> bool:
    return bool(get(config, "graph.build_incremental", True))


def graph_communities_enabled(config: dict[str, Any]) -> bool:
    return bool(get(config, "graph.communities_enabled", True))


def graph_communities_threshold(config: dict[str, Any]) -> int:
    return int(get(config, "graph.communities_incremental_threshold", 5))


def ingest_deadline(config: dict[str, Any]) -> str:
    return get(config, "ingest.deadline", "03:00")


def ingest_timezone(config: dict[str, Any]) -> str:
    return get(config, "ingest.timezone", "Europe/Berlin")


def distill_rollup_all(config: dict[str, Any]) -> bool:
    return bool(get(config, "distill.rollup_all", True))

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
    model = get(config, "llm.model")
    if not model:
        raise SystemExit(
            "FATAL: llm.model fehlt in activewiki.json — kein Fallback-Modell mehr erlaubt. "
            "Bitte llm.model setzen (z.B. 'qwen3.8-fp8-fast')."
        )
    return model


def _coerce_bool(raw_val: Any, key: str, default: bool = False) -> bool:
    """Config value → bool, type-robust without crashing (pattern: _coerce_max_tokens).

    bool → returned directly; int 0/1 → bool; strings 'true'/'false'/'1'/'0'
    (case-insensitive, leading/trailing whitespace ignored) → matching bool.
    Anything else (None, list, garbage string) → default + warning on stderr:
    a broken switch must never crash the nightly run — it may only land
    silently on the safe default (thinking off).
    """
    if isinstance(raw_val, bool):
        return raw_val
    if isinstance(raw_val, int) and not isinstance(raw_val, bool):
        if raw_val in (0, 1):
            return bool(raw_val)
    if isinstance(raw_val, str):
        v = raw_val.strip().lower()
        if v in ("true", "1", "yes", "on"):
            return True
        if v in ("false", "0", "no", "off"):
            return False
    print(f"[config] WARNING {key}: invalid bool value {raw_val!r} — "
          f"default {default} applies.", file=sys.stderr)
    return default


def resolve_role_model(config: dict[str, Any], role_key: str) -> str:
    """Model for one pipeline role — fallback chain: role key → llm.model.

    role_key is the key inside the llm object (e.g. 'build_model'). Missing,
    empty or non-string value → silently falls back to llm.model (the old
    behaviour). llm.model itself stays mandatory (FATAL as in llm_model()):
    no role key + no llm.model = configuration error, no model fallback in
    code (config requirement since 2026-08-17).
    """
    role_model = get(config, f"llm.{role_key}")
    if isinstance(role_model, str) and role_model.strip():
        return role_model.strip()
    if role_model is not None and not isinstance(role_model, str):
        print(f"[config] WARNING llm.{role_key}: not a string ({role_model!r}) — "
              f"falling back to llm.model.", file=sys.stderr)
    return llm_model(config)


def build_model(config: dict[str, Any]) -> str:
    """Model for the build role (distill.py): llm.build_model → llm.model."""
    return resolve_role_model(config, "build_model")


def build_thinking(config: dict[str, Any]) -> bool:
    """Thinking switch for the build role: llm.build_thinking, default False
    (behaviour with the key missing stays exactly as before)."""
    raw = get(config, "llm.build_thinking")
    return _coerce_bool(raw, "llm.build_thinking", False) if raw is not None else False


def qc_model(config: dict[str, Any]) -> str:
    """Model for the quality-check role (check.py).

    Naming decision (2026-10-18): the QC key lives in the existing top-level
    quality_check section (analogous to timeout_seconds/max_tokens — role
    parameters belong together, model analogy like embeddings.model), NOT in
    the llm object like build_model (build is the default llm role, its key
    extends llm directly).
    Fallback: quality_check.model → llm.model (old behaviour).
    """
    role_model = get(config, "quality_check.model")
    if isinstance(role_model, str) and role_model.strip():
        return role_model.strip()
    if role_model is not None and not isinstance(role_model, str):
        print(f"[config] WARNING quality_check.model: not a string "
              f"({role_model!r}) — falling back to llm.model.", file=sys.stderr)
    return llm_model(config)


def qc_thinking(config: dict[str, Any]) -> bool:
    """Thinking switch for the quality check: quality_check.thinking, default
    False (old behaviour thinking=false when the key is missing)."""
    raw = get(config, "quality_check.thinking")
    return _coerce_bool(raw, "quality_check.thinking", False) if raw is not None else False


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


def graph_html_mode(config: dict[str, Any]) -> str:
    """HTML-Viewer-Modus: "2d" | "3d" | "both" (Default "3d")."""
    mode = str(get(config, "graph.html_mode", "3d") or "3d").strip().lower()
    return mode if mode in ("2d", "3d", "both") else "3d"


def ingest_deadline(config: dict[str, Any]) -> str:
    return get(config, "ingest.deadline", "03:00")


def ingest_timezone(config: dict[str, Any]) -> str:
    return get(config, "ingest.timezone", "Europe/Berlin")


def distill_rollup_all(config: dict[str, Any]) -> bool:
    return bool(get(config, "distill.rollup_all", True))

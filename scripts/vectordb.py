#!/usr/bin/env python3
"""ActiveWiki: scope-aware vector index + knowledge graph over wiki/ + sources/.

Storage: vectordb/index.sqlite with chunks table (incl. scope column) and
float32 embedding BLOBs. Search: brute-force cosine similarity in numpy.
Access: --session-key resolves allowed scopes via config/scopes.json.

Configuration via activewiki.json (see activewiki.example.json).
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import random
import re
import sqlite3
import subprocess
import sys
import time
import traceback
import urllib.request
from datetime import timezone
from pathlib import Path

# ── Config loading ───────────────────────────────────────────────────────────
# Must be before any path-dependent code.
# We accept --config as the very first arg (before subcommand parsing).
_config_path: str | None = None
for _arg in sys.argv[1:]:
    if _arg == "--config" and sys.argv[1:].index(_arg) + 1 < len(sys.argv[1:]):
        _config_path = sys.argv[1:].index(_arg) + 2
        if _config_path <= len(sys.argv):
            _config_path = sys.argv[_config_path]
        break
    elif _arg.startswith("--config="):
        _config_path = _arg.split("=", 1)[1]
        break

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    load_config, get,
    wikis_root, scopes, scopes_config_path,
    ollama_url, embed_model, embed_dims, db_path,
    llm_model, llm_url, llm_temperature, llm_max_tokens,
    graph_incremental, graph_communities_enabled, graph_communities_threshold,
)

_CONFIG = load_config(_config_path)
_WIKIS_ROOT = wikis_root(_CONFIG)
_SCOPES = scopes(_CONFIG)


class _LogStream:
    """Write to both terminal and daily ingest log."""
    def __init__(self, original: IO[str], log_path: Path) -> None:
        self._original = original
        self._log_path = log_path
        self._log_fh = open(log_path, "a", encoding="utf-8", errors="replace")

    def write(self, text: str) -> int:
        self._original.write(text)
        self._log_fh.write(text)
        return len(text)

    def flush(self) -> None:
        self._original.flush()
        self._log_fh.flush()

    def close(self) -> None:
        self._log_fh.close()
from typing import Any, Dict, IO, List, Optional, Tuple

import numpy as np
import yaml

# ── igraph (Phase 2: community detection) - graceful degradation ──
try:
    import igraph as ig
    IGRAPH_AVAILABLE = True
except ImportError:
    ig = None  # type: ignore[assignment]
    IGRAPH_AVAILABLE = False

# Tolerate filenames/strings carrying PEP 383 surrogate escapes so prints
# over old or externally-supplied data do not crash the pipeline.
sys.stdout.reconfigure(errors="backslashreplace")
sys.stderr.reconfigure(errors="backslashreplace")

# Centralised daily log (same file as distill.py / run_inbox.sh)
_LOG_DIR = _WIKIS_ROOT / "logs"
_LOG_DIR.mkdir(exist_ok=True)
_LOG_DATE = datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=2))).strftime("%Y-%m-%d")
_LOG_FILE = _LOG_DIR / f"ingest-{_LOG_DATE}.log"
# Only redirect to log when explicitly requested (e.g. by run_inbox.sh via WIKIS_LOG_REDIRECT=1).
# Default: no redirect — prevents JSON/stats/search output from polluting the ingest log.
if os.environ.get("WIKIS_LOG_REDIRECT"):
    _orig_stdout = sys.stdout
    _orig_stderr = sys.stderr
    try:
        sys.stdout = _LogStream(_orig_stdout, _LOG_FILE)
        sys.stderr = _LogStream(_orig_stderr, _LOG_FILE)
    except Exception:
        pass

SOURCES = _WIKIS_ROOT / "sources"
WIKI = _WIKIS_ROOT / "wiki"
VECTORDB = _WIKIS_ROOT / "vectordb"
DB_PATH = db_path(_CONFIG)
SCOPES_CONFIG = scopes_config_path(_CONFIG)
SCOPES = _SCOPES

# ── Vector Embedding Config (from activewiki.json) ──
EMBED_MODEL = os.environ.get("ACTIVEWIKI_EMBED_MODEL", embed_model(_CONFIG))
OLLAMA_URL = os.environ.get("OLLAMA_URL", ollama_url(_CONFIG, "embeddings"))
EMBED_DIMS = int(os.environ.get("ACTIVEWIKI_EMBED_DIMS", str(embed_dims(_CONFIG))))
MAX_CHUNK_CHARS = 1500
MIN_CHUNK_CHARS = 80

# ── Semantic Chunking Config ──
SEMANTIC_CHUNKING = os.environ.get("ACTIVEWIKI_SEMANTIC_CHUNKING", "1") == "1"
SEMANTIC_THRESHOLD = float(os.environ.get("ACTIVEWIKI_SEMANTIC_THRESHOLD", "0.70"))

# ── Graph Extraction Config (Two-Tier, from activewiki.json) ──
GRAPH_MODEL = os.environ.get("ACTIVEWIKI_GRAPH_MODEL", llm_model(_CONFIG))   # Entity extraction
SUMMARY_MODEL = os.environ.get("ACTIVEWIKI_SUMMARY_MODEL", llm_model(_CONFIG))  # Community summaries
VLLM_URL = os.environ.get("VLLM_URL", llm_url(_CONFIG))                       # OpenAI-compatible endpoint
RETRY_MAX = 3
RETRY_BASE = 1.0
RATE_LIMIT_S = 0.5
FUZZY_THRESHOLD = 85

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n?(.*)$", re.DOTALL)
H2_SPLIT_RE = re.compile(r"(?=^##\s)", re.MULTILINE)
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)

# ── Entity / Relationship type lists ──
ENTITY_TYPES = [
    "PERSON", "ORGANIZATION", "LOCATION", "DOCUMENT", "PROPERTY",
    "FACILITY", "CONCEPT", "DATE", "MONEY", "EVENT",
]

# ── Cross-type fuzzy threshold (stricter than same-type FUZZY_THRESHOLD) ──
CROSS_TYPE_FUZZY_THRESHOLD = 90

# ── Priority types for registry prompt injection (max 10 each) ──
PRIORITY_ENTITY_TYPES = ["PERSON", "PROPERTY", "ORGANIZATION", "LOCATION"]
MAX_ENTRIES_PER_PRIORITY_TYPE = 10
MAX_REGISTRY_PROMPT_ENTRIES = 30


# ═══════════════════════════════════════════════════════════════════════
#  Entity Registry - Persistent Known-Entity Store
# ═══════════════════════════════════════════════════════════════════════


class EntityRegistry:
    """Persistente Known-Entity Registry.

    Provides a fast lookup layer on top of the SQLite entities table.
    On first start, seeds itself from all existing DB entities so that
    subsequent incremental builds immediately benefit from consolidation.

    Priority chain in resolve_entity():
      1. Registry lookup (highest priority)
      2. ID exact match
      3. Label NOCASE exact match
      4. Fuzzy match within same entity_type (threshold 85%)
      5. Cross-type fuzzy match (threshold 90%)
      6. Insert as new entity
    """

    FILE = VECTORDB / "entity_registry.json"

    def __init__(self) -> None:
        # db_id → {id, label, type, description}
        self.entities: Dict[str, Dict[str, Any]] = {}

    # ── Loading ──────────────────────────────────────────────────────

    def load_or_seed(self, conn: sqlite3.Connection) -> None:
        """Load registry from JSON or seed from DB on first run."""
        if self.FILE.exists():
            self._load_json()
        else:
            self._seed_from_db(conn)

    def _load_json(self) -> None:
        """Load registry from persisted JSON file."""
        try:
            data = json.loads(self.FILE.read_text(encoding="utf-8"))
            self.entities = data.get("entities", {})
            log("graph-reg", f"Loaded {len(self.entities)} entities from registry")
        except (json.JSONDecodeError, OSError) as e:
            log("warn", f"Failed to load registry JSON: {e} - seeding from DB", stderr=True)
            # We don't have a conn here, so we'll start empty
            self.entities = {}

    def _seed_from_db(self, conn: sqlite3.Connection) -> None:
        """Seed registry from all existing entities in the database.

        Called only on first run (when no registry JSON exists).
        Duplicate labels are merged - first ID wins.
        """
        rows = conn.execute(
            "SELECT id, label, entity_type, COALESCE(description, '') FROM entities "
            "ORDER BY label COLLATE NOCASE"
        ).fetchall()

        for db_id, label, etype, desc in rows:
            canonical = self.lookup(label)
            if not canonical:
                self.entities[db_id] = {
                    "id": db_id,
                    "label": label,
                    "type": etype,
                    "description": desc[:200],
                }

        log("graph-reg", f"Seeded {len(self.entities)} entities from DB")

    # ── Persistence ──────────────────────────────────────────────────

    def save(self) -> None:
        """Atomically persist registry to JSON (.tmp → os.rename)."""
        VECTORDB.mkdir(exist_ok=True)
        tmp_path = self.FILE.with_suffix(".tmp")
        data = {"entities": self.entities, "count": len(self.entities)}
        tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.rename(str(tmp_path), str(self.FILE))
        log("graph-reg", f"Saved registry with {len(self.entities)} entities")

    # ── Lookup ───────────────────────────────────────────────────────

    def lookup(self, label: str) -> str | None:
        """Case-insensitive exact lookup by label. Returns DB-ID or None."""
        label_lower = label.lower()
        for entry in self.entities.values():
            if entry["label"].lower() == label_lower:
                return entry["id"]
        return None

    # ── Registration ─────────────────────────────────────────────────

    def add_if_new(self, entity: Dict[str, Any]) -> str:
        """Add entity to registry only if not already known.

        Checks by label (case-insensitive). Returns the DB-ID
        (existing or newly added).
        """
        existing_id = self.lookup(entity.get("label", ""))
        if existing_id:
            return existing_id

        db_id = entity["id"]
        if db_id not in self.entities:
            self.entities[db_id] = {
                "id": db_id,
                "label": entity.get("label", db_id),
                "type": entity.get("type", "CONCEPT"),
                "description": (entity.get("description", "") or "")[:200],
            }
        return db_id

    # ── Prompt Integration ───────────────────────────────────────────

    def to_prompt_section(self, max_entries: int = MAX_REGISTRY_PROMPT_ENTRIES) -> str:
        """Generate a Known-Entities section for the LLM system prompt.

        Selection: max 10 per priority type (PERSON/PROPERTY/ORG/LOCATION),
        recent-first (by insertion order). Remaining slots filled with
        other types.
        """
        entries: List[str] = []
        used_ids: set[str] = set()

        # Priority types first
        for ptype in PRIORITY_ENTITY_TYPES:
            count = 0
            for entry in self.entities.values():
                if count >= MAX_ENTRIES_PER_PRIORITY_TYPE:
                    break
                if entry["id"] in used_ids:
                    continue
                if entry["type"] == ptype:
                    entries.append(f"- `{entry['id']}` | {entry['label']} ({entry['type']})")
                    used_ids.add(entry["id"])
                    count += 1
                if len(entries) >= max_entries:
                    break

        # Fill remaining with other types
        if len(entries) < max_entries:
            for entry in self.entities.values():
                if entry["id"] in used_ids:
                    continue
                entries.append(f"- `{entry['id']}` | {entry['label']} ({entry['type']})")
                used_ids.add(entry["id"])
                if len(entries) >= max_entries:
                    break

        if not entries:
            return ""

        return (
            "## BEKANNTEN ENTITIES (nutze diese IDs bei Matches!)\n"
            "Diese Entities existieren bereits im Graph. Wenn du eine Entity findest,\n"
            "die einem dieser Einträge entspricht, nutze die vorhandene ID:\n\n"
        ) + "\n".join(entries) + "\n"

RELATION_TYPES = [
    "BESITZT", "VERTRAG_MIT", "BEFINDET_SICH_IN", "BEZOGEN_AUF",
    "TEIL_VON", "FINANZIERT_VON", "VERSICHERT_BEI", "MIETET_BEI",
    "VERMIETET_AN", "ARBEITET_BEI", "KREDIT_BEI",
    "HAT_KOMPONENTE", "HAT_EIGENSCHAFT", "REFERENZIERT",
    "STAMMT_VON", "KOSTET",
]

# ── Timestamped Logging ─────────────────────────────────────────────────────

def log(level: str, message: str, *, stderr: bool = False) -> None:
    """Print a log line with ISO-8601 Berlin timestamp.

    All output goes to stdout (stderr param kept for API compatibility
    but ignored) to avoid stream interleaving in the log file.
    """
    from zoneinfo import ZoneInfo
    ts = datetime.datetime.now(ZoneInfo("Europe/Berlin")).strftime("%Y-%m-%dT%H:%M:%S %Z")
    # Force single-line: collapse newlines to prevent multi-line log entries
    message = message.replace("\n", "\\n").replace("\r", "\\r")
    line = f"[{ts}] [{level}] {message}"
    print(line, flush=True)  # always stdout, always flushed


# ── Entity Extraction System Prompt ──
EXTRACTION_SYSTEM_PROMPT = """\
Du extrahiere Entities und Beziehungen für einen Knowledge Graph aus Immobilien-/Finanzdokumenten.

## PRÄZEDENZ: Präzision > Recall!
Es ist besser, 3 gute Entities zu extrahieren als 20 mittelmäßige. Sei streng.

## ENTITY-TYPEN (erlaubt)
PERSON - konkrete Namen (Vor- + Nachname)
ORGANIZATION - Unternehmen, Behörden, Institutionen mit vollem Namen
LOCATION - konkrete Orte (Orte, Regionen, Länder) mit Bedeutung im Kontext
DOCUMENT - konkrete Dokumente mit Titel/Nummer (Urteile, Gesetze, Verträge)
PROPERTY - konkrete Grundstücke/Immobilien mit Adresse oder Flurnummer
FACILITY - konkrete Gebäude, Anlagen, Infrastruktur
CONCEPT - fachliche Begriffe die ein eigenes Thema darstellen (nicht Generic-Wörter)
DATE - konkrete Daten die Handlungen markieren (Ereignisdaten, Fristen)
MONEY - konkrete Geldbeträge mit Kontext (Miete, Kaufpreis, Darlehen)
EVENT - konkrete Events (Gerichtstermine, Baubeginn, Unterschriften)

## NOISE - NICHT EXTRAHIEREN (kritisch!)
- Frontmatter-Tags/Keywords aus YAML-Metadaten ("nuts-3", "trends", "immobilien") → das sind Metadaten keine Entities
- Generische Begriffe ("Immobilien", "Investitionen", "Markt", "Wirtschaft") → zu allgemein
- Hash-Referenzen ("d41d8cd9f000-Musterdokument") → technisch irrelevant
- Adjektive/Substantive ohne Eigenname ("der Markt", "eine Analyse", "dieser Vertrag")
- Abschnittstiteln als eigene Entities wenn sie keinen konkreten Namen enthalten
- Wiederholungen desselben Entity → dedupliziere sofort
- Platzhalter-Labels wie "Haupt-Entity der Wiki-Seite XYZ"

## RELATION-TYPEN (spezifisch zuerst!)

Immobilien & Finanzen:
BESITZT - jemand/eigentümer besitzt etwas
VERTRAG_MIT - Vertragspartner
BEFINDET_SICH_IN - geografisch/räumlich enthalten in
TEIL_VON - strukturell Teil von etwas
FINANZIERT_VON - Geldquelle/Darlehen
VERSICHERT_BEI - Versicherungsanbieter
MIETET_BEI - Mieter-Beziehung
VERMIETET_AN - Vermieter-Beziehung
ARBEITET_BEI - Arbeitgeber
KREDIT_BEI - Kreditinstitut
KOSTET - Preis/Gebühr/Betrag einer Sache

Technik & Produkte:
HAT_KOMPONENTE - besteht aus / enthält Teile (z.B. Anlage hat Wechselrichter)
HAT_EIGENSCHAFT - Attribut/Merkmal/Parameter (z.B. Leistung, Kapazität)
STAMMT_VON - Herkunft/Erzeuger/Hersteller

Allgemein:
REFERENZIERT - verweist auf, zitiert, nennt explizit
BEZOGEN_AUF - NUR als absoluter letzter Ausweg wenn nichts Spezifischeres passt

## REGELN
1. Maximum 20 Entities pro Seite. Weniger ist besser.
2. Nur Entities die als Graph-Knoten Sinn ergeben — wäre "Immobilien" als Knoten nützlich? NEIN.
3. Relations MÜSSEN spezifisch sein. Wenn du BEZOGEN_AUF verwenden willst, überlege erst: Was ist die spezifische Verbindung? Ist es eine Komponente (HAT_KOMPONENTE)? Eine Eigenschaft (HAT_EIGENSCHAFT)? Eine Referenz (REFERENZIERT)? Eine Herkunft (STAMMT_VON)? Einen Betrag (KOSTET)? Wähle den passendsten Typ — erst dann den Fallback.
4. Beachte: Die Dokumente können verschiedene Themenbereiche abdecken (Immobilien, Technik, Kochbuch, Finanzen, Haushalte). Wähle den Relation-Typ passend zum Kontext — nicht nur Immobilien-Relations.
5. Jede Relation muss substantiell sein — „X befindet sich in Y" ist gut. "X ist bezogen auf Y" ist schlecht.
6. Keine Spekulation — nur was explizit im Text steht.
7. IDs: kebab-case, eindeutig, kurz
8. Labels auf Deutsch, descriptions auf Deutsch (max 25 Zeichen)
9. Nur gültiges JSON — kein Markdown, kein Text davor/danach

## CANONICAL NAMES (wichtig für Deduplizierung!)
- Nutze immer den **kürzesten sinnvollen Namen**: "Musterort" statt "projekt-musterort".
- Nutze den gebräuchlichen Namen: "Max" statt "Max Mustermann" (wenn die Entity schon bekannt ist).
- Wenn eine Entity sowohl LOCATION als auch CONCEPT sein könnte: wähle **LOCATION** für reale Orte,
  **CONCEPT** nur für abstrakte Themen.
- Wenn du unten bekannte Entities siehst und eine neue Entity dazu passt: nutze die vorhandene ID!

## NEGATIVE BEISPIELE (NICHT TUN)
Bad: {"id": "immobilien", "label": "Immobilien", "type": "CONCEPT"} → Zu generisch
Bad: {"id": "d41d8cd9f000-Musterdokument", "label": "d41d8cd9f000-Musterdokument", "type": "DOCUMENT"} → Hash-Müll
Bad: {"id": "trends", "label": "trends", "type": "CONCEPT"} → Frontmatter-Tag
Bad: {"id": "haupt-entity-xyz", "label": "Haupt-Entity der Seite xyz", "type": "CONCEPT"} → Meta-Platzhalter
Bad: {"source": "A", "target": "B", "type": "BEZOGEN_AUF"} → Fast immer falsch

## POSITIVE BEISPIELE (mit Relations!)
Good: {
  "entities": [
    {"id": "max-mustermann", "label": "Max Mustermann", "type": "PERSON"},
    {"id": "haus-hauptstr-12a", "label": "Haus Hauptstr. 23b", "type": "PROPERTY"},
    {"id": "gruenwald", "label": "Gruenwald", "type": "LOCATION"},
    {"id": "landratsamt-musterstadt", "label": "Landratsamt Musterstadt", "type": "ORGANIZATION"}
  ],
  "relationships": [
    {"source": "max-mustermann", "target": "haus-muster-1a", "type": "BESITZT", "description": "Max besitzt das Haus"},
    {"source": "haus-hauptstr-12a", "target": "gruenwald", "type": "BEFINDET_SICH_IN", "description": "Haus liegt in Gruenwald"},
    {"source": "max-mustermann", "target": "behoerde-musterstadt", "type": "VERTRAG_MIT", "description": "Vertrag mit Behörde"}
  ]
}

Good (Immobilien):
{
  "entities": [
    {"id": "allianz-versicherung", "label": "Allianz Versicherung", "type": "ORGANIZATION"},
    {"id": "wohnung-muenchen-maxvorstadt", "label": "Wohnung München Maxvorstadt", "type": "PROPERTY"},
    {"id": "1200-euro-miete", "label": "1200€ monatliche Miete", "type": "MONEY"}
  ],
  "relationships": [
    {"source": "wohnung-muenchen-maxvorstadt", "target": "allianz-versicherung", "type": "VERSICHERT_BEI", "description": "Wohnung versichert bei Allianz"},
    {"source": "wohnung-muenchen-maxvorstadt", "target": "1200-euro-miete", "type": "KOSTET", "description": "Monatliche Miete"}
  ]
}

Good (Technik):
{
  "entities": [
    {"id": "e3dc-hauskraftwerk-s10", "label": "E3DC Hauskraftwerk S10", "type": "FACILITY"},
    {"id": "e3dc-gmbh", "label": "E3DC GmbH", "type": "ORGANIZATION"},
    {"id": "10kw-leistung", "label": "10 kW Spitzenleistung", "type": "CONCEPT"}
  ],
  "relationships": [
    {"source": "e3dc-hauskraftwerk-s10", "target": "e3dc-gmbh", "type": "STAMMT_VON", "description": "Hersteller des Systems"},
    {"source": "e3dc-hauskraftwerk-s10", "target": "10kw-leistung", "type": "HAT_EIGENSCHAFT", "description": "Maximale Leistung"}
  ]
}

Good (Kochbuch):
{
  "entities": [
    {"id": "schnitzel-wiener-art", "label": "Schnitzel Wiener Art", "type": "CONCEPT"},
    {"id": "semmelbrösel", "label": "Semmelbrösel", "type": "CONCEPT"}
  ],
  "relationships": [
    {"source": "schnitzel-wiener-art", "target": "semmelbrösel", "type": "HAT_KOMPONENTE", "description": "Wird paniert mit Semmelbrösel"}
  ]
}

## ANTWORTFORMAT (BINDEND!)
Du MUSST gültiges JSON zurückgeben mit genau dieser Struktur:
{
  "entities": [
    {"id": "kebab-case-id", "label": "Lesbarer Name", "type": "ENTITY_TYPE", "description": "Max 25 Zeichen"}
  ],
  "relationships": [
    {"source": "entity_id_a", "target": "entity_id_b", "type": "RELATION_TYPE", "description": "Was verbindet sie"}
  ]
}

- Wenn du ≥3 Entities gefunden hast: mindestens 1-2 Beziehungen erstellen!
- relations ist ein Pflichtfeld - nicht weglassen, nicht leer lassen wenn Entities da sind
- Nutze die spezifischen Relation-Typen passend zum Dokument-Kontext
- BEZOGEN_AUF nur wenn wirklich nichts anderes passt (< 10% aller Relations)
"""


# ── Community Summary System Prompt ──
COMMUNITY_SYSTEM_PROMPT = """\
Du erstellst prägnante Zusammenfassungen für Entity-Communities in einem Wissensgraph.

Erstelle:
1. Label (kurz, prägnant, 2-5 Wörter)
2. Summary (2-3 Sätze, was verbindet diese Entities?)

Output als JSON: {"label": "...", "summary": "..."}
Kein Markdown-Codeblock, kein Text davor/danach.
"""


# ═══════════════════════════════════════════════════════════════════════
#  Vector Functions
# ═══════════════════════════════════════════════════════════════════════


def strip_frontmatter(text: str) -> tuple[dict, str]:
    m = FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    return (yaml.safe_load(m.group(1)) or {}), m.group(2)


def chunk_markdown(text: str) -> list[tuple[str, str]]:
    """Split markdown into (section_label, chunk_text) pairs.

    Uses semantic chunking (embedding-based) when SEMANTIC_CHUNKING is enabled
    and Ollama is reachable; falls back to character-based splitting otherwise.
    """
    text = HTML_COMMENT_RE.sub("", text)
    pieces: list[tuple[str, str]] = []
    parts = H2_SPLIT_RE.split(text.strip())
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part.startswith("## "):
            first_nl = part.find("\n")
            label = part[3:first_nl].strip() if first_nl > 0 else part[3:].strip()
            body = part[first_nl + 1 :].strip() if first_nl > 0 else ""
        else:
            label = "_preamble_"
            body = part
        for chunk in _split_long(body):
            if len(chunk) >= MIN_CHUNK_CHARS:
                pieces.append((label, chunk))
    return pieces


# ═══════════════════════════════════════════════════════════════════════
#  Semantic Chunking Helpers
# ═══════════════════════════════════════════════════════════════════════

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D vectors."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(dot / norm) if norm > 0 else 0.0


def _semantic_split_long(body: str) -> list[str]:
    """Greedy semantic chunking based on paragraph-level embeddings.

    Algorithm:
    1. Split body into paragraphs (double-newline separator).
    2. Compute all paragraph embeddings in a single batch.
    3. Walk paragraphs greedily:
       - Add paragraph to current chunk if it fits under MAX_CHUNK_CHARS
         AND cosine similarity to previous paragraph >= SEMANTIC_THRESHOLD.
       - Otherwise start a new chunk.
    4. Return chunks (caller filters by MIN_CHUNK_CHARS).

    Returns an empty list on any embedding failure so the caller can fall
    back to character-based splitting.
    """
    if len(body) <= MAX_CHUNK_CHARS:
        return [body] if body.strip() else []

    paragraphs = [
        p.strip()
        for p in re.split(r"\n\s*\n", body)
        if p.strip()
    ]
    if not paragraphs:
        return []

    # Split any oversized paragraphs before embedding to ensure no chunk
    # exceeds MAX_CHUNK_CHARS (prevents bge-m3 embedding errors on monster
    # paragraphs that would otherwise pass through unsplit).
    expanded_paras = []
    for para in paragraphs:
        if len(para) > MAX_CHUNK_CHARS:
            # Char-based split (not _split_long to avoid recursion)
            pieces = []
            start = 0
            while start < len(para):
                end = start + MAX_CHUNK_CHARS
                if end >= len(para):
                    pieces.append(para[start:])
                    break
                # Try to break at a sentence boundary
                break_point = para.rfind('.', start, end)
                if break_point == -1:
                    break_point = para.rfind('\n', start, end)
                if break_point == -1 or break_point < start + 100:
                    break_point = end
                pieces.append(para[start:break_point + 1].strip())
                start = break_point + 1
            expanded_paras.extend(pieces)
        else:
            expanded_paras.append(para)
    paragraphs = expanded_paras

    if len(paragraphs) == 1:
        return [paragraphs[0]]

    # Batch-embed all paragraphs
    try:
        embeddings = _embed_raw(paragraphs)
    except Exception as e:
        log("warn", f" semantic chunking unavailable: {e} — falling back to char-split")
        return []  # signal fallback

    # Sanitize: replace NaN/zero embeddings with zero vectors to prevent
    # cosine similarity crashes on bge-m3 edge cases.
    for i in range(len(embeddings)):
        if np.isnan(embeddings[i]).any() or np.all(embeddings[i] == 0):
            log("warn", f" bad embedding for paragraph {i}, replacing with zeros")
            embeddings[i] = np.zeros_like(embeddings[i])

    chunks: list[str] = []
    current_paras: list[str] = [paragraphs[0]]
    current_len = len(paragraphs[0])
    prev_emb = embeddings[0]

    for i in range(1, len(paragraphs)):
        para = paragraphs[i]
        emb = embeddings[i]
        sim = _cosine_similarity(prev_emb, emb)

        # Check hard size limit first
        would_exceed = current_len + len(para) + 2 > MAX_CHUNK_CHARS

        if would_exceed or sim < SEMANTIC_THRESHOLD:
            # Flush current chunk
            chunks.append("\n\n".join(current_paras))
            current_paras = [para]
            current_len = len(para)
        else:
            current_paras.append(para)
            current_len += len(para) + 2

        prev_emb = emb

    # Flush remaining
    if current_paras:
        chunks.append("\n\n".join(current_paras))

    return chunks


def _split_long(body: str) -> list[str]:
    """Split a long body into chunks.

    Tries semantic chunking first if enabled; falls back to character-based
    splitting on embedding failure or when the flag is off.
    """
    # Fast path: body already fits
    if len(body) <= MAX_CHUNK_CHARS:
        return [body] if body.strip() else []

    # Try semantic chunking if enabled
    if SEMANTIC_CHUNKING:
        semantic_result = _semantic_split_long(body)
        if semantic_result:
            return semantic_result

    # ── Char-based fallback (original algorithm) ──
    chunks: list[str] = []
    buf = ""
    for para in re.split(r"\n\s*\n", body):
        para = para.strip()
        if not para:
            continue
        if len(buf) + len(para) + 2 > MAX_CHUNK_CHARS and buf:
            chunks.append(buf.strip())
            buf = para
        else:
            buf = f"{buf}\n\n{para}" if buf else para
    if buf.strip():
        chunks.append(buf.strip())
    return chunks


def _embed_raw(texts: list[str]) -> np.ndarray:
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/embed",
        data=json.dumps({"model": EMBED_MODEL, "input": texts}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=3600) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return np.asarray(body["embeddings"], dtype=np.float32)


def embed(texts: list[str]) -> np.ndarray:
    """Embed texts, returning L2-normalized vectors. Chunks that trigger
    model errors (e.g. bge-m3 NaN quantization edge cases on Ollama) get a
    zero vector and a warning; callers drop zero-norm rows before indexing."""
    if not texts:
        return np.zeros((0, EMBED_DIMS), dtype=np.float32)
    try:
        arr = _embed_raw(texts)
    except urllib.error.HTTPError:
        arr = np.zeros((len(texts), EMBED_DIMS), dtype=np.float32)
        for i, t in enumerate(texts):
            try:
                arr[i] = _embed_raw([t])[0]
            except urllib.error.HTTPError as e:
                detail = ""
                try:
                    detail = e.read().decode("utf-8", errors="replace")[:120]
                except Exception:
                    pass
                preview = t[:80].replace("\n", " ")
                log("warn", f"embed skipped chunk (HTTP {e.code}): {preview!r}... err={detail}", stderr=True)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0, 1.0, norms)
    return arr / safe_norms


# ═══════════════════════════════════════════════════════════════════════
#  SQLite Schema & Helpers
# ═══════════════════════════════════════════════════════════════════════


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    cols = [row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    return column in cols


def _has_unique_constraint(conn: sqlite3.Connection) -> bool:
    for idx_row in conn.execute("PRAGMA index_list(chunks)").fetchall():
        idx_name = idx_row[1]
        if not idx_name.startswith("sqlite_autoindex_chunks_"):
            continue
        cols = [r[2] for r in conn.execute(f"PRAGMA index_info({idx_name})").fetchall()]
        if cols == ["scope", "kind", "ref", "chunk_idx"]:
            return True
    return False


def _record_migration(conn: sqlite3.Connection, name: str) -> None:
    """Insert a migration marker into _migrations (idempotent)."""
    conn.execute("INSERT OR IGNORE INTO _migrations(name) VALUES (?)", (name,))
    conn.commit()


def _record_graph_migration(conn: sqlite3.Connection, name: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO _graph_migrations(name) VALUES (?)", (name,)
    )
    conn.commit()


def _migrate_unique_constraint(conn: sqlite3.Connection) -> None:
    """Add UNIQUE(scope, kind, ref, chunk_idx) to an existing chunks table."""
    log("migrate", " adding UNIQUE(scope, kind, ref, chunk_idx) to chunks ...")
    conn.execute("DROP TABLE IF EXISTS chunks_new")

    conn.execute(
        """
        CREATE TABLE chunks_new (
            id INTEGER PRIMARY KEY,
            scope TEXT NOT NULL,
            kind TEXT NOT NULL,
            ref TEXT NOT NULL,
            section TEXT,
            chunk_idx INTEGER NOT NULL,
            content TEXT NOT NULL,
            content_hash TEXT NOT NULL DEFAULT '',
            embedding BLOB NOT NULL,
            UNIQUE(scope, kind, ref, chunk_idx)
        )
        """)

    conn.execute(
        """
        INSERT INTO chunks_new (id, scope, kind, ref, section, chunk_idx,
                                content, content_hash, embedding)
        SELECT id, scope, kind, ref, section, chunk_idx,
               content, content_hash, embedding
        FROM chunks
        WHERE id IN (
            SELECT MAX(id) FROM chunks
            GROUP BY scope, kind, ref, chunk_idx
        )
        """
    )

    conn.execute("BEGIN IMMEDIATE")
    conn.execute("DROP TABLE chunks")
    conn.execute("ALTER TABLE chunks_new RENAME TO chunks")
    conn.execute("COMMIT")

    conn.executescript(
        """
        CREATE INDEX IF NOT EXISTS idx_chunks_scope ON chunks(scope);
        CREATE INDEX IF NOT EXISTS idx_chunks_ref ON chunks(scope, kind, ref);
        CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(content_hash);
        """
    )

    _record_migration(conn, "unique_constraint_scope_kind_ref_chunk_idx")
    new_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    log("migrate", f" done - {new_count} rows after dedup")


def init_db(conn: sqlite3.Connection) -> None:
    """Initialize both vector index and graph tables (idempotent)."""
    # ── Vector index tables ──
    conn.execute(
        "CREATE TABLE IF NOT EXISTS _migrations (name TEXT PRIMARY KEY)"
    )
    conn.commit()

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            scope TEXT NOT NULL,
            kind TEXT NOT NULL,
            ref TEXT NOT NULL,
            section TEXT,
            chunk_idx INTEGER NOT NULL,
            content TEXT NOT NULL,
            content_hash TEXT NOT NULL DEFAULT '',
            embedding BLOB NOT NULL,
            UNIQUE(scope, kind, ref, chunk_idx)
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_scope ON chunks(scope);
        CREATE INDEX IF NOT EXISTS idx_chunks_ref ON chunks(scope, kind, ref);
        CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(content_hash);
        """
    )

    if not _has_unique_constraint(conn):
        _migrate_unique_constraint(conn)

    # ── Graph tables ──
    conn.execute(
        "CREATE TABLE IF NOT EXISTS _graph_migrations (name TEXT PRIMARY KEY)"
    )
    conn.commit()

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS entities (
            id            TEXT PRIMARY KEY,
            label         TEXT NOT NULL,
            entity_type   TEXT NOT NULL,
            description   TEXT,
            wiki_page     TEXT,
            created_at    TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at    TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS relationships (
            id            TEXT PRIMARY KEY,
            source_id     TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
            target_id     TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
            relation_type TEXT NOT NULL,
            description   TEXT,
            created_at    TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_entities_label  ON entities(label COLLATE NOCASE);
        CREATE INDEX IF NOT EXISTS idx_entities_type   ON entities(entity_type);
        CREATE INDEX IF NOT EXISTS idx_entities_page   ON entities(wiki_page);
        CREATE INDEX IF NOT EXISTS idx_rel_source      ON relationships(source_id);
        CREATE INDEX IF NOT EXISTS idx_rel_target      ON relationships(target_id);
        CREATE INDEX IF NOT EXISTS idx_rel_type        ON relationships(relation_type);

        CREATE TABLE IF NOT EXISTS communities (
            id                TEXT PRIMARY KEY,
            level             INTEGER NOT NULL,
            label             TEXT,
            summary           TEXT,
            summary_embedding BLOB,
            entity_count      INTEGER DEFAULT 0,
            created_at        TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at        TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS community_members (
            community_id TEXT NOT NULL REFERENCES communities(id) ON DELETE CASCADE,
            entity_id    TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
            PRIMARY KEY (community_id, entity_id)
        );

        CREATE INDEX IF NOT EXISTS idx_cm_entity     ON community_members(entity_id);
        CREATE INDEX IF NOT EXISTS idx_cm_community  ON community_members(community_id);
        """
    )

    # ── Incremental Graph Tables (v2) ──
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS entity_pages (
            entity_id  TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
            wiki_page  TEXT NOT NULL,
            PRIMARY KEY (entity_id, wiki_page)
        );
        CREATE INDEX IF NOT EXISTS idx_ep_wiki ON entity_pages(wiki_page);

        CREATE TABLE IF NOT EXISTS relationship_pages (
            rel_id     TEXT NOT NULL REFERENCES relationships(id) ON DELETE CASCADE,
            wiki_page  TEXT NOT NULL,
            PRIMARY KEY (rel_id, wiki_page)
        );
        CREATE INDEX IF NOT EXISTS idx_rp_wiki ON relationship_pages(wiki_page);

        CREATE TABLE IF NOT EXISTS graph_page_state (
            wiki_page      TEXT PRIMARY KEY,
            body_hash      TEXT NOT NULL,
            processed_at   TEXT NOT NULL DEFAULT (datetime('now'))
        );
        """
    )

    # Add wiki_page column to relationships if not present (legacy migration)
    if _has_column(conn, "relationships", "wiki_page") is False:
        conn.execute("ALTER TABLE relationships ADD COLUMN wiki_page TEXT")
        conn.commit()
        _record_graph_migration(conn, "v2_relationships_wiki_page")

    _record_graph_migration(conn, "v2_incremental_tables")


# ═══════════════════════════════════════════════════════════════════════
#  Document Gathering
# ═══════════════════════════════════════════════════════════════════════


def gather_documents() -> list[tuple[str, str, str, str]]:
    """Yield (scope, kind, ref, markdown_body) for every wiki page and source doc."""
    docs: list[tuple[str, str, str, str]] = []
    for scope in SCOPES:
        wiki_scope = WIKI / scope
        if wiki_scope.exists():
            for p in sorted(wiki_scope.glob("*.md")):
                _, body = strip_frontmatter(p.read_text(encoding="utf-8"))
                docs.append((scope, "wiki", p.stem, body))
        src_scope = SOURCES / scope
        if src_scope.exists():
            # Structured sources: subfolder/document.md
            for md in sorted(src_scope.glob("*/document.md")):
                docs.append((scope, "source", md.parent.name, md.read_text(encoding="utf-8")))
            # Flat sources: direct .md files (e.g. youtube transcripts)
            for md in sorted(src_scope.glob("*.md")):
                docs.append((scope, "source", md.stem, md.read_text(encoding="utf-8")))
    return docs


def gather_wiki_pages() -> list[Path]:
    """Return sorted list of all wiki page paths."""
    pages: list[Path] = []
    for scope in SCOPES:
        wiki_dir = WIKI / scope
        if wiki_dir.exists():
            pages.extend(sorted(wiki_dir.glob("*.md")))
    return pages


def _hash_content(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


# ═══════════════════════════════════════════════════════════════════════
#  Vector Build
# ═══════════════════════════════════════════════════════════════════════


def build_vectors() -> int:
    """Build/update the vector index. Returns number of chunks inserted."""
    VECTORDB.mkdir(exist_ok=True)

    existing_db = DB_PATH.exists()
    need_full_rebuild = False

    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    if existing_db and not _has_column(conn, "chunks", "content_hash"):
        bak = DB_PATH.with_suffix(".sqlite.bak")
        conn.close()
        DB_PATH.rename(bak)
        log("migrate", f" old DB backed up to {bak} (missing content_hash column)")
        conn = sqlite3.connect(DB_PATH)
        init_db(conn)
        need_full_rebuild = True

    existing_doc_hashes: dict[tuple[str, str, str], set[str]] = {}
    if existing_db and not need_full_rebuild:
        for row in conn.execute(
            "SELECT scope, kind, ref, content_hash FROM chunks"
        ).fetchall():
            key = (row[0], row[1], row[2])
            existing_doc_hashes.setdefault(key, set()).add(row[3])
        log("inc", f" loaded hashes for {len(existing_doc_hashes)} doc(s) from DB")

    docs = gather_documents()
    doc_hashes: dict[tuple[str, str, str], set[str]] = {}
    doc_chunks: dict[tuple[str, str, str], list[tuple[int, str, str, str]]] = {}
    for scope, kind, ref, body in docs:
        key = (scope, kind, ref)
        chunks = chunk_markdown(body)
        hashes = set()
        chunk_list = []
        for idx, (section, content) in enumerate(chunks):
            ch = _hash_content(content)
            hashes.add(ch)
            chunk_list.append((idx, section, content, ch))
        doc_hashes[key] = hashes
        doc_chunks[key] = chunk_list
        log("info", f" {scope}/{kind}:{ref} -> {len(chunks)} chunk(s)")

    live_refs = set(doc_hashes.keys())
    db_refs = conn.execute(
        "SELECT DISTINCT scope, kind, ref FROM chunks"
    ).fetchall()
    deleted_docs = 0
    for scope, kind, ref in db_refs:
        if (scope, kind, ref) not in live_refs:
            conn.execute(
                "DELETE FROM chunks WHERE scope=? AND kind=? AND ref=?",
                (scope, kind, ref),
            )
            deleted_docs += 1
    if deleted_docs:
        conn.commit()
        log("inc", f" removed chunks for {deleted_docs} deleted doc(s)")

    changed_refs: list[tuple[str, str, str]] = []
    unchanged_count = 0
    if not need_full_rebuild:
        for ref_key, new_hashes in doc_hashes.items():
            old_hashes = existing_doc_hashes.get(ref_key, set())
            if old_hashes == new_hashes:
                unchanged_count += len(doc_chunks[ref_key])
            else:
                changed_refs.append(ref_key)
                conn.execute(
                    "DELETE FROM chunks WHERE scope=? AND kind=? AND ref=?",
                    ref_key,
                )
        conn.commit()
        if changed_refs:
            log("inc", f" cleared old chunks for {len(changed_refs)} changed doc(s)")

    rows_to_insert: list[tuple] = []
    flat_texts: list[str] = []
    for ref_key, chunk_list in doc_chunks.items():
        doc_changed = ref_key in changed_refs
        for idx, section, content, ch in chunk_list:
            if not need_full_rebuild and not doc_changed:
                continue
            rows_to_insert.append((*ref_key, section, idx, content, ch))
            flat_texts.append(content)

    inserted = 0
    embed_failed = 0
    if flat_texts:
        batch_size = 8
        vectors = np.zeros((len(flat_texts), EMBED_DIMS), dtype=np.float32)
        for start in range(0, len(flat_texts), batch_size):
            end = min(start + batch_size, len(flat_texts))
            vectors[start:end] = embed(flat_texts[start:end])
            log("info", f" embedded {end}/{len(flat_texts)}")

        for (scope, kind, ref, section, idx, content, ch), vec in zip(rows_to_insert, vectors):
            if not np.any(vec):
                embed_failed += 1
                continue
            conn.execute(
                "INSERT OR REPLACE INTO chunks(scope, kind, ref, section, chunk_idx, content, content_hash, embedding) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (scope, kind, ref, section, idx, content, ch, vec.tobytes()),
            )
            inserted += 1
    else:
        log("inc", " no new or changed chunks to embed")

    conn.commit()
    conn.close()

    skip_note = f" (failed {embed_failed})" if embed_failed else ""
    total_docs = sum(len(c) for c in doc_chunks.values())
    log("done", f" +{inserted} inserted, ~{unchanged_count} unchanged chunks, "
        f"-{deleted_docs} deleted docs{skip_note} ({total_docs} total chunks scanned) → {DB_PATH}")
    return inserted


# ═══════════════════════════════════════════════════════════════════════
#  Graph: Entity Resolution (Deduplication)
# ═══════════════════════════════════════════════════════════════════════


def _token_set_overlap(a: str, b: str) -> int:
    """Naive token-set Jaccard similarity × 100."""
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa or not sb:
        return 0
    intersection = sa & sb
    union = sa | sb
    return int(len(intersection) / len(union) * 100)


def _levenshtein_ratio(a: str, b: str) -> int:
    """Levenshtein distance → similarity percentage. Pure Python."""
    if a == b:
        return 100
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0
    if abs(la - lb) > 20:
        return 0
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        curr = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    dist = prev[lb]
    max_len = max(la, lb)
    return int((1 - dist / max_len) * 100)


def _fuzzy_match_score(a: str, b: str) -> int:
    """Combined score: max of token-set overlap and levenshtein ratio."""
    return max(_token_set_overlap(a, b), _levenshtein_ratio(a, b))


def _merge_description(old: str | None, new: str | None) -> str | None:
    """Merge two descriptions, keeping unique info."""
    if not old:
        return new
    if not new:
        return old
    if old == new:
        return old
    return f"{old} | {new}"


def resolve_entity(
    conn: sqlite3.Connection,
    new_id: str,
    label: str,
    entity_type: str,
    description: str,
    wiki_page: str,
    registry: EntityRegistry | None = None,
) -> str:
    """Check for existing similar entity. If found → update, else insert.
    Returns the canonical entity id.

    Priority chain:
      1. Registry lookup (highest priority - authoritative mapping)
      2. ID exact match
      3. Label NOCASE exact match
      4. Fuzzy match within same entity_type (threshold 85%)
      5. Cross-type fuzzy match (threshold 90%, first-write-wins on type)
      6. Insert as new entity
    """

    # ── Priority 1: Registry lookup ──────────────────────────────
    if registry is not None:
        reg_id = registry.lookup(label)
        if reg_id:
            # Registry is authoritative - verify it exists in DB
            row = conn.execute("SELECT id FROM entities WHERE id = ?", (reg_id,)).fetchone()
            if row:
                return reg_id

    # ── Priority 2: ID exact match ───────────────────────────────
    row = conn.execute(
        "SELECT id, label, description FROM entities WHERE id = ?", (new_id,)
    ).fetchone()
    if row:
        eid, _, old_desc = row
        merged = _merge_description(old_desc, description)
        conn.execute(
            "UPDATE entities SET description = ?, updated_at = datetime('now') WHERE id = ?",
            (merged, eid),
        )
        return eid

    # ── Priority 3: Label NOCASE exact match ─────────────────────
    row = conn.execute(
        "SELECT id, label, description FROM entities WHERE label COLLATE NOCASE = ?",
        (label,),
    ).fetchone()
    if row:
        eid, _, old_desc = row
        merged = _merge_description(old_desc, description)
        conn.execute(
            "UPDATE entities SET description = ?, updated_at = datetime('now') WHERE id = ?",
            (merged, eid),
        )
        return eid

    # ── Priority 4: Fuzzy match within same entity_type ──────────
    candidates = conn.execute(
        "SELECT id, label, description FROM entities WHERE entity_type = ?",
        (entity_type,),
    ).fetchall()
    best_score = 0
    best_eid = None
    best_desc = None
    for cid, c_label, c_desc in candidates:
        score = _fuzzy_match_score(label, c_label)
        if score > best_score:
            best_score = score
            best_eid = cid
            best_desc = c_desc

    if best_score >= FUZZY_THRESHOLD and best_eid:
        merged = _merge_description(best_desc, description)
        conn.execute(
            "UPDATE entities SET description = ?, updated_at = datetime('now') WHERE id = ?",
            (merged, best_eid),
        )
        log("dedup", f"  [dedup] '{label}' → matched '{best_eid}' (score={best_score}%)")
        return best_eid

    # ── Priority 5: Cross-type fuzzy match (threshold 90%) ───────
    all_entities = conn.execute(
        "SELECT id, label, entity_type, description FROM entities"
    ).fetchall()
    cross_best_score = 0
    cross_best_id = None
    cross_best_label = None
    cross_best_type = None
    cross_best_desc = None
    for eid, e_label, e_type, e_desc in all_entities:
        if e_type == entity_type:
            continue  # already checked in Priority 4
        score = _fuzzy_match_score(label, e_label)
        if score >= CROSS_TYPE_FUZZY_THRESHOLD and score > cross_best_score:
            cross_best_score = score
            cross_best_id = eid
            cross_best_label = e_label
            cross_best_type = e_type
            cross_best_desc = e_desc

    if cross_best_id:
        # First-write-wins: older entity's type is preserved
        log("dedup",
            f"  [cross-type-dedup] '{label}'[{entity_type}] → "
            f"'{cross_best_label}'[{cross_best_type}] "
            f"(score={cross_best_score}%) - Typ des älteren Entities "
            f"({cross_best_type}) beibehalten (first-write-wins)")
        merged = _merge_description(cross_best_desc, description)
        conn.execute(
            "UPDATE entities SET description = ?, updated_at = datetime('now') WHERE id = ?",
            (merged, cross_best_id),
        )
        return cross_best_id

    # ── Priority 6: Insert as new entity ─────────────────────────
    conn.execute(
        "INSERT INTO entities(id, label, entity_type, description, wiki_page) "
        "VALUES (?, ?, ?, ?, ?)",
        (new_id, label, entity_type, description, wiki_page),
    )
    return new_id


# ═══════════════════════════════════════════════════════════════════════
#  Graph: Frontmatter Seed Extraction
# ═══════════════════════════════════════════════════════════════════════


def _stable_entity_id(entity_type: str, label: str) -> str:
    """Generate a stable entity ID from type + normalized label."""
    normalized = label.lower().strip()
    # Umlaute normalisieren
    normalized = normalized.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
    normalized = normalized.replace("ß", "ss")
    # Sonderzeichen entfernen
    normalized = re.sub(r"[^\w\s-]", "", normalized)
    normalized = re.sub(r"[\s_]+", "-", normalized)
    normalized = normalized.strip("-").lower()

    hash_input = f"{entity_type}::{normalized}"
    return hashlib.sha256(hash_input.encode()).hexdigest()[:12]


def _slugify(text: str) -> str:
    """Simple kebab-case slugifier."""
    s = text.lower().strip()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_]+", "-", s)
    s = re.sub(r"-+", "-", s)
    return s.strip("-")


def _seed_from_frontmatter(
    fm: dict, slug: str
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Extract initial entities from YAML frontmatter (before LLM).

    IMPORTANT: No automatic BEZOGEN_AUF relations. Entities only.
    Relations must come from the LLM extraction step.
    """
    entities: List[Dict[str, str]] = []
    relationships: List[Dict[str, str]] = []  # always empty - LLM owns relations

    title = fm.get("title", slug)
    title_id = _slugify(title)
    entities.append({
        "id": title_id,
        "label": title,
        "type": "DOCUMENT",
        "description": f"Wiki-Seite: {slug}",
    })

    topics = fm.get("topics", [])
    if isinstance(topics, list):
        for topic in topics:
            tid = _slugify(str(topic))
            if tid != title_id:
                entities.append({
                    "id": tid,
                    "label": str(topic),
                    "type": "CONCEPT",
                    "description": f"Thema aus '{slug}'",
                })
                # KEINE automatische BEZOGEN_AUF Relation hier!

    sources = fm.get("sources", [])
    if isinstance(sources, list):
        for src in sources:
            sid = _slugify(str(src))
            entities.append({
                "id": sid,
                "label": str(src),
                "type": "DOCUMENT",
                "description": f"Quelle aus '{slug}'",
            })
            # KEINE automatische BEZOGEN_AUF Relation hier!

    return entities, relationships


def _fix_json(raw: str) -> str | None:
    """Try to repair common LLM JSON output issues.

    Fixes: trailing commas, unterminated strings, unescaped newlines,
    single quotes, missing closing brackets.
    Returns repaired string or None if input looks unrecoverable.
    """
    if not raw:
        return None

    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    # Find the outermost JSON-like boundaries
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        start = text.find("[")
        end = text.rfind("]")
        if start < 0 or end <= start:
            return None
    text = text[start:end + 1]

    # Fix unterminated strings: lines ending with ": " or "" without close
    lines = text.split("\n")
    fixed_lines = []
    for line in lines:
        # Count quotes - if odd, string is likely unterminated
        stripped = line.rstrip()
        quote_count = stripped.count('"') - stripped.count('\\"')
        if quote_count % 2 == 1:
            # Odd quotes - try to close the string
            if stripped.endswith(':'):
                stripped += ' ""'
            elif stripped.endswith('"'):
                # Already has opening quote but no closing
                # Try to find if there's a value part
                parts = stripped.split('":', 1)
                if len(parts) == 1:
                    stripped += '"'
                else:
                    # Has key":  but no value - close with empty and comma
                    stripped = parts[0] + '": "",'
            else:
                # String in middle of line is broken - just close it
                stripped += '"'
        fixed_lines.append(stripped)
    text = "\n".join(fixed_lines)

    # Remove trailing commas before } or ]
    text = re.sub(r',\s*([}\]])', r'\1', text)

    # Fix single quotes on property names ONLY (before colon, not inside string values)
    # Avoids corrupting apostrophes like "It's" inside values
    text = re.sub(r"(?<=:)\\s*'([^']+)'(?=\\s*[,}:\\\\])", r' "\1"', text)

    return text


# ═══════════════════════════════════════════════════════════════════════
#  Graph: Ollama Entity Extraction
# ═══════════════════════════════════════════════════════════════════════


def _ollama_extract(text: str, known_entities_text: str | None = None) -> Dict[str, Any]:
    """Call vLLM for entity extraction. Retries with exponential backoff.

    Args:
        text: The markdown body to extract entities from.
        known_entities_text: Pre-formatted Known-Entities section from
            EntityRegistry.to_prompt_section(). Injected into the system
            prompt so the LLM can reuse existing entity IDs.
    """
    system_prompt = EXTRACTION_SYSTEM_PROMPT
    if known_entities_text:
        system_prompt = EXTRACTION_SYSTEM_PROMPT + "\n" + known_entities_text

    payload = {
        "model": GRAPH_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Extrahiere Entities und Beziehungen aus:\n\n{text}"},
        ],
        "temperature": 0.3,
        "max_tokens": 8192,
    }

    last_err: Optional[Exception] = None
    raw = ''
    for attempt in range(RETRY_MAX):
        try:
            req = urllib.request.Request(
                f"{VLLM_URL}/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=3600) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            msg = body["choices"][0]["message"]
            raw = (msg.get("content") or msg.get("reasoning") or "").strip()

            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
                cleaned = re.sub(r"\s*```$", "", cleaned)
            result = json.loads(cleaned)
            return result

        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", errors="replace")[:200]
            except Exception:
                pass
            last_err = e
            log("warn",
                f"vLLM HTTP {e.code} (attempt {attempt + 1}/{RETRY_MAX}): {detail}",
                stderr=True)
            if attempt < RETRY_MAX - 1:
                time.sleep(RETRY_BASE * (2 ** attempt))
        except json.JSONDecodeError as e:
            last_err = e
            preview = raw[:300].replace("\n", " ").replace("\r", " ")
            log("warn",
                f"JSON parse error (attempt {attempt + 1}/{RETRY_MAX}): {e} | Raw preview: {preview}",
                stderr=True)
            # Auto-fix common LLM JSON issues before retrying
            if attempt < RETRY_MAX - 1:
                fixed = _fix_json(raw)
                if fixed:
                    log("info", f"  [auto-fix] Trying repaired JSON...")
                    try:
                        result_fixed = json.loads(fixed)
                        if isinstance(result_fixed, dict):
                            return result_fixed
                    except json.JSONDecodeError:
                        pass  # fall through to retry
                time.sleep(RETRY_BASE * (2 ** attempt))
        except Exception as e:
            last_err = e
            log("warn",
                f"vLLM error (attempt {attempt + 1}/{RETRY_MAX}): {e}",
                stderr=True)
            if attempt < RETRY_MAX - 1:
                time.sleep(RETRY_BASE * (2 ** attempt))

    # Alle Retries erschöpft - JSON-Content nochmal versuchen zu reparieren
    if raw:
        fixed = _fix_json(raw)
        if fixed:
            try:
                result_final = json.loads(fixed)
                if isinstance(result_final, dict):
                    return result_final
            except json.JSONDecodeError:
                pass  # auch Auto-Fix konnte es nicht retten

    raise RuntimeError(
        f"vLLM extraction failed after {RETRY_MAX} attempts: {last_err}"
    )


def _ollama_health_check() -> bool:
    """Quick health check: is Ollama reachable (needed for embeddings)?"""
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            resp.read()
        return True
    except Exception as e:
        log("error", f" Ollama not reachable at {OLLAMA_URL}: {e}", stderr=True)
        return False


def _vllm_health_check() -> bool:
    """Quick health check: is vLLM reachable (needed for extraction + summaries)?"""
    try:
        req = urllib.request.Request(
            f"{VLLM_URL}/v1/models",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()
        return True
    except Exception as e:
        log("error", f" vLLM not reachable at {VLLM_URL}: {e}", stderr=True)
        return False


# ═══════════════════════════════════════════════════════════════════════
#  Graph: Page Processing
# ═══════════════════════════════════════════════════════════════════════


def process_page(
    filepath: Path,
    conn: sqlite3.Connection,
    dry_run: bool = False,
    wiki_page: str | None = None,
    autocommit: bool = True,
    registry: EntityRegistry | None = None,
) -> Tuple[int, int]:
    """Process a single wiki page for entity/relationship extraction.
    Returns (entities_added, relations_added).

    Args:
        filepath: Path to the wiki markdown file.
        conn: Open SQLite connection.
        dry_run: If True, skip DB writes.
        wiki_page: Pre-computed "scope/slug" string (used by incremental builder).
        autocommit: If True, commit after writes. Set False when caller manages
            the transaction (e.g., update_graph_incremental).
        registry: Optional EntityRegistry for known-entity dedup and
            prompt injection. When provided, top-30 known entities are
            injected into the LLM system prompt, and resolve_entity()
            uses registry lookup as highest priority.
    """
    slug = filepath.stem
    scope = filepath.parts[-2]
    if wiki_page is None:
        wiki_page = f"{scope}/{slug}"
    full_text = filepath.read_text(encoding="utf-8")

    fm, body = strip_frontmatter(full_text)
    log("page", f" {scope}/{slug}")

    # 1. Frontmatter seed
    seed_entities, seed_relations = _seed_from_frontmatter(fm, slug)

    # 2. LLM extraction (body only, trimmed to ~28K chars)
    llm_entities: List[Dict] = []
    llm_relations: List[Dict] = []
    if body.strip():
        truncated = body[:28000]
        try:
            # Inject known entities into prompt if registry is available
            known_text = registry.to_prompt_section() if registry else None
            result = _ollama_extract(truncated, known_entities_text=known_text)
            llm_entities = result.get("entities", [])
            llm_relations = result.get("relationships", [])
            log("llm", f"  [llm] {len(llm_entities)} entities, {len(llm_relations)} relations")

            # Fix 3: Validate LLM relation types - warn if BEZOGEN_AUF dominates
            if llm_relations:
                bezogen_count = sum(1 for r in llm_relations if r.get("type") == "BEZOGEN_AUF")
                bezogen_pct = (bezogen_count / len(llm_relations)) * 100
                if bezogen_pct >= 50:
                    log("warn",
                        f"  [graph-health] {bezogen_count}/{len(llm_relations)} LLM relations are BEZOGEN_AUF ({bezogen_pct:.0f}%). Prompt may need tuning.",
                        stderr=True)
        except RuntimeError as e:
            log("skip", f"  [skip] {e}", stderr=True)
        time.sleep(RATE_LIMIT_S)

    all_entities = seed_entities + llm_entities
    all_relations = seed_relations + llm_relations

    if dry_run:
        return len(all_entities), len(all_relations)

    # 3. Resolve entities (dedup + insert/update)
    id_map: Dict[str, str] = {}
    added = 0
    for ent in all_entities:
        orig_id = ent.get("id", _slugify(ent.get("label", "")))
        # Compute stable ID for dedup across pages; resolve_entity() falls back to fuzzy-match
        # so existing entities with legacy IDs are still found by label.
        stable_id = _stable_entity_id(
            ent.get("type", "CONCEPT"),
            ent.get("label", orig_id),
        )
        canonical = resolve_entity(
            conn,
            new_id=stable_id,
            label=ent.get("label", orig_id),
            entity_type=ent.get("type", "CONCEPT"),
            description=ent.get("description", ""),
            wiki_page=wiki_page,
            registry=registry,
        )
        # Register new entities in the registry for future pages
        if registry:
            registry.add_if_new({
                "id": canonical,
                "label": ent.get("label", orig_id),
                "type": ent.get("type", "CONCEPT"),
                "description": ent.get("description", ""),
            })
        id_map[orig_id] = canonical
        # Junction table: entity → wiki_page
        conn.execute(
            "INSERT OR IGNORE INTO entity_pages(entity_id, wiki_page) VALUES (?, ?)",
            (canonical, wiki_page),
        )
        row = conn.execute(
            "SELECT created_at, updated_at FROM entities WHERE id = ?", (canonical,)
        ).fetchone()
        if row and row[0] == row[1]:
            added += 1

    # 4. Insert relationships (map IDs, ensure FK integrity)
    inserted_rels = 0
    for rel in all_relations:
        src_orig = rel.get("source", "")
        tgt_orig = rel.get("target", "")
        src_id = id_map.get(src_orig, src_orig)
        tgt_id = id_map.get(tgt_orig, tgt_orig)

        for eid in (src_id, tgt_id):
            existing = conn.execute(
                "SELECT 1 FROM entities WHERE id = ?", (eid,)
            ).fetchone()
            if not existing:
                conn.execute(
                    "INSERT OR IGNORE INTO entities(id, label, entity_type, description, wiki_page) "
                    "VALUES (?, ?, 'CONCEPT', '', ?)",
                    (eid, eid, wiki_page),
                )
                # Also register the stub entity in entity_pages
                conn.execute(
                    "INSERT OR IGNORE INTO entity_pages(entity_id, wiki_page) VALUES (?, ?)",
                    (eid, wiki_page),
                )

        rel_type = rel.get("type")
        if not rel_type:
            log("warn", f"  [graph-health] Relation missing type, dropping: {rel}", stderr=True)
            continue
        rel_desc = rel.get("description", "")
        rel_id = hashlib.sha256(
            f"{src_id}::{tgt_id}::{rel_type}".encode()
        ).hexdigest()[:16]

        existing_rel = conn.execute(
            "SELECT 1 FROM relationships WHERE source_id=? AND target_id=? AND relation_type=?",
            (src_id, tgt_id, rel_type),
        ).fetchone()
        if not existing_rel:
            conn.execute(
                "INSERT OR IGNORE INTO relationships(id, source_id, target_id, relation_type, description) "
                "VALUES (?, ?, ?, ?, ?)",
                (rel_id, src_id, tgt_id, rel_type, rel_desc),
            )
            # Junction table: relationship → wiki_page
            conn.execute(
                "INSERT OR IGNORE INTO relationship_pages(rel_id, wiki_page) VALUES (?, ?)",
                (rel_id, wiki_page),
            )
            inserted_rels += 1
        else:
            # Existing relation: still track it in junction table for this page
            conn.execute(
                "INSERT OR IGNORE INTO relationship_pages(rel_id, wiki_page) VALUES (?, ?)",
                (rel_id, wiki_page),
            )

    if autocommit:
        conn.commit()
    return added, inserted_rels


# ═══════════════════════════════════════════════════════════════════════
#  Graph: Incremental Update
# ═══════════════════════════════════════════════════════════════════════


def update_graph_incremental() -> None:
    """Incremental graph update — only process changed wiki pages."""

    # 1. Health check
    if not _vllm_health_check():
        log("error", " vLLM unreachable", stderr=True)
        sys.exit(1)

    VECTORDB.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    init_db(conn)

    # 2. Load or seed entity registry
    registry = EntityRegistry()
    registry.load_or_seed(conn)

    # 3. Gather all wiki pages + compute body hashes
    pages = gather_wiki_pages()
    changed = []
    unchanged = 0

    for p in pages:
        scope = p.parts[-2]
        slug = p.stem
        wiki_page = f"{scope}/{slug}"

        full_text = p.read_text(encoding="utf-8")
        _, body = strip_frontmatter(full_text)
        current_hash = hashlib.sha256(body.encode()).hexdigest()[:16]

        # Compare with stored hash
        row = conn.execute(
            "SELECT body_hash FROM graph_page_state WHERE wiki_page = ?",
            (wiki_page,),
        ).fetchone()

        if row and row[0] == current_hash:
            unchanged += 1
        else:
            changed.append((p, wiki_page, current_hash))

    log("graph-incr", f"Found {len(changed)} changed pages, {unchanged} unchanged out of {len(pages)} total")

    if not changed:
        log("graph-incr", "No pages changed — skipping graph update")
        conn.close()
        return

    # 4. Process only changed pages
    total_entities_added = 0
    total_relations_added = 0
    total_entities_removed = 0
    total_relations_removed = 0

    for i, (p, wiki_page, body_hash) in enumerate(changed, 1):
        log("graph-incr", f"[{i}/{len(changed)}] Processing {wiki_page}")

        # Atomic transaction: cleanup → LLM extraction → state update.
        # If any step fails, the entire page update rolls back - no data loss.
        page_success = False
        conn.execute("BEGIN IMMEDIATE")
        try:
            # --- STEP 1: Count before removal ---
            old_rels = conn.execute(
                "SELECT COUNT(DISTINCT rp.rel_id) FROM relationship_pages rp "
                "WHERE rp.wiki_page = ?",
                (wiki_page,),
            ).fetchone()[0]

            # Remove mappings for this page
            conn.execute("DELETE FROM entity_pages WHERE wiki_page = ?", (wiki_page,))
            conn.execute("DELETE FROM relationship_pages WHERE wiki_page = ?", (wiki_page,))

            # Clean up orphan entities (no pages reference them anymore)
            orphan_count = conn.execute("""
                SELECT COUNT(*) FROM entities e
                WHERE NOT EXISTS (SELECT 1 FROM entity_pages ep WHERE ep.entity_id = e.id)
            """).fetchone()[0]

            conn.execute("""
                DELETE FROM entities WHERE NOT EXISTS (
                    SELECT 1 FROM entity_pages ep WHERE ep.entity_id = entities.id
                )
            """)

            # Clean up orphan relationships (no pages reference them anymore)
            conn.execute("""
                DELETE FROM relationships WHERE NOT EXISTS (
                    SELECT 1 FROM relationship_pages rp WHERE rp.rel_id = relationships.id
                )
            """)

            total_entities_removed += orphan_count
            total_relations_removed += old_rels

            # --- STEP 2: LLM extraction (inside same transaction, no autocommit) ---
            added, rels = process_page(p, conn, wiki_page=wiki_page, autocommit=False, registry=registry)
            total_entities_added += added
            total_relations_added += rels

            # --- STEP 3: Update page state ---
            conn.execute(
                "INSERT OR REPLACE INTO graph_page_state(wiki_page, body_hash, processed_at) "
                "VALUES (?, ?, datetime('now'))",
                (wiki_page, body_hash),
            )

            conn.commit()
            page_success = True
            log("graph-incr", f"  Done: {wiki_page}")
        except Exception as e:
            conn.rollback()
            log("error", f"  {wiki_page}: {e} (rolled back, will retry next run)", stderr=True)
            tb_str = traceback.format_exc().replace("\n", " | ")
            log("error", f"Traceback: {tb_str}", stderr=True)

    # 5. Persist registry (only after ALL pages committed successfully)
    # This avoids race conditions between registry-write and DB transactions.
    registry.save()

    # 6. Summary
    log("graph-incr", "")
    log("graph-incr", "=== Incremental Graph Update Complete ===")
    log("graph-incr", f"Pages processed: {len(changed)}/{len(pages)}")
    log("graph-incr", f"Entities removed: {total_entities_removed}")
    log("graph-incr", f"Relations removed: {total_relations_removed}")
    log("graph-incr", f"Entities added: {total_entities_added}")
    log("graph-incr", f"Relations added: {total_relations_added}")

    conn.close()


# ═══════════════════════════════════════════════════════════════════════
#  Graph: Build Command
# ═══════════════════════════════════════════════════════════════════════


def build_graph(page_slug: str | None = None, incremental: bool = False) -> None:
    """Build the knowledge graph (entity extraction + relationships).

    Args:
        page_slug: Single page slug (None = full rebuild).
        incremental: If True and page_slug is None, only process changed pages.
    """
    if not _vllm_health_check():
        log("error", " Cannot proceed - vLLM is unreachable. Check if vLLM is running.", stderr=True)
        sys.exit(1)

    VECTORDB.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    init_db(conn)
    _record_graph_migration(conn, "v1_initial_schema")

    if page_slug:
        found = False
        for scope in SCOPES:
            p = WIKI / scope / f"{page_slug}.md"
            if p.exists():
                wiki_page_sp = f"{scope}/{page_slug}"
                added, rels = process_page(p, conn, wiki_page=wiki_page_sp)
                log("done", f" {scope}/{page_slug}: +{added} entities, +{rels} relations")
                # Record page state
                full_text = p.read_text(encoding="utf-8")
                _, body = strip_frontmatter(full_text)
                body_hash = hashlib.sha256(body.encode()).hexdigest()
                conn.execute(
                    "INSERT OR REPLACE INTO graph_page_state(wiki_page, body_hash, processed_at) "
                    "VALUES (?, ?, datetime('now'))",
                    (wiki_page_sp, body_hash),
                )
                conn.commit()
                found = True
                break
        if not found:
            log("error", f" Page '{page_slug}.md' not found in any scope", stderr=True)
            conn.close()
            sys.exit(1)
    elif incremental:
        # Defer to dedicated incremental function
        conn.close()
        update_graph_incremental()
        return
    else:
        log("graph", " clearing existing graph data ...")
        conn.execute("DELETE FROM relationship_pages")
        conn.execute("DELETE FROM entity_pages")
        conn.execute("DELETE FROM graph_page_state")
        conn.execute("DELETE FROM relationships")
        conn.execute("DELETE FROM entities")
        conn.commit()

        total_entities = 0
        total_relations = 0
        skipped = 0
        pages = gather_wiki_pages()

        for i, p in enumerate(pages, 1):
            print(f"[progress] {i}/{len(pages)}", end="\r", flush=True)
            scope_p = p.parts[-2]
            slug_p = p.stem
            wiki_page_p = f"{scope_p}/{slug_p}"
            try:
                added, rels = process_page(p, conn, wiki_page=wiki_page_p)
                total_entities += added
                total_relations += rels
            except Exception as e:
                log("error", f"{p.name}: {e}", stderr=True)
                tb_str = traceback.format_exc().replace("\n", " | ")
                log("error", f"Traceback: {tb_str}", stderr=True)
                skipped += 1

            # Record page state for incremental tracking
            full_text = p.read_text(encoding="utf-8")
            _, body = strip_frontmatter(full_text)
            body_hash = hashlib.sha256(body.encode()).hexdigest()
            conn.execute(
                "INSERT OR REPLACE INTO graph_page_state(wiki_page, body_hash, processed_at) "
                "VALUES (?, ?, datetime('now'))",
                (wiki_page_p, body_hash),
            )

        conn.commit()

        log("graph", f"Done: {total_entities} entities, {total_relations} relations "
              f"from {len(pages)} pages ({skipped} skipped) → {DB_PATH}")

    conn.close()


# ═══════════════════════════════════════════════════════════════════════
#  Graph: Community Detection (Phase 2)
# ═══════════════════════════════════════════════════════════════════════


def _embed_single(text: str) -> Optional[bytes]:
    """Get float32 embedding BLOB for a single text via Ollama bge-m3."""
    try:
        vecs = embed([text])
        if vecs.size > 0 and np.any(vecs[0]):
            return vecs[0].tobytes()
        return None
    except Exception as e:
        log("warn", f"  [warn] embedding failed: {e}", stderr=True)
        return None


def _ollama_community_summary(
    entity_rows: List[Tuple[str, str, str]],
    rel_rows: List[Tuple[str, str, str]],
) -> Tuple[str, str]:
    """Call vLLM for a community summary. Returns (label, summary)."""
    entity_lines = []
    for eid, elabel, etype in entity_rows:
        entity_lines.append(f"  - {elabel} [{etype}]")
    entity_list = "\n".join(entity_lines)

    rel_lines = []
    for src, tgt, rtype in rel_rows:
        rel_lines.append(f"  - {src} -{rtype}→ {tgt}")
    rel_list = "\n".join(rel_lines)

    user_prompt = f"""\
Entities ({len(entity_rows)}):
{entity_list}

Beziehungen ({len(rel_rows)}):
{rel_list}

Erstelle Label und Summary als JSON."""

    payload = {
        "model": SUMMARY_MODEL,
        "messages": [
            {"role": "system", "content": COMMUNITY_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 512,
    }

    last_err: Optional[Exception] = None
    for attempt in range(RETRY_MAX):
        try:
            req = urllib.request.Request(
                f"{VLLM_URL}/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            msg = body["choices"][0]["message"]
            # qwen3.6 schiebt Output manchmal ins "reasoning"-Field statt "content"
            raw = msg.get("content") or msg.get("reasoning") or ""

            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
                cleaned = re.sub(r"\s*```$", "", cleaned)
            result = json.loads(cleaned)
            return result.get("label", "Community"), result.get("summary", "")

        except Exception as e:
            last_err = e
            log("warn",
                f"community summary vLLM error (attempt {attempt+1}/{RETRY_MAX}): {e}",
                stderr=True)
            if attempt < RETRY_MAX - 1:
                # Auto-fix attempt on JSON-parse errors
                if isinstance(e, json.JSONDecodeError) and 'raw' in locals() and raw:
                    fixed = _fix_json(raw)
                    if fixed:
                        try:
                            result = json.loads(fixed)
                            if isinstance(result, dict):
                                return result.get("label", "Community"), result.get("summary", "")
                        except json.JSONDecodeError:
                            pass
                time.sleep(RETRY_BASE * (2 ** attempt))

    log("warn",
        f"community summary vLLM failed after {RETRY_MAX} attempts, using fallback: {last_err}",
        stderr=True)
    types = [t for _, _, t in entity_rows]
    dominant = max(set(types), key=types.count) if types else "CONCEPT"
    return f"{dominant}-Gruppe", f"Community von {len(entity_rows)} Entities, hauptsächlich {dominant}."


def _build_igraph(conn: sqlite3.Connection):
    """Load entities + relationships from SQLite into an in-memory igraph."""
    entities = conn.execute(
        "SELECT id, label, entity_type FROM entities"
    ).fetchall()
    relationships = conn.execute(
        "SELECT source_id, target_id, relation_type FROM relationships"
    ).fetchall()

    g = ig.Graph()
    g.add_vertices(len(entities))
    for i, (eid, label, etype) in enumerate(entities):
        g.vs[i]["id"] = eid
        g.vs[i]["label"] = label
        g.vs[i]["type"] = etype

    vertex_map = {e[0]: i for i, e in enumerate(entities)}
    for src, tgt, rel_type in relationships:
        if src in vertex_map and tgt in vertex_map:
            g.add_edge(vertex_map[src], vertex_map[tgt], type=rel_type)

    return g


def _needs_community_rebuild(conn: sqlite3.Connection) -> bool:
    """Check if entities/relationships changed significantly since last community build."""
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    if "communities" not in tables:
        return True

    has_communities = conn.execute("SELECT COUNT(*) FROM communities").fetchone()[0]
    if has_communities == 0:
        return True

    current_entities = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    tracked_entities = conn.execute(
        "SELECT COUNT(DISTINCT entity_id) FROM community_members"
    ).fetchone()[0]

    if current_entities - tracked_entities > 5:
        return True

    new_rels = conn.execute(
        "SELECT COUNT(*) FROM relationships WHERE created_at > datetime('now', '-1 day')"
    ).fetchone()[0]
    if new_rels > 5:
        return True

    return False


def _detect_communities(
    conn: sqlite3.Connection,
    incremental: bool = False,
) -> int:
    """Run Leiden community detection, store results in SQLite.
    Returns number of communities created."""
    if incremental:
        needs_rebuild = _needs_community_rebuild(conn)
        if not needs_rebuild:
            log("communities", " No significant changes detected - skipping rebuild.")
            return 0
        else:
            log("communities", " Significant changes detected - rebuilding.")

    conn.execute("DELETE FROM community_members")
    conn.execute("DELETE FROM communities")
    conn.commit()

    g = _build_igraph(conn)
    n_verts = len(g.vs)
    n_edges = len(g.es)
    log("communities", f" Graph: {n_verts} vertices, {n_edges} edges")

    if n_verts == 0:
        log("communities", " Empty graph - nothing to cluster.")
        return 0

    result = g.community_leiden(objective_function="modularity")
    comm_count = 0

    for idx, vertex_indices in enumerate(result):
        comm_id = f"level-0-idx-{idx}"
        member_ids = [g.vs[vi]["id"] for vi in vertex_indices]

        if not member_ids:
            continue

        # Skip small communities (<5 entities) - not worth the LLM cost
        if len(member_ids) < 5:
            log("community", f"  Skipping {comm_id}: {len(member_ids)} entities (<5 threshold)")
            continue

        entity_rows = conn.execute(
            "SELECT id, label, entity_type FROM entities WHERE id IN ("
            + ",".join("?" * len(member_ids)) + ")",
            member_ids,
        ).fetchall()

        rel_rows = conn.execute(
            "SELECT e1.label, e2.label, r.relation_type "
            "FROM relationships r "
            "JOIN entities e1 ON r.source_id = e1.id "
            "JOIN entities e2 ON r.target_id = e2.id "
            "WHERE r.source_id IN (" + ",".join("?" * len(member_ids)) + ") "
            "AND r.target_id IN (" + ",".join("?" * len(member_ids)) + ")",
            member_ids + member_ids,
        ).fetchall()

        label, summary = _ollama_community_summary(entity_rows, rel_rows)
        log("community", f"  [community] {comm_id}: {label} ({len(member_ids)} entities)")
        time.sleep(RATE_LIMIT_S)

        embedding_blob = _embed_single(summary)

        conn.execute(
            "INSERT INTO communities(id, level, label, summary, summary_embedding, entity_count) "
            "VALUES (?, 0, ?, ?, ?, ?)",
            (comm_id, label, summary, embedding_blob, len(member_ids)),
        )

        for eid in member_ids:
            conn.execute(
                "INSERT INTO community_members(community_id, entity_id) VALUES (?, ?)",
                (comm_id, eid),
            )
        comm_count += 1

    conn.commit()
    _record_graph_migration(conn, "v2_community_detection")
    conn.commit()

    log("communities", f" Done: {comm_count} communities created.")
    return comm_count


# ═══════════════════════════════════════════════════════════════════════
#  Unified Build (vectors + graph)
# ═══════════════════════════════════════════════════════════════════════


def build(run_graph: bool = False, page_slug: str | None = None,
          run_communities: bool = False, communities_incremental: bool = False,
          graph_incremental: bool = False) -> int:
    """Full build pipeline: vector index → graph extraction → community detection.

    Args:
        run_graph: If True, also build the knowledge graph after vector index.
        page_slug: If set, only process this single page (graph only, not vectors).
        run_communities: If True, also run community detection after graph build.
        communities_incremental: Skip community rebuild if ≤5 new entities.
        graph_incremental: If True, only process changed wiki pages (incremental graph).

    Returns number of vector chunks inserted (0 if graph-only build).
    """
    VECTORDB.mkdir(exist_ok=True)

    # Phase 1: Vector index
    if not page_slug:
        inserted = build_vectors()
    else:
        # For single-page graph updates, still ensure DB schema exists
        conn = sqlite3.connect(DB_PATH)
        init_db(conn)
        conn.close()
        inserted = 0

    # Phase 2: Knowledge Graph
    if run_graph:
        log("info", "\n" + "=" * 60)
        log("info", "PHASE 2: Knowledge Graph Build")
        log("info", "=" * 60)
        build_graph(page_slug=page_slug, incremental=graph_incremental)

        # Phase 3: Community Detection
        if run_communities:
            log("info", "\n" + "=" * 60)
            log("info", "PHASE 3: Community Detection")
            log("info", "=" * 60)
            if not IGRAPH_AVAILABLE:
                log("error",
                    "python-igraph is not installed.\n"
                    "  Install it with:\n"
                    "    pip install python-igraph",
                    stderr=True)
                sys.exit(1)

            if not _vllm_health_check():
                log("error", " vLLM is unreachable - needed for community summaries.", stderr=True)
                sys.exit(1)

            conn = sqlite3.connect(DB_PATH)
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            init_db(conn)
            try:
                _detect_communities(conn, incremental=communities_incremental)
            finally:
                conn.close()

    return inserted


# ═══════════════════════════════════════════════════════════════════════
#  Search & Stats
# ═══════════════════════════════════════════════════════════════════════


def load_index(allowed_scopes: list[str] | None = None) -> tuple[np.ndarray, list[dict]]:
    conn = sqlite3.connect(DB_PATH)
    if allowed_scopes:
        placeholders = ",".join("?" * len(allowed_scopes))
        rows = conn.execute(
            f"SELECT id, scope, kind, ref, section, chunk_idx, content, content_hash, embedding "
            f"FROM chunks WHERE scope IN ({placeholders})",
            allowed_scopes,
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, scope, kind, ref, section, chunk_idx, content, content_hash, embedding FROM chunks"
        ).fetchall()
    conn.close()
    if not rows:
        return np.zeros((0, EMBED_DIMS), dtype=np.float32), []
    vecs = np.stack([np.frombuffer(r[8], dtype=np.float32) for r in rows])
    meta = [
        {
            "id": r[0],
            "scope": r[1],
            "kind": r[2],
            "ref": r[3],
            "section": r[4],
            "chunk_idx": r[5],
            "content": r[6],
        }
        for r in rows
    ]
    return vecs, meta


def resolve_scopes_from_session_key(session_key: str) -> tuple[list[str], str]:
    """Returns (allowed_scopes, matched_entry_name)."""
    if not SCOPES_CONFIG.exists():
        raise SystemExit(f"scopes config not found at {SCOPES_CONFIG}")
    cfg = json.loads(SCOPES_CONFIG.read_text(encoding="utf-8"))
    if session_key:
        for entry in cfg.get("entries", []):
            for pattern in entry.get("sessionKeyPatterns", []):
                if pattern in session_key:
                    return list(entry.get("scopes") or []), str(entry.get("name", "?"))
    default = cfg.get("default") or {"scopes": ["public"]}
    return list(default.get("scopes") or ["public"]), "default"


def search(
    query: str,
    k: int = 5,
    allowed_scopes: list[str] | None = None,
) -> list[dict]:
    if not DB_PATH.exists():
        raise SystemExit(f"no index at {DB_PATH} - run `vectordb.py build` first")
    vecs, meta = load_index(allowed_scopes)
    if len(meta) == 0:
        return []
    q = embed([query])[0]
    scores = vecs @ q
    top = np.argsort(-scores)[:k]
    return [{**meta[i], "score": float(scores[i])} for i in top]


def stats() -> dict:
    if not DB_PATH.exists():
        return {"db_path": str(DB_PATH), "exists": False}
    conn = sqlite3.connect(DB_PATH)
    total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    by_scope = dict(
        conn.execute("SELECT scope, COUNT(*) FROM chunks GROUP BY scope").fetchall()
    )
    by_kind = dict(
        conn.execute("SELECT kind, COUNT(*) FROM chunks GROUP BY kind").fetchall()
    )
    refs = conn.execute(
        "SELECT DISTINCT scope, kind, ref FROM chunks ORDER BY scope, kind, ref"
    ).fetchall()
    conn.close()
    return {
        "db_path": str(DB_PATH),
        "chunks": total,
        "by_scope": by_scope,
        "by_kind": by_kind,
        "refs": [f"{s}/{k}:{r}" for s, k, r in refs],
    }


# ═══════════════════════════════════════════════════════════════════════
#  Graph Commands (direct - no subprocess delegation)
# ═══════════════════════════════════════════════════════════════════════


def _graph_has_tables() -> bool:
    """Check whether graph tables exist in the DB."""
    if not DB_PATH.exists():
        return False
    conn = sqlite3.connect(DB_PATH)
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    conn.close()
    return "entities" in tables


def cmd_graph_stats() -> dict:
    """Return graph statistics as a dict."""
    if not DB_PATH.exists():
        return {"db_path": str(DB_PATH), "exists": False}

    conn = sqlite3.connect(DB_PATH)
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    if "entities" not in tables:
        conn.close()
        return {"db_path": str(DB_PATH), "exists": True, "entities_table": False,
                 "message": "Graph tables not yet created. Run `vectordb.py graph build` first."}

    total_entities = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    total_relations = conn.execute("SELECT COUNT(*) FROM relationships").fetchone()[0]

    by_type = dict(conn.execute(
        "SELECT entity_type, COUNT(*) FROM entities GROUP BY entity_type ORDER BY COUNT(*) DESC"
    ).fetchall())

    by_rel_type = dict(conn.execute(
        "SELECT relation_type, COUNT(*) FROM relationships GROUP BY relation_type ORDER BY COUNT(*) DESC"
    ).fetchall())

    orphans = conn.execute(
        """
        SELECT e.id, e.label, e.entity_type
        FROM entities e
        WHERE e.id NOT IN (SELECT source_id FROM relationships)
          AND e.id NOT IN (SELECT target_id FROM relationships)
        ORDER BY e.label
        """
    ).fetchall()

    top_connected = conn.execute(
        """
        SELECT e.id, e.label, COUNT(r.id) AS degree
        FROM entities e
        JOIN relationships r ON (r.source_id = e.id OR r.target_id = e.id)
        GROUP BY e.id
        ORDER BY degree DESC
        LIMIT 10
        """
    ).fetchall()

    pages = conn.execute(
        "SELECT COUNT(DISTINCT wiki_page) FROM entities WHERE wiki_page IS NOT NULL"
    ).fetchone()[0]

    conn.close()

    return {
        "db_path": str(DB_PATH),
        "entities": total_entities,
        "relationships": total_relations,
        "by_entity_type": by_type,
        "by_relation_type": by_rel_type,
        "orphans": len(orphans),
        "orphan_details": [{"id": o[0], "label": o[1], "type": o[2]} for o in orphans[:20]],
        "top_connected": [{"id": t[0], "label": t[1], "degree": t[2]} for t in top_connected],
        "wiki_pages_represented": pages,
    }


def cmd_graph_validate() -> dict:
    """Validate the graph: find orphans, duplicates, issues."""
    if not DB_PATH.exists():
        return {"error": "Database not found"}

    conn = sqlite3.connect(DB_PATH)
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    if "entities" not in tables:
        conn.close()
        return {"error": "Graph tables not created yet"}

    issues: List[dict] = []

    orphans = conn.execute(
        """
        SELECT e.id, e.label, e.entity_type, e.wiki_page
        FROM entities e
        WHERE e.id NOT IN (SELECT source_id FROM relationships)
          AND e.id NOT IN (SELECT target_id FROM relationships)
        ORDER BY e.label
        """
    ).fetchall()
    if orphans:
        issues.append({
            "issue": "orphan_entities",
            "count": len(orphans),
            "details": [{"id": o[0], "label": o[1], "type": o[2], "page": o[3]} for o in orphans[:30]],
        })

    all_entities = conn.execute(
        "SELECT id, label, entity_type FROM entities ORDER BY entity_type, label"
    ).fetchall()
    type_groups: Dict[str, List[Tuple[str, str, str]]] = {}
    for e in all_entities:
        type_groups.setdefault(e[2], []).append(e)
    dup_pairs = []
    for etype, group in type_groups.items():
        if len(group) > 200:
            log("warn", f"  [warn] Skipping duplicate check for type '{etype}' ({len(group)} entities, limit 200)", stderr=True)
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                score = _fuzzy_match_score(all_entities[i][1], all_entities[j][1])
                if score >= FUZZY_THRESHOLD:
                    dup_pairs.append({
                        "a": all_entities[i][1],
                        "b": all_entities[j][1],
                        "type": all_entities[i][2],
                        "score": score,
                    })
    if dup_pairs:
        issues.append({
            "issue": "potential_duplicates",
            "count": len(dup_pairs),
            "details": dup_pairs[:30],
        })

    broken = conn.execute(
        """
        SELECT r.id, r.source_id, r.target_id, r.relation_type
        FROM relationships r
        LEFT JOIN entities es ON r.source_id = es.id
        LEFT JOIN entities et ON r.target_id = et.id
        WHERE es.id IS NULL OR et.id IS NULL
        """
    ).fetchall()
    if broken:
        issues.append({
            "issue": "broken_references",
            "count": len(broken),
            "details": [{"rel_id": b[0], "source": b[1], "target": b[2], "type": b[3]} for b in broken],
        })

    loops = conn.execute(
        "SELECT id, source_id, relation_type FROM relationships WHERE source_id = target_id"
    ).fetchall()
    if loops:
        issues.append({
            "issue": "self_loops",
            "count": len(loops),
            "details": [{"rel_id": l[0], "entity": l[1], "type": l[2]} for l in loops],
        })

    conn.close()

    return {
        "issues_found": len(issues),
        "issues": issues,
        "healthy": len(issues) == 0,
    }


def _entities_to_json(conn: sqlite3.Connection, rows) -> list[dict]:
    """Build the graph JSON shape (entity + 1-hop relations) for rows of
    (id, label, entity_type, description, wiki_page). Shared by graph search/pages."""
    result = []
    for eid, label, etype, desc, page in rows:
        rels_out = conn.execute(
            "SELECT r.relation_type, r.description, e.label "
            "FROM relationships r JOIN entities e ON r.target_id = e.id "
            "WHERE r.source_id = ?",
            (eid,),
        ).fetchall()
        rels_in = conn.execute(
            "SELECT r.relation_type, r.description, e.label "
            "FROM relationships r JOIN entities e ON r.source_id = e.id "
            "WHERE r.target_id = ?",
            (eid,),
        ).fetchall()
        result.append({
            "label": label,
            "type": etype,
            "description": desc or "",
            "wiki_page": page or "",
            "outgoing": [
                {"relation_type": rt, "target": tl, "description": rd or ""}
                for rt, rd, tl in rels_out[:10]
            ],
            "incoming": [
                {"relation_type": rt, "source": sl, "description": rd or ""}
                for rt, rd, sl in rels_in[:10]
            ],
        })
    return result


def cmd_graph_pages(pages: list[str], as_json: bool = False, scopes: list[str] | None = None) -> None:
    """Vector→graph bridge: return entities (+1-hop relations) anchored on a set of
    wiki pages, ordered by the priority of the pages given (i.e. vector-search rank).

    Args:
        pages: wiki_page values ("scope/slug") to fetch entities for, highest-priority first.
        as_json: if True, emit a JSON array (same shape as `graph search`).
        scopes: optional scope allow-list (defensive; pages already carry their scope).
    """
    if not _graph_has_tables():
        if as_json:
            print(json.dumps([], ensure_ascii=False))
        else:
            log("error", " Graph not built yet. Run `vectordb.py graph build` first.", stderr=True)
        return

    if scopes:
        allowed_prefixes = tuple(f"{s}/" for s in scopes)
        pages = [p for p in pages if p and p.startswith(allowed_prefixes)]

    # Dedupe, preserving the incoming (vector-rank) order
    seen: set[str] = set()
    ordered: list[str] = []
    for p in pages:
        if p and p not in seen:
            seen.add(p)
            ordered.append(p)

    conn = sqlite3.connect(DB_PATH)
    rows = []
    if ordered:
        placeholders = ",".join("?" * len(ordered))
        fetched = conn.execute(
            f"SELECT id, label, entity_type, description, wiki_page FROM entities "
            f"WHERE wiki_page IN ({placeholders})",
            ordered,
        ).fetchall()
        # Keep entities in the priority order of their source page
        rank = {p: i for i, p in enumerate(ordered)}
        fetched.sort(key=lambda r: rank.get(r[4], 1_000_000))
        rows = fetched

    if as_json:
        result = _entities_to_json(conn, rows)
        conn.close()
        print(json.dumps(result, ensure_ascii=False))
    else:
        log("info", f"Found {len(rows)} entity/entities on {len(ordered)} page(s):")
        for eid, label, etype, desc, page in rows:
            log("info", f"  ● {label} [{etype}]  ({page})")
        conn.close()


def cmd_graph_search(query: str, as_json: bool = False, scopes: list[str] | None = None) -> None:
    """Simple graph lookup: find entities matching query + their 1-hop relations.

    Args:
        query: Text to match against entity labels and descriptions.
        as_json: If True, emit a JSON array to stdout instead of log-lines.
        scopes: Optional list of scope prefixes (e.g. ["private", "family"])
                to filter entities by their wiki_page field.
    """
    if not _graph_has_tables():
        if as_json:
            print(json.dumps([], ensure_ascii=False))
        else:
            log("error", " Graph not built yet. Run `vectordb.py graph build` first.", stderr=True)
        return

    conn = sqlite3.connect(DB_PATH)
    like = f"%{query}%"
    rows = conn.execute(
        "SELECT id, label, entity_type, description, wiki_page FROM entities "
        "WHERE label LIKE ? COLLATE NOCASE OR description LIKE ? COLLATE NOCASE",
        (like, like),
    ).fetchall()

    # Scope filter: wiki_page is always "scope/slug"
    if scopes:
        scope_prefixes = tuple(f"{s}/" for s in scopes)
        rows = [r for r in rows if r[4] and r[4].startswith(scope_prefixes)]

    if not rows:
        conn.close()
        if as_json:
            print(json.dumps([], ensure_ascii=False))
        else:
            log("info", f"No entities matching '{query}'")
        return

    if as_json:
        result = _entities_to_json(conn, rows)
        conn.close()
        print(json.dumps(result, ensure_ascii=False))
    else:
        log("info", f"Found {len(rows)} entity/entities matching '{query}':")

        for eid, label, etype, desc, page in rows:
            log("info", f"  \u25cf {label} [{etype}]")
            if desc:
                log("info", f"    {desc}")
            if page:
                log("info", f"    page: {page}")

            rels_out = conn.execute(
                "SELECT r.relation_type, r.description, e.label "
                "FROM relationships r JOIN entities e ON r.target_id = e.id "
                "WHERE r.source_id = ?",
                (eid,),
            ).fetchall()

            rels_in = conn.execute(
                "SELECT r.relation_type, r.description, e.label "
                "FROM relationships r JOIN entities e ON r.source_id = e.id "
                "WHERE r.target_id = ?",
                (eid,),
            ).fetchall()

            if rels_out:
                log("info", f"    \u2192 outgoing ({len(rels_out)}):")
                for rtype, rdesc, target_label in rels_out[:10]:
                    log("info", f"      \u2514\u2500 {rtype} \u2192 {target_label}")
                    if rdesc:
                        log("info", f"         {rdesc}")

            if rels_in:
                log("info", f"    \u2190 incoming ({len(rels_in)}):")
                for rtype, rdesc, source_label in rels_in[:10]:
                    log("info", f"      \u2514\u2500 {source_label} {rtype}")
                    if rdesc:
                        log("info", f"         {rdesc}")

            log("info", "")

        conn.close()



        rels_out = conn.execute(
            "SELECT r.relation_type, r.description, e.label "
            "FROM relationships r JOIN entities e ON r.target_id = e.id "
            "WHERE r.source_id = ?",
            (eid,),
        ).fetchall()

        rels_in = conn.execute(
            "SELECT r.relation_type, r.description, e.label "
            "FROM relationships r JOIN entities e ON r.source_id = e.id "
            "WHERE r.target_id = ?",
            (eid,),
        ).fetchall()

        if rels_out:
            log("info", f"    → outgoing ({len(rels_out)}):")
            for rtype, rdesc, target_label in rels_out[:10]:
                log("info", f"      └─ {rtype} → {target_label}")
                if rdesc:
                    log("info", f"         {rdesc}")

        if rels_in:
            log("info", f"    ← incoming ({len(rels_in)}):")
            for rtype, rdesc, source_label in rels_in[:10]:
                log("info", f"      └─ {source_label} {rtype}")
                if rdesc:
                    log("info", f"         {rdesc}")

        log("info", "")

    conn.close()


# ═══════════════════════════════════════════════════════════════════════
#  Community Commands
# ═══════════════════════════════════════════════════════════════════════


def cmd_communities_build(incremental: bool = False) -> None:
    """Build community detection + summaries."""
    if not IGRAPH_AVAILABLE:
        log("error",
            "python-igraph is not installed.\n"
            "  Install it with:\n"
            "    pip install python-igraph\n"
            "  Phase 1 commands (graph build, stats, validate) still work without it.",
            stderr=True)
        sys.exit(1)

    if not DB_PATH.exists():
        log("error", " Database not found. Run `vectordb.py graph build` first.", stderr=True)
        sys.exit(1)

    if not _vllm_health_check():
        log("error", " vLLM is unreachable - needed for community summaries.", stderr=True)
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    init_db(conn)

    try:
        _detect_communities(conn, incremental=incremental)
    finally:
        conn.close()


def cmd_communities_stats() -> dict:
    """Return community statistics."""
    if not DB_PATH.exists():
        return {"db_path": str(DB_PATH), "exists": False}

    conn = sqlite3.connect(DB_PATH)
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    if "communities" not in tables:
        conn.close()
        return {"db_path": str(DB_PATH), "communities_table": False,
                 "message": "Communities not yet built. Run `vectordb.py graph communities build` first."}

    total_communities = conn.execute("SELECT COUNT(*) FROM communities").fetchone()[0]
    by_level = dict(conn.execute(
        "SELECT level, COUNT(*) FROM communities GROUP BY level ORDER BY level"
    ).fetchall())

    top_communities = conn.execute(
        "SELECT id, level, label, entity_count FROM communities "
        "ORDER BY entity_count DESC LIMIT 10"
    ).fetchall()

    total_memberships = conn.execute(
        "SELECT COUNT(*) FROM community_members"
    ).fetchone()[0]

    conn.close()

    return {
        "db_path": str(DB_PATH),
        "total_communities": total_communities,
        "by_level": by_level,
        "total_memberships": total_memberships,
        "top_communities": [
            {"id": c[0], "level": c[1], "label": c[2], "entity_count": c[3]}
            for c in top_communities
        ],
    }


def cmd_communities_list(level: Optional[int] = None) -> None:
    """List communities as readable text."""
    if not DB_PATH.exists():
        log("error", " Database not found.", stderr=True)
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    if "communities" not in tables:
        conn.close()
        log("info", "Communities not yet built. Run `vectordb.py graph communities build` first.")
        return

    query = "SELECT id, level, label, entity_count, summary FROM communities"
    params: Tuple = ()
    if level is not None:
        query += " WHERE level = ?"
        params = (level,)
    query += " ORDER BY level, entity_count DESC"

    communities = conn.execute(query, params).fetchall()

    if not communities:
        conn.close()
        log("info", "No communities found.")
        return

    log("info", f"Communities ({len(communities)} total):")
    for cid, clevel, clabel, ccount, csummary in communities:
        level_label = {0: "leaf", 1: "parent", 2: "root"}.get(clevel, f"level-{clevel}")
        log("{level_label}", f"  [{level_label}] {clabel} ({ccount} entities)")
        if csummary:
            log("info", f"         {csummary[:120]}")
        log("info", f"         ID: {cid}")
        log("info", "")

    conn.close()


def cmd_communities_show(community_id: str) -> None:
    """Show detailed info for a single community."""
    if not DB_PATH.exists():
        log("error", " Database not found.", stderr=True)
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    if "communities" not in tables:
        conn.close()
        log("info", "Communities not yet built.")
        return

    comm = conn.execute(
        "SELECT id, level, label, summary, entity_count FROM communities WHERE id = ?",
        (community_id,),
    ).fetchone()

    if not comm:
        conn.close()
        log("info", f"Community '{community_id}' not found.")
        return

    cid, clevel, clabel, csummary, ccount = comm
    level_label = {0: "leaf", 1: "parent", 2: "root"}.get(clevel, f"level-{clevel}")

    log("info", f"{'='*60}")
    log("info", f"Community: {clabel}")
    log("info", f"ID: {cid}")
    log("info", f"Level: {level_label} ({clevel})")
    log("info", f"Entities: {ccount}")
    if csummary:
        log("info", f"Summary:")
        log("info", f"  {csummary}")

    members = conn.execute(
        "SELECT e.id, e.label, e.entity_type, e.description "
        "FROM community_members cm "
        "JOIN entities e ON cm.entity_id = e.id "
        "WHERE cm.community_id = ? "
        "ORDER BY e.label",
        (community_id,),
    ).fetchall()

    if members:
        log("info", f"Members ({len(members)}):")
        for mid, mlabel, mtype, mdesc in members:
            log("info", f"  ● {mlabel} [{mtype}]")
            if mdesc:
                log("info", f"    {mdesc}")

    member_ids = [m[0] for m in members]
    if member_ids:
        placeholders = ",".join("?" * len(member_ids))
        rels = conn.execute(
            f"SELECT r.relation_type, e1.label, e2.label, r.description "
            f"FROM relationships r "
            f"JOIN entities e1 ON r.source_id = e1.id "
            f"JOIN entities e2 ON r.target_id = e2.id "
            f"WHERE r.source_id IN ({placeholders}) "
            f"AND r.target_id IN ({placeholders}) "
            f"ORDER BY r.relation_type",
            member_ids + member_ids,
        ).fetchall()

        if rels:
            log("info", f"Internal Relationships ({len(rels)}):")
            for rtype, src, tgt, rdesc in rels:
                log("info", f"  {src} -{rtype}→ {tgt}")
                if rdesc:
                    log("info", f"    {rdesc}")

    log("info", f"{'='*60}")
    conn.close()


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════


def parse_scopes_arg(value: str | None) -> list[str] | None:
    if not value:
        return None
    parts = [v.strip() for v in value.split(",") if v.strip()]
    bad = [p for p in parts if p not in SCOPES]
    if bad:
        raise SystemExit(f"unknown scope(s): {bad} (valid: {list(SCOPES)})")
    return parts


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Wiki Vector Index + Knowledge Graph - search, build, analyze"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    # ── Vector commands ──
    sp_build = sub.add_parser("build", help="Build vector index (optionally + graph)")
    sp_build.add_argument("--graph", action="store_true",
                          help="Also build knowledge graph after vector index")
    sp_build.add_argument("--graph-incremental", action="store_true",
                          help="Incremental graph update (only changed pages)")
    sp_build.add_argument("--page", help="Process single page (graph only, skips vector rebuild)")
    sp_build.add_argument("--communities", action="store_true",
                          help="Also run community detection (requires --graph)")
    sp_build.add_argument("--communities-incremental", action="store_true",
                          help="Skip community rebuild if ≤5 new entities")

    sp_search = sub.add_parser("search", help="Search the vector index")
    sp_search.add_argument("query")
    sp_search.add_argument("-k", type=int, default=5)
    sp_search.add_argument("--json", action="store_true")
    scopes_group = sp_search.add_mutually_exclusive_group()
    scopes_group.add_argument(
        "--session-key",
        help="sessionKey from active-memory; resolved via config/scopes.json",
    )
    scopes_group.add_argument(
        "--scopes",
        help="Comma-separated list of scopes to search (default: all). Ignored when --session-key is set.",
    )
    sub.add_parser("stats", help="Show index statistics")

    # ── Graph commands ──
    sp_graph = sub.add_parser("graph", help="Knowledge graph operations")
    sp_graph.add_argument("--incremental", action="store_true",
                          help="Incremental graph update (only changed pages)")
    graph_sub = sp_graph.add_subparsers(dest="graph_cmd")

    # graph build
    sp_gb = graph_sub.add_parser("build", help="Build/rebuild the knowledge graph")
    sp_gb.add_argument("--page", help="Process a single page by slug (incremental)")
    sp_gb.add_argument("--incremental", action="store_true",
                       help="Incremental graph update (only changed pages)")
    sp_gb.add_argument("--communities", action="store_true",
                       help="Also run community detection after graph build")
    sp_gb.add_argument("--communities-incremental", action="store_true",
                       help="Skip community rebuild if ≤5 new entities")

    # graph stats
    graph_sub.add_parser("stats", help="Show graph statistics")

    # graph validate
    graph_sub.add_parser("validate", help="Validate graph integrity")

    # graph search
    sp_gs = graph_sub.add_parser("search", help="Search the knowledge graph")
    sp_gs.add_argument("query")
    sp_gs.add_argument("--json", action="store_true", help="Output as JSON")
    sp_gs.add_argument("--scopes", type=str, default=None,
        help="Comma-separated list of scopes to filter by")

    # graph pages (vector→graph bridge: entities anchored on given wiki pages)
    sp_gp = graph_sub.add_parser("pages", help="Entities (+1-hop) for given wiki pages")
    sp_gp.add_argument("--pages", type=str, required=True,
        help="Comma-separated wiki_page refs ('scope/slug'), highest-priority first")
    sp_gp.add_argument("--json", action="store_true", help="Output as JSON")
    sp_gp.add_argument("--scopes", type=str, default=None,
        help="Comma-separated scope allow-list (defensive filter)")

    # graph communities
    sp_gc = graph_sub.add_parser("communities", help="Community detection & summaries (Phase 2)")
    comm_sub = sp_gc.add_subparsers(dest="comm_cmd")

    sp_cbuild = comm_sub.add_parser("build", help="Build communities + LLM summaries")
    sp_cbuild.add_argument("--incremental", action="store_true",
                           help="Skip rebuild if ≤5 new entities")

    comm_sub.add_parser("stats", help="Community statistics")

    sp_clist = comm_sub.add_parser("list", help="List communities")
    sp_clist.add_argument("--level", type=int, default=None,
                          help="Filter by hierarchy level (0=leaf, 1=parent)")

    sp_cshow = comm_sub.add_parser("show", help="Show a single community")
    sp_cshow.add_argument("id", help="Community ID (e.g. level-0-idx-0)")

    args = ap.parse_args()

    # ── Dispatch ──
    if args.cmd == "build":
        graph_inc = getattr(args, "graph_incremental", False)
        build(
            run_graph=getattr(args, "graph", False) or graph_inc,
            page_slug=getattr(args, "page", None),
            run_communities=getattr(args, "communities", False),
            communities_incremental=getattr(args, "communities_incremental", False),
            graph_incremental=graph_inc,
        )

    elif args.cmd == "search":
        allowed: list[str] | None
        source = "all"
        if args.session_key is not None:
            allowed, name = resolve_scopes_from_session_key(args.session_key)
            source = f"session-key:{name}"
        else:
            allowed = parse_scopes_arg(args.scopes)
            if allowed is None:
                source = "all"
            else:
                source = "scopes-flag"
        results = search(args.query, args.k, allowed)
        if args.json:
            print(json.dumps(results, indent=2, ensure_ascii=False))
        else:
            scope_label = ",".join(allowed) if allowed else "all"
            log("info", f"# query scope: {scope_label} (via {source})")
            for r in results:
                preview = r["content"][:160].replace("\n", " ")
                log("info", f"[{r['score']:.3f}] {r['scope']}/{r['kind']}:{r['ref']} § {r['section']}")
                log("info", f"        {preview}...")

    elif args.cmd == "stats":
        print(json.dumps(stats(), indent=2, ensure_ascii=False))

    elif args.cmd == "graph":
        gc = getattr(args, "graph_cmd", None)
        graph_inc_top = getattr(args, "incremental", False)
        if gc is None and graph_inc_top:
            # Shorthand: `graph --incremental` → direct incremental graph update (skip vectors)
            build_graph(page_slug=None, incremental=True)
        elif gc == "build":
            page_slug = getattr(args, "page", None)
            run_comm = getattr(args, "communities", False)
            comm_inc = getattr(args, "communities_incremental", False)
            graph_inc = getattr(args, "incremental", False)
            build(run_graph=True, page_slug=page_slug,
                  run_communities=run_comm, communities_incremental=comm_inc,
                  graph_incremental=graph_inc)
        elif gc == "stats":
            result = cmd_graph_stats()
            print(json.dumps(result, indent=2, ensure_ascii=False))
        elif gc == "validate":
            result = cmd_graph_validate()
            print(json.dumps(result, indent=2, ensure_ascii=False))
        elif gc == "search":
            scopes = None
            if getattr(args, "scopes", None):
                scopes = [s.strip() for s in args.scopes.split(",")]
            cmd_graph_search(args.query, as_json=getattr(args, "json", False), scopes=scopes)
        elif gc == "pages":
            scopes = None
            if getattr(args, "scopes", None):
                scopes = [s.strip() for s in args.scopes.split(",")]
            pages = [p.strip() for p in args.pages.split(",") if p.strip()]
            cmd_graph_pages(pages, as_json=getattr(args, "json", False), scopes=scopes)
        elif gc == "communities":
            cc = getattr(args, "comm_cmd", None)
            if cc == "build":
                cmd_communities_build(incremental=getattr(args, "incremental", False))
            elif cc == "stats":
                result = cmd_communities_stats()
                print(json.dumps(result, indent=2, ensure_ascii=False))
            elif cc == "list":
                cmd_communities_list(level=getattr(args, "level", None))
            elif cc == "show":
                cmd_communities_show(args.id)
            else:
                ap.print_help()
        else:
            ap.print_help()

    return 0


if __name__ == "__main__":
    sys.exit(main())

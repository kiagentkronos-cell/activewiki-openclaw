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
import html
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

# JSON-Schema für Community-Summary (vermeidet vLLM json_object Bugs)
_SCHEMA_COMMUNITY_SUMMARY = {
    'type': 'json_schema',
    'json_schema': {
        'name': 'community_summary',
        'strict': False,
        'schema': {
            'type': 'object',
            'properties': {
                'label': {'type': 'string'},
                'summary': {'type': 'string'},
            },
            'required': ['label', 'summary'],
            'additionalProperties': False,
        },
    },
}

# ── Graph Exclude Filter ──
# Comma-separated prefix patterns; matching wiki_pages are skipped for KG extraction.
# Vector-DB indexing and distillation remain unaffected.
GRAPH_EXCLUDE_PATTERNS = [
    p for p in os.environ.get("ACTIVEWIKI_GRAPH_EXCLUDE", "public/spass-").split(",")
    if p.strip()
]


def _should_exclude_from_graph(wiki_page: str) -> bool:
    """Check whether *wiki_page* (format "scope/slug") should be excluded from Knowledge-Graph entity extraction.

    Matches any prefix in :data:`GRAPH_EXCLUDE_PATTERNS`. Returns ``True`` when the page must be skipped.
    """
    for pattern in GRAPH_EXCLUDE_PATTERNS:
        if wiki_page.startswith(pattern):
            return True
    return False


# ── Entity / Relationship type lists ──
ENTITY_TYPES = [
    "PERSON", "ORGANIZATION", "LOCATION", "DOCUMENT", "PROPERTY",
    "FACILITY", "CONCEPT", "DATE", "MONEY", "EVENT", "RATIONALE",
]

# ── Cross-type fuzzy threshold (stricter than same-type FUZZY_THRESHOLD) ──
CROSS_TYPE_FUZZY_THRESHOLD = 90

# ── Priority types for registry prompt injection (max 10 each) ──
PRIORITY_ENTITY_TYPES = ["PERSON", "PROPERTY", "ORGANIZATION", "LOCATION"]
MAX_ENTRIES_PER_PRIORITY_TYPE = 10
MAX_REGISTRY_PROMPT_ENTRIES = 30

# ── Confidence Levels ──────────────────────────────────────────────
# Numeric ordering for confidence comparison (higher = more certain)
CONFIDENCE_ORDER = {
    "extracted": 3,   # Direct statement in source text
    "inferred": 2,    # Logical conclusion derived from text
    "weak": 1,        # Speculative or uncertain connection
}
DEFAULT_CONFIDENCE = "inferred"

# ── HTML Graph Export Config ───────────────────────────────────────
HTML_GRAPH_MAX_NODES = 200
HTML_GRAPH_MAX_EDGES = 500
_D3_LOCAL_PATH = Path(__file__).parent.parent / "assets" / "d3.v7.min.js"
_GRAPH_OUTPUT_DIR = Path(__file__).parent.parent / "output"
_HTML_TEMPLATE_PATH = Path(__file__).parent.parent / "assets" / "template.html"


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
    # Why-Nodes (Phase 2B): Rationale ↔ Fact linking
    "ERKLÄRT_MIT", "HINTET_AUF", "WEGEN",
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

## WHY-NODES: Rationale-Extraktion (Phase 2B)
Wenn ein Text einen Grund, eine Erklärung oder einen Kommentar liefert, extrahiere
den Grund als eigene Rationale-Entity und verlinke sie mit der Fakten-Entity.

**Trigger-Wörter:** "da", "weil", "wegen", "aufgrund", "Hinweis:", "# NOTE:", "Kommentar:", "Grund:"

**Regeln:**
1. **NUR einfache Fälle:** Ursache + Wirkung im selben Satz, eindeutig zugeordnet.
2. **NICHT extrahieren bei komplexen Fällen:** "Miete stieg wegen X, aber Versicherung fiel wegen Y" → zu komplex, überspringen!
3. Rationale-Entity-Typ = **RATIONALE**, Label = die Kurzform des Grundes (max 80 Zeichen)
4. Verknüpfung: Rationale-Entity → Fakten-Entity mit **ERKLÄRT_MIT** (oder **WEGEN** / **HINTET_AUF**)
5. Confidence für Rationale-Relations = **extracted** wenn der Grund direkt im Text steht, **inferred** wenn abgeleitet

**Beispiel:**
> Mietpreis wurde von 650€ auf 720€ erhöht, da die Nebenkosten sich 2024 verdoppelt haben.

Extraktion:
- Entity: `720-euro-miete` (Typ: MONEY, Label: "720€ Miete")
- Entity: `nebenkosten-verdopplung-2024` (Typ: RATIONALE, Label: "Nebenkosten verdoppelten sich 2024")
- Relation: `nebenkosten-verdopplung-2024` -ERKLÄRT_MIT→ `720-euro-miete` (confidence: extracted)

**Negatives Beispiel (NICHT tun):**
> Miete stieg wegen Inflation, aber die Versicherung fiel wegen weniger Schäden.
→ Zwei Gründe für zwei Fakten → zu komplex → KEINE Rationale-Extraktion!

## POSITIVE BEISPIELE (mit Relations + Confidence!)
Good (extracted — direkte Aussagen):
{
  "entities": [
    {"id": "max-mustermann", "label": "Max Mustermann", "type": "PERSON"},
    {"id": "gruenwald", "label": "Gruenwald", "type": "LOCATION"},
    {"id": "haus-gruenwald", "label": "Haus in Gruenwald", "type": "PROPERTY"},
    {"id": "720-euro-miete", "label": "720€ Miete", "type": "MONEY"}
  ],
  "relationships": [
    {"source": "max-mustermann", "target": "gruenwald", "type": "BELEGT", "description": "Max wohnt in Gruenwald", "confidence": "extracted"},
    {"source": "max-mustermann", "target": "haus-gruenwald", "type": "BESITZT", "description": "Max besitzt das Haus", "confidence": "extracted"},
    {"source": "haus-gruenwald", "target": "720-euro-miete", "type": "KOSTET", "description": "Mietpreis beträgt 720€", "confidence": "extracted"}
  ]
}

Good (inferred — Schlussfolgerung):
{
  "entities": [
    {"id": "max-mustermann", "label": "Max Mustermann", "type": "PERSON"},
    {"id": "immobilien-investor", "label": "Immobilieninvestor", "type": "CONCEPT"}
  ],
  "relationships": [
    {"source": "max-mustermann", "target": "immobilien-investor", "type": "HAT_EIGENSCHAFT", "description": "Max investiert in Immobilien", "confidence": "inferred"}
  ]
}

Good (weak — spekulative Verbindung):
{
  "entities": [
    {"id": "simba", "label": "Simba", "type": "CONCEPT"}
  ],
  "relationships": [
    {"source": "simba", "target": "hund", "type": "BEZOGEN_AUF", "description": "Simba könnte ein Hund sein", "confidence": "weak"}
  ]
}

## CONFIDENCE LEVELS (wichtig für Vertrauen!)
Jede Beziehung braucht ein Confidence-Level:
- **extracted** = direkte Aussage im Text (z.B. "Mietpreis: 720€", "Max wohnt in Gruenwald")
- **inferred** = logische Schlussfolgerung (z.B. "Max besitzt Haus → er investiert")
- **weak** = schwache/spekulative Verbindung (z.B. "Simba" — Hund? Katze? Person?)

Regeln:
1. Wenn es direkt im Text steht → **extracted**
2. Wenn du es schlussfolgern musst → **inferred**
3. Wenn du dir unsicher bist → **weak** (lieber weak als falsches extracted!)

## POSITIVE BEISPIELE (mit Relations!)
Good: {
  "entities": [
    {"id": "max-mustermann", "label": "Max Mustermann", "type": "PERSON"},
    {"id": "haus-hauptstr-12a", "label": "Haus Hauptstr. 23b", "type": "PROPERTY"},
    {"id": "gruenwald", "label": "Gruenwald", "type": "LOCATION"},
    {"id": "landratsamt-musterstadt", "label": "Landratsamt Musterstadt", "type": "ORGANIZATION"}
  ],
  "relationships": [
    {"source": "max-mustermann", "target": "haus-muster-1a", "type": "BESITZT", "description": "Max besitzt das Haus", "confidence": "extracted"},
    {"source": "haus-hauptstr-12a", "target": "gruenwald", "type": "BEFINDET_SICH_IN", "description": "Haus liegt in Gruenwald", "confidence": "extracted"},
    {"source": "max-mustermann", "target": "behoerde-musterstadt", "type": "VERTRAG_MIT", "description": "Vertrag mit Behörde", "confidence": "inferred"}
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
    {"source": "wohnung-muenchen-maxvorstadt", "target": "allianz-versicherung", "type": "VERSICHERT_BEI", "description": "Wohnung versichert bei Allianz", "confidence": "extracted"},
    {"source": "wohnung-muenchen-maxvorstadt", "target": "1200-euro-miete", "type": "KOSTET", "description": "Monatliche Miete", "confidence": "extracted"}
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
    {"source": "e3dc-hauskraftwerk-s10", "target": "e3dc-gmbh", "type": "STAMMT_VON", "description": "Hersteller des Systems", "confidence": "extracted"},
    {"source": "e3dc-hauskraftwerk-s10", "target": "10kw-leistung", "type": "HAT_EIGENSCHAFT", "description": "Maximale Leistung", "confidence": "extracted"}
  ]
}

Good (Kochbuch):
{
  "entities": [
    {"id": "schnitzel-wiener-art", "label": "Schnitzel Wiener Art", "type": "CONCEPT"},
    {"id": "semmelbrösel", "label": "Semmelbrösel", "type": "CONCEPT"}
  ],
  "relationships": [
    {"source": "schnitzel-wiener-art", "target": "semmelbrösel", "type": "HAT_KOMPONENTE", "description": "Wird paniert mit Semmelbrösel", "confidence": "extracted"}
  ]
}

Good (Rationale / Why-Node):
{
  "entities": [
    {"id": "720-euro-miete", "label": "720€ Miete", "type": "MONEY"},
    {"id": "nebenkosten-verdopplung-2024", "label": "Nebenkosten verdoppelten sich 2024", "type": "RATIONALE"}
  ],
  "relationships": [
    {"source": "nebenkosten-verdopplung-2024", "target": "720-euro-miete", "type": "ERKLÄRT_MIT", "description": "Miete stieg wegen Nebenkosten", "confidence": "extracted"}
  ]
}

## ANTWORTFORMAT (BINDEND!)
Du MUSST gültiges JSON zurückgeben mit genau dieser Struktur:
{
  "entities": [
    {"id": "kebab-case-id", "label": "Lesbarer Name", "type": "ENTITY_TYPE", "description": "Max 25 Zeichen"}
  ],
  "relationships": [
    {"source": "entity_id_a", "target": "entity_id_b", "type": "RELATION_TYPE", "description": "Was verbindet sie", "confidence": "extracted|inferred|weak"}
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


def _has_migration(conn: sqlite3.Connection, name: str) -> bool:
    """Check whether a named graph migration has already been applied."""
    row = conn.execute(
        "SELECT 1 FROM _graph_migrations WHERE name = ?", (name,)
    ).fetchone()
    return row is not None


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

    # Add valid_until column to relationships if not present (soft-delete for dedup)
    if not _has_column(conn, "relationships", "valid_until"):
        conn.execute("ALTER TABLE relationships ADD COLUMN valid_until TEXT DEFAULT NULL")
        conn.commit()
        _record_graph_migration(conn, "v3_relationships_valid_until")

    # Add confidence column to relationships if not present (Phase 2A: Confidence Tags)
    if not _has_column(conn, "relationships", "confidence"):
        conn.execute("ALTER TABLE relationships ADD COLUMN confidence TEXT DEFAULT 'inferred'")
        conn.commit()
        _record_graph_migration(conn, "v4_relationships_confidence")

    # Rename ambiguous → weak confidence values (migration v6)
    if not _has_migration(conn, "v6_confidence_weak"):
        conn.execute("UPDATE relationships SET confidence = 'weak' WHERE confidence = 'ambiguous'")
        conn.commit()
        _record_graph_migration(conn, "v6_confidence_weak")

    # Add valid_until column to entities if not present (soft-delete for entity dedup)
    if not _has_column(conn, "entities", "valid_until"):
        conn.execute("ALTER TABLE entities ADD COLUMN valid_until TEXT DEFAULT NULL")
        conn.commit()
        _record_graph_migration(conn, "v5_entities_valid_until")


# ═══════════════════════════════════════════════════════════════════════
#  Graph: Facts Cleanup (Dedup + Orphan Removal)
# ═══════════════════════════════════════════════════════════════════════


def _deduplicate_relationships(conn: sqlite3.Connection) -> int:
    """Soft-delete older duplicate relationships (same source+target+relation, multiple targets).

    When the same entity has conflicting facts (e.g., old rent vs new rent),
    the newest row (highest rowid) stays active; older rows get valid_until set.

    Confidence-aware: a newer row can only replace an older row if its confidence
    is >= the older row's confidence. An 'extracted' fact can NEVER be overwritten
    by an 'inferred' or 'weak' one.

    Returns number of relationships invalidated."""
    affected = conn.execute("""
        UPDATE relationships SET valid_until = datetime('now')
        WHERE id IN (
            SELECT r1.id FROM relationships r1
            JOIN relationships r2
                ON r1.source_id = r2.source_id
                AND r1.target_id = r2.target_id
                AND r1.relation_type = r2.relation_type
                AND r1.id != r2.id
                AND r1.valid_until IS NULL AND r2.valid_until IS NULL
                AND r1.rowid < r2.rowid
                AND CASE r1.confidence
                      WHEN 'extracted' THEN 3
                      WHEN 'inferred' THEN 2
                      ELSE 1
                    END
                 <= CASE r2.confidence
                      WHEN 'extracted' THEN 3
                      WHEN 'inferred' THEN 2
                      ELSE 1
                    END
        )
    """).rowcount
    return affected


def _cleanup_orphaned_relationships(conn: sqlite3.Connection) -> int:
    """Soft-delete relationships whose source or target entity no longer exists.
    Returns number of relationships invalidated."""
    affected = conn.execute("""
        UPDATE relationships SET valid_until = datetime('now')
        WHERE (source_id NOT IN (SELECT id FROM entities)
               OR target_id NOT IN (SELECT id FROM entities))
               AND valid_until IS NULL
    """).rowcount
    return affected


# ── String-based Similarity Helpers (Hybrid Match) ────────────────────────
# Stopwords for Jaccard: German articles, conjunctions, prepositions
# NOTE: Stored in normalized form (ae/oe/ue, no umlauts) to match _normalize_label_for_jaccard output
_STOPWORDS_JACCARD = frozenset({
    # Articles
    "der", "die", "das", "ein", "eine", "einer", "eines", "einem", "einen",
    "des", "dem", "den", "derselbe", "dieselbe", "dasselbe",
    # Conjunctions
    "und", "oder", "aber", "doch", "sowie", "beziehungsweise",
    # Prepositions
    "von", "zum", "zur", "im", "am", "an", "auf", "bei", "in", "mit",
    "nach", "neben", "ohne", "unter", "ueber", "vor", "zwischen", "durch",
    "gegen", "hinter", "innerhalb", "außerhalb", "seit", "waehrend",
    # Common filler
    "ist", "sind", "war", "wurde", "hat", "haben", "wird", "werden",
    "auch", "noch", "mehr", "nur", "sehr", "wie", "was", "wo",
    # Building type prefixes (don't change core identity)
    "haus", "gebaeude", "einfamilienhaus", "mehrfamilienhaus", "wohnhaus",
    "gewerbepark", "buero", "bueros", "lokal", "lokale", "raum", "raeume",
    "villa", "hof", "hofanlage", "anlage", "objekt", "immobilie",
    "grundstueck", "grundstuecke", "parzelle", "parzellen",
    "wohnung", "wohnungen", "appartment", "appartement",
})


def _extract_address(label: str) -> tuple[str, str] | None:
    """Extract street name + house number from an entity label.

    Returns (street_name_normalized, house_number) or None if not address-like.

    Examples:
      "Haus Musterstrasse 32" -> ("musterstrasse", "32")
      "Musterort Beispielstrasse 12a" -> ("beispielstrasse", "12a")
      "DHH Nr. 4, Beispielstrasse 12" -> ("beispielstrasse", "12")
      "Altgebaeude Hauptstr. 23" -> ("hauptstr", "23")
    """
    import re as _re

    normalized = label.lower().replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
    normalized = normalized.replace("str.", "strasse").replace("straße", "strasse")

    match = _re.search(r"([a-z][-a-z]*)\s+(\d+[a-z]?)\s*$", normalized)
    if match:
        street_name = match.group(1).strip()
        house_number = match.group(2).strip()

        non_street_words = frozenset({
            "haus", "gebaeude", "einfamilienhaus", "mehrfamilienhaus",
            "wohnhaus", "villa", "hof", "objekt", "immobilie",
            "grundstueck", "wohnung", "anwesen", "gelände", "dhh",
        })

        parts = normalized.split()
        if street_name in non_street_words and len(parts) > 2:
            for k in range(len(parts) - 2, -1, -1):
                if parts[k] not in non_street_words and len(parts[k]) > 2:
                    street_name = parts[k]
                    break

        if len(street_name) > 2:
            return (street_name, house_number)

    return None


def _normalize_label_for_jaccard(label: str) -> set[str]:
    """Normalize a label for Jaccard comparison: lowercase, remove stopwords, split into tokens."""
    # Normalize umlauts and abbreviations
    normalized = label.lower()
    normalized = normalized.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
    normalized = normalized.replace("ß", "ss")
    # Normalize common abbreviations
    normalized = normalized.replace("str.", "straße")
    normalized = normalized.replace("strasse", "straße")
    normalized = normalized.replace("nr.", "nummer")
    # Tokenize: split on whitespace, punctuation, and special chars
    tokens = re.findall(r"[a-zA-ZäöüÄÖÜß0-9]+", normalized)
    # Remove stopwords and very short tokens
    tokens = {t for t in tokens if t not in _STOPWORDS_JACCARD and len(t) > 1}
    return tokens


def _jaccard_similarity(a: str, b: str) -> float:
    """Compute Jaccard similarity between two labels (0.0–1.0).

    Normalizes both labels, removes stopwords, then computes
    |intersection| / |union| of the resulting token sets.
    """
    set_a = _normalize_label_for_jaccard(a)
    set_b = _normalize_label_for_jaccard(b)
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


def _longest_common_subsequence_ratio(a: str, b: str) -> float:
    """Compute LCS ratio between two lowercased strings (0.0–1.0).

    Uses dynamic programming on the lowercased, whitespace-normalized strings.
    Returns 2 * LCS_length / (len(a) + len(b)).
    """
    sa = a.lower().replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
    sb = b.lower().replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
    # Normalize whitespace
    sa = re.sub(r"\s+", " ", sa).strip()
    sb = re.sub(r"\s+", " ", sb).strip()
    la, lb = len(sa), len(sb)
    if la == 0 or lb == 0:
        return 0.0
    # Cap length to avoid O(n²) blowup on very long labels
    if la > 200 or lb > 200:
        return 0.0
    # Standard LCS DP
    prev = [0] * (lb + 1)
    for i in range(1, la + 1):
        curr = [0] * (lb + 1)
        for j in range(1, lb + 1):
            if sa[i - 1] == sb[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev = curr
    lcs_len = prev[lb]
    return 2.0 * lcs_len / (la + lb)


def _string_match_score(a: str, b: str) -> tuple[float, float]:
    """Compute both Jaccard and LCS similarity between two labels.

    Returns (jaccard_score, lcs_score), both in [0.0, 1.0].
    """
    return (_jaccard_similarity(a, b), _longest_common_subsequence_ratio(a, b))


# ═══════════════════════════════════════════════════════════════════════
#  Entity Deduplication (Embedding-Based + Hybrid String Match)
# ═══════════════════════════════════════════════════════════════════════


def _embed_entity_labels(labels: list[str]) -> np.ndarray:
    """Batch-embed entity labels, returning L2-normalized float32 vectors.

    Returns a (len(labels), EMBED_DIMS) array. Zero vectors for any
    individual embedding failures so the caller can detect them.
    """
    if not labels:
        return np.zeros((0, EMBED_DIMS), dtype=np.float32)
    return embed(labels)


def _cosine_matrix(A: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity matrix for rows of A (already L2-normalized).

    Returns (N, N) float64 matrix where M[i,j] = cosine(A[i], A[j]).
    """
    return (A @ A.T).astype(np.float64)


class _UnionFind:
    """Minimal Union-Find for clustering."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path compression
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

    def clusters(self) -> dict[int, list[int]]:
        groups: dict[int, list[int]] = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            groups.setdefault(root, []).append(i)
        return groups


def deduplicate_entities(
    conn: sqlite3.Connection,
    threshold: float = 0.95,
    string_threshold: float = 0.66,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Hybrid entity deduplication: embedding + string-based matching.

    1. Load all active entities and their labels.
    2. Batch-embed the labels.
    3. Compute pairwise cosine similarity.
    4. Compute pairwise Jaccard similarity (string-based).
    5. Union-Find clustering: merge if cosine >= threshold
       OR (jaccard >= string_threshold AND same entity_type).
    6. For each cluster with >1 entity: pick canonical (most links),
       redirect all relationship pointers, soft-delete duplicates.

    Returns a report dict with counts and merge details.
    """
    # ── Step 1: Load active entities ────────────────────────────────
    rows = conn.execute(
        "SELECT id, label, entity_type, description FROM entities WHERE valid_until IS NULL"
    ).fetchall()

    if len(rows) < 2:
        return {"entities_loaded": len(rows), "clusters": 0, "merged": 0, "redirects": 0}

    ids = [r[0] for r in rows]
    labels = [r[1] for r in rows]

    log("dedup", f"Loaded {len(rows)} active entities for embedding-based dedup")

    # ── Step 2: Batch-embed labels ──────────────────────────────────
    try:
        vecs = _embed_entity_labels(labels)
    except Exception as e:
        log("error", f"Entity embedding failed: {e}", stderr=True)
        return {"error": str(e)}

    # Drop zero-norm rows (embedding failures) — they can't match anything
    valid_mask = np.linalg.norm(vecs, axis=1) > 0
    valid_indices = [i for i in range(len(rows)) if valid_mask[i]]
    valid_vecs = vecs[valid_mask]
    valid_ids = [ids[i] for i in valid_indices]
    valid_id_to_local = {eid: li for li, eid in enumerate(valid_ids)}

    if len(valid_vecs) < 2:
        return {
            "entities_loaded": len(rows),
            "embeddable": len(valid_vecs),
            "clusters": 0,
            "merged": 0,
            "redirects": 0,
        }

    log("dedup", f"Embedded {len(valid_vecs)}/{len(rows)} labels (skipped {len(rows) - len(valid_vecs)} zero-norm)")

    # ── Step 3: Pairwise cosine similarity ──────────────────────────
    sim = _cosine_matrix(valid_vecs)

    # ── Step 3.5: Pairwise Jaccard similarity (string-based) ────────
    # Build entity_type lookup and token set cache for same-type + min-token checks
    valid_types = [rows[i][2] for i in valid_indices]  # entity_type per valid index
    valid_token_sets = [_normalize_label_for_jaccard(labels[idx]) for idx in valid_indices]
    jaccard_sim = np.zeros((len(valid_vecs), len(valid_vecs)), dtype=np.float64)
    for i in range(len(valid_vecs)):
        for j in range(i + 1, len(valid_vecs)):
            jacc = _jaccard_similarity(labels[valid_indices[i]], labels[valid_indices[j]])
            jaccard_sim[i, j] = jacc
            jaccard_sim[j, i] = jacc

    # ── Step 4: Union-Find clustering (hybrid: cosine OR jaccard+same_type) ──
    uf = _UnionFind(len(valid_vecs))
    pair_count_cosine = 0
    pair_count_jaccard = 0
    for i in range(len(valid_vecs)):
        for j in range(i + 1, len(valid_vecs)):
            matched = False
            if sim[i, j] >= threshold:
                matched = True
                pair_count_cosine += 1
            elif (jaccard_sim[i, j] >= string_threshold
                  and valid_types[i] == valid_types[j]
                  and len(valid_token_sets[i]) >= 2 and len(valid_token_sets[j]) >= 2
                  and len(valid_token_sets[i] & valid_token_sets[j]) >= 2):
                matched = True
                pair_count_jaccard += 1
            if matched:
                uf.union(i, j)

    clusters = uf.clusters()
    multi_clusters = {root: members for root, members in clusters.items() if len(members) > 1}

    if not multi_clusters:
        log("dedup", f"No clusters above thresholds (cosine={threshold:.2f}, jaccard={string_threshold:.2f})")
        return {
            "entities_loaded": len(rows),
            "embeddable": len(valid_vecs),
            "pairs_above_cosine_threshold": pair_count_cosine,
            "pairs_above_jaccard_threshold": pair_count_jaccard,
            "clusters": 0,
            "merged": 0,
            "redirects": 0,
        }

    log("dedup", f"Found {len(multi_clusters)} clusters (cosine pairs={pair_count_cosine}, jaccard pairs={pair_count_jaccard})")

    # ── Step 5: Dry-run check ────────────────────────────────────────
    if dry_run:
        total_merged_dry = sum(len(m) - 1 for m in multi_clusters.values())
        log("dedup", f"[DRY RUN] Would merge {total_merged_dry} entities across {len(multi_clusters)} clusters")

        dry_merges = []
        for root, members in multi_clusters.items():
            if not members:
                continue
            canonical_idx = members[0]
            ci_global = valid_indices[canonical_idx]
            can_label = labels[ci_global]
            can_id = rows[ci_global][0]

            for mi in members[1:]:
                di_global = valid_indices[mi]
                dup_label = labels[di_global]
                dup_id = rows[di_global][0]

                cos_score = float(sim[canonical_idx, mi]) if canonical_idx < len(sim) and mi < len(sim[0]) else -1.0
                jac_score = float(jaccard_sim[canonical_idx, mi]) if canonical_idx < len(jaccard_sim) and mi < len(jaccard_sim[0]) else -1.0
                
                if cos_score >= threshold:
                    method = "cosine"
                elif jac_score >= string_threshold:
                    method = "jaccard"
                else:
                    method = "transitive"

                dry_merges.append({
                    "canonical": can_id,
                    "canonical_label": can_label,
                    "duplicate": dup_id,
                    "duplicate_label": dup_label,
                    "method": method,
                    "cosine_similarity": round(cos_score, 4),
                    "jaccard_similarity": round(jac_score, 4),
                })

                log("dedup", f"  [WOULD MERGE] '{dup_id}' -> '{can_id}' ({method}, cos={cos_score:.4f}, jac={jac_score:.4f})")

        return {
            "dry_run": True,
            "would_merge": total_merged_dry,
            "would_clusters": len(multi_clusters),
            "merges": dry_merges,
        }

    # ── Step 5: Merge each cluster ──────────────────────────────────
    total_merged = 0
    total_redirects = 0
    merge_report: list[dict] = []

    for root, members in multi_clusters.items():
        member_ids = [valid_ids[m] for m in members]

        # Pick canonical: entity with most active relationships
        link_counts = conn.execute(
            "SELECT e.id, COUNT(r.id) AS cnt "
            "FROM entities e "
            "LEFT JOIN relationships r "
            "  ON (r.source_id = e.id OR r.target_id = e.id) "
            "  AND r.valid_until IS NULL "
            "WHERE e.id IN (" + ",".join("?" * len(member_ids)) + ") "
            "GROUP BY e.id",
            member_ids,
        ).fetchall()

        # Canonical = most links; tie-break by oldest (lowest ROWID)
        canonical_id = max(
            member_ids,
            key=lambda eid: (
                next((cnt for eid2, cnt in link_counts if eid2 == eid), 0),
                -conn.execute("SELECT ROWID FROM entities WHERE id = ?", (eid,)).fetchone()[0],
            ),
        )

        duplicates = [eid for eid in member_ids if eid != canonical_id]

        if not duplicates:
            continue

        canonical_row = conn.execute(
            "SELECT label, entity_type, description FROM entities WHERE id = ?",
            (canonical_id,),
        ).fetchone()
        canonical_label = canonical_row[0]

        for dup_id in duplicates:
            dup_row = conn.execute(
                "SELECT label, entity_type, description FROM entities WHERE id = ?",
                (dup_id,),
            ).fetchone()
            dup_label = dup_row[0]

            # Redirect relationships: source
            src_redirect = conn.execute(
                "UPDATE relationships SET source_id = ? WHERE source_id = ? AND valid_until IS NULL",
                (canonical_id, dup_id),
            ).rowcount

            # Redirect relationships: target
            tgt_redirect = conn.execute(
                "UPDATE relationships SET target_id = ? WHERE target_id = ? AND valid_until IS NULL",
                (canonical_id, dup_id),
            ).rowcount

            total_redirects += src_redirect + tgt_redirect

            # Merge descriptions into canonical
            if dup_row[2] and canonical_row[2] and dup_row[2] != canonical_row[2]:
                merged_desc = _merge_description(canonical_row[2], dup_row[2])
                conn.execute(
                    "UPDATE entities SET description = ?, updated_at = datetime('now') WHERE id = ?",
                    (merged_desc, canonical_id),
                )

            # Soft-delete the duplicate
            conn.execute(
                "UPDATE entities SET valid_until = datetime('now') WHERE id = ?",
                (dup_id,),
            )
            total_merged += 1

            # Get similarity scores for report
            if canonical_id in valid_id_to_local and dup_id in valid_id_to_local:
                ci = valid_id_to_local[canonical_id]
                di = valid_id_to_local[dup_id]
                cos_score = float(sim[ci, di])
                jac_score = float(jaccard_sim[ci, di])
            else:
                cos_score = -1.0
                jac_score = -1.0

            # Determine match method
            if cos_score >= threshold:
                match_method = "cosine"
                primary_score = cos_score
            elif jac_score >= string_threshold:
                match_method = "jaccard"
                primary_score = jac_score
            else:
                match_method = "transitive"
                primary_score = max(cos_score, jac_score)

            merge_report.append({
                "canonical": canonical_id,
                "canonical_label": canonical_label,
                "duplicate": dup_id,
                "duplicate_label": dup_label,
                "cosine_similarity": round(cos_score, 4),
                "jaccard_similarity": round(jac_score, 4),
                "match_method": match_method,
                "primary_score": round(primary_score, 4),
                "redirects": src_redirect + tgt_redirect,
            })

            log("dedup",
                f"  [merge] '{dup_label}' → '{canonical_label}' "
                f"(method={match_method}, cos={cos_score:.4f}, jac={jac_score:.4f}, redirects={src_redirect + tgt_redirect})")

    conn.commit()

    # Clean up orphaned relationships after entity soft-deletes
    orphan_cleanup = _cleanup_orphaned_relationships(conn)
    conn.commit()

    report = {
        "entities_loaded": len(rows),
        "embeddable": len(valid_vecs),
        "cosine_threshold": threshold,
        "jaccard_threshold": string_threshold,
        "pairs_above_cosine_threshold": pair_count_cosine,
        "pairs_above_jaccard_threshold": pair_count_jaccard,
        "clusters": len(multi_clusters),
        "merged": total_merged,
        "redirects": total_redirects,
        "orphan_rels_cleaned": orphan_cleanup,
        "merges": merge_report,
    }
    log("dedup", f"Done: {total_merged} entities merged, {total_redirects} relationships redirected")
    return report


# ═══════════════════════════════════════════════════════════════════════
#  Embedding-Based Entity Prevention (during extraction)
# ═══════════════════════════════════════════════════════════════════════


def _resolve_by_embedding(
    conn: sqlite3.Connection,
    label: str,
    entity_type: str,
    threshold: float = 0.92,
    string_threshold: float = 0.75,
    _cache: dict[str, np.ndarray] | None = None,
) -> str | None:
    """Try to find an existing entity by embedding similarity, with Jaccard fallback.

    Two-stage matching:
      1. Embedding-based cosine similarity (same type only)
      2. Jaccard token overlap (same type only, min 2 shared tokens, min 2 total tokens)

    Returns the canonical entity id if a match is found, or None.

    Called as Priority 3.5 in resolve_entity(), between label NOCASE
    match and fuzzy match, to catch semantic duplicates that string
    methods miss (e.g. 'Musterstraße 32' vs 'Haus Musterstraße 32').

    Optimisation: batch-embeds all candidate labels in a single Ollama
    call rather than N+1 individual calls. Uses *_cache* (shared across
    calls within the same page) to avoid re-embedding the same existing
    entities repeatedly.

    Args:
        _cache: Optional dict mapping label → embedding vector. Populated
            on first call and reused for subsequent calls in the same
            process_page() invocation. DO NOT pass from outside.
    """
    # Load same-type entities
    same_type_rows = conn.execute(
        "SELECT id, label FROM entities WHERE entity_type = ? AND valid_until IS NULL",
        (entity_type,),
    ).fetchall()

    if not same_type_rows:
        return None

    # Filter out exact label match (handled earlier)
    label_lower = label.lower()
    candidates = [(eid, e_label) for eid, e_label in same_type_rows
                   if e_label.lower() != label_lower]

    if not candidates:
        return None

    # ── Stage 1: Embedding-based matching ────────────────────────────
    if _cache is None:
        _cache = {}
    existing_labels = [e_label for _, e_label in candidates]
    missing_labels = [l for l in existing_labels if l not in _cache]

    if missing_labels:
        try:
            missing_vecs = embed(missing_labels)
            for ml, mv in zip(missing_labels, missing_vecs):
                if np.any(mv):
                    _cache[ml] = mv
        except Exception as e:
            log("warn", f"  [embed-dedup] embedding failed for existing entities: {e}")
            # Fall through to Jaccard below

    # Embed the new label
    try:
        q_vec = embed([label])
        if q_vec.size == 0 or not np.any(q_vec[0]):
            # Embedding failed — fall through to Jaccard
            pass
        else:
            q = q_vec[0]
            best_score = 0.0
            best_id = None

            for eid, e_label in candidates:
                e_vec = _cache.get(e_label)
                if e_vec is None or not np.any(e_vec):
                    continue
                sim = float(np.dot(q, e_vec) / (np.linalg.norm(q) * np.linalg.norm(e_vec)))
                if sim > best_score:
                    best_score = sim
                    best_id = eid

            if best_score >= threshold and best_id:
                log("dedup",
                    f"  [embed-dedup] '{label}' → matched '{best_id}' "
                    f"(cosine={best_score:.4f}, threshold={threshold})")
                return best_id
    except Exception as e:
        log("warn", f"  [embed-dedup] embedding failed for '{label}': {e}")

    # ── Stage 2: Jaccard fallback (string-based, same type) ──────────
    # Require min 2 shared tokens AND min 2 total tokens per label (avoid merging single words)
    new_tokens = _normalize_label_for_jaccard(label)
    if len(new_tokens) < 2:
        return None  # single-word labels must rely on embedding or exact match

    best_jaccard = 0.0
    best_jaccard_id = None

    for eid, e_label in candidates:
        existing_tokens = _normalize_label_for_jaccard(e_label)
        if len(existing_tokens) < 2:
            continue  # skip single-word existing labels for Jaccard match
        shared = len(new_tokens & existing_tokens)
        if shared < 2:
            continue  # need at least 2 shared tokens to avoid false merges
        jac = _jaccard_similarity(label, e_label)
        if jac > best_jaccard:
            best_jaccard = jac
            best_jaccard_id = eid

    if best_jaccard >= string_threshold and best_jaccard_id:
        log("dedup",
            f"  [jaccard-dedup] '{label}' → matched '{best_jaccard_id}' "
            f"(jaccard={best_jaccard:.4f}, threshold={string_threshold})")
        return best_jaccard_id

    return None


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
    _embed_cache: dict[str, np.ndarray] | None = None,
) -> str:
    """Check for existing similar entity. If found → update, else insert.
    Returns the canonical entity id.

    Priority chain:
      1. Registry lookup (highest priority - authoritative mapping)
      2. ID exact match
      3. Label NOCASE exact match
      3.5. Embedding-based match (catches semantic duplicates)
      4. Fuzzy match within same entity_type (threshold 85%)
      5. Cross-type fuzzy match (threshold 90%, first-write-wins on type)
      6. Insert as new entity

    Args:
        _embed_cache: Optional shared cache of existing entity embeddings.
            Populated lazily and reused across calls within the same
            process_page() invocation to avoid redundant embedding calls.
            DO NOT pass from outside.
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

    # ── Priority 3.5: Embedding-based match (catches semantic dups) ──
    embed_id = _resolve_by_embedding(conn, label, entity_type, _cache=_embed_cache)
    if embed_id:
        existing_row = conn.execute(
            "SELECT id, label, description FROM entities WHERE id = ?", (embed_id,)
        ).fetchone()
        if existing_row:
            eid, _, old_desc = existing_row
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
        "temperature": llm_temperature(_CONFIG),
        "max_tokens": 8192,
        "thinking": "off",  # vLLM: prevents » thinking tags in JSON output
    }

    last_err: Optional[Exception] = None
    raw = ''
    for attempt in range(RETRY_MAX):
        try:
            req = urllib.request.Request(
                f"{VLLM_URL}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=3600) as resp:
                raw_bytes = resp.read()
                if not raw_bytes:
                    raise ValueError("vLLM returned 200 OK with empty body")
                body = json.loads(raw_bytes.decode("utf-8"))
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
            f"{VLLM_URL}/models",
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

    # Exclude filter — skip KG extraction for matched pages
    if _should_exclude_from_graph(wiki_page):
        log("graph-exclude", f" Skipping {wiki_page} (excluded from graph)")
        return 0, 0

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
        # Extract confidence from LLM output; default to 'inferred' if missing
        rel_confidence = rel.get("confidence", DEFAULT_CONFIDENCE)
        if rel_confidence not in CONFIDENCE_ORDER:
            log("warn", f"  [graph-health] Invalid confidence '{rel_confidence}', defaulting to '{DEFAULT_CONFIDENCE}'", stderr=True)
            rel_confidence = DEFAULT_CONFIDENCE

        rel_id = hashlib.sha256(
            f"{src_id}::{tgt_id}::{rel_type}".encode()
        ).hexdigest()[:16]

        # SELECT-first pattern: check existing rel by rel_id, decide INSERT vs UPDATE
        existing_row = conn.execute(
            "SELECT confidence FROM relationships WHERE id = ?",
            (rel_id,),
        ).fetchone()

        if existing_row is None:
            # New relationship — insert with confidence
            conn.execute(
                "INSERT OR IGNORE INTO relationships(id, source_id, target_id, relation_type, description, confidence) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (rel_id, src_id, tgt_id, rel_type, rel_desc, rel_confidence),
            )
            inserted_rels += 1
        elif CONFIDENCE_ORDER[rel_confidence] > CONFIDENCE_ORDER[existing_row[0]]:
            # Existing relationship — upgrade confidence if new one is higher
            conn.execute(
                "UPDATE relationships SET confidence = ?, description = ? WHERE id = ?",
                (rel_confidence, rel_desc, rel_id),
            )
            log("graph-conf", f"  [confidence-upgrade] {src_id} -{rel_type}-> {tgt_id}: {existing_row[0]} → {rel_confidence}")
        # else: existing confidence >= new confidence — skip (preserve higher certainty)

        # Junction table: relationship → wiki_page (always track, regardless of insert/update)
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
        # Exclude filter — skip KG extraction for matched pages
        if _should_exclude_from_graph(wiki_page):
            log("graph-incr", f"Skipping {wiki_page} (excluded from graph)")
            # Still record state so we don't re-check next run
            conn.execute(
                "INSERT OR REPLACE INTO graph_page_state(wiki_page, body_hash, processed_at) "
                "VALUES (?, ?, datetime('now'))",
                (wiki_page, body_hash),
            )
            conn.commit()
            continue

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

        # Facts cleanup: deduplicate conflicting facts + remove orphans
        deduped = _deduplicate_relationships(conn)
        orphans = _cleanup_orphaned_relationships(conn)
        conn.commit()

        log("graph", f"Done: {total_entities} entities, {total_relations} relations "
              f"from {len(pages)} pages ({skipped} skipped) → {DB_PATH}")
        log("graph", f"Deduplicated {deduped} relationships, cleaned up {orphans} orphans")

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
        "temperature": llm_temperature(_CONFIG),
        "max_tokens": 2048,
        "response_format": _SCHEMA_COMMUNITY_SUMMARY,
        "chat_template_kwargs": {"enable_thinking": False},  # vLLM: prevents thinking tags in JSON output
    }

    last_err: Optional[Exception] = None
    for attempt in range(RETRY_MAX):
        try:
            req = urllib.request.Request(
                f"{VLLM_URL}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=3600) as resp:
                raw_bytes = resp.read()
                if not raw_bytes:
                    raise ValueError("vLLM returned 200 OK with empty body")
                body = json.loads(raw_bytes.decode("utf-8"))
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
        "SELECT source_id, target_id, relation_type FROM relationships WHERE valid_until IS NULL"
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
            "AND r.target_id IN (" + ",".join("?" * len(member_ids)) + ") "
            "AND r.valid_until IS NULL",
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


def cmd_graph_god_nodes(top_n: int = 10) -> None:
    """Show the top-N entities by connection count (most connected nodes).

    Prints a formatted table to stdout.
    """
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT e.label, e.entity_type, COUNT(DISTINCT r.id) AS connections
        FROM entities e
        LEFT JOIN relationships r
            ON (r.source_id = e.id OR r.target_id = e.id)
            AND r.valid_until IS NULL
        WHERE e.valid_until IS NULL
        GROUP BY e.id
        ORDER BY connections DESC, e.label ASC
        LIMIT ?
    """, (top_n,)).fetchall()
    conn.close()

    if not rows:
        print("No entities found.")
        return

    # Column widths
    rank_w = len(str(top_n))
    label_w = max(len(r[0]) for r in rows) + 2
    type_w = max(max(len(r[1]) for r in rows), 6)  # at least "TYPE"
    conn_w = max(len(str(r[2])) for r in rows) + 2  # at least "CONN"

    def _pad(s, w):
        return str(s).ljust(w)

    # Header
    header = f"{'#':<{rank_w}} | {'Label':<{label_w}} | {'Type':<{type_w}} | {'Connections':<{conn_w}}"
    sep = "-" * len(header)

    print(f"\n{_pad('', rank_w)} {_pad(f'Top {top_n} God Nodes', label_w + type_w + conn_w + 8)}")
    print(sep)
    print(header)
    print(sep)

    for i, (label, etype, conns) in enumerate(rows, 1):
        print(f"{_pad(i, rank_w)} | {_pad(label, label_w)} | {_pad(etype, type_w)} | {_pad(conns, conn_w)}")


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
    total_relations_active = conn.execute(
        "SELECT COUNT(*) FROM relationships WHERE valid_until IS NULL"
    ).fetchone()[0]
    total_relations_inactive = conn.execute(
        "SELECT COUNT(*) FROM relationships WHERE valid_until IS NOT NULL"
    ).fetchone()[0]
    total_relations = total_relations_active + total_relations_inactive

    by_type = dict(conn.execute(
        "SELECT entity_type, COUNT(*) FROM entities GROUP BY entity_type ORDER BY COUNT(*) DESC"
    ).fetchall())

    by_rel_type = dict(conn.execute(
        "SELECT relation_type, COUNT(*) FROM relationships WHERE valid_until IS NULL GROUP BY relation_type ORDER BY COUNT(*) DESC"
    ).fetchall())

    # Confidence distribution (Phase 2A)
    confidence_dist = dict(conn.execute(
        "SELECT confidence, COUNT(*) FROM relationships WHERE valid_until IS NULL GROUP BY confidence ORDER BY COUNT(*) DESC"
    ).fetchall())

    orphans = conn.execute(
        """
        SELECT e.id, e.label, e.entity_type
        FROM entities e
        WHERE e.id NOT IN (SELECT source_id FROM relationships WHERE valid_until IS NULL)
          AND e.id NOT IN (SELECT target_id FROM relationships WHERE valid_until IS NULL)
        ORDER BY e.label
        """
    ).fetchall()

    top_connected = conn.execute(
        """
        SELECT e.id, e.label, COUNT(r.id) AS degree
        FROM entities e
        JOIN relationships r ON (r.source_id = e.id OR r.target_id = e.id)
        WHERE r.valid_until IS NULL
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
        "relationships_active": total_relations_active,
        "relationships_inactive": total_relations_inactive,
        "by_entity_type": by_type,
        "by_relation_type": by_rel_type,
        "confidence_distribution": confidence_dist,
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
        WHERE e.id NOT IN (SELECT source_id FROM relationships WHERE valid_until IS NULL)
          AND e.id NOT IN (SELECT target_id FROM relationships WHERE valid_until IS NULL)
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
        WHERE r.valid_until IS NULL AND (es.id IS NULL OR et.id IS NULL)
        """
    ).fetchall()
    if broken:
        issues.append({
            "issue": "broken_references",
            "count": len(broken),
            "details": [{"rel_id": b[0], "source": b[1], "target": b[2], "type": b[3]} for b in broken],
        })

    loops = conn.execute(
        "SELECT id, source_id, relation_type FROM relationships WHERE source_id = target_id AND valid_until IS NULL"
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


def _entities_to_json(
    conn: sqlite3.Connection,
    rows,
    with_rationale: bool = False,
) -> list[dict]:
    """Build the graph JSON shape (entity + 1-hop relations) for rows of
    (id, label, entity_type, description, wiki_page). Shared by graph search/pages.

    Args:
        conn: Open SQLite connection.
        rows: Rows of (id, label, entity_type, description, wiki_page).
        with_rationale: If True, also include incoming ERKLÄRT_MIT/WEGEN/HINTET_AUF
            relations from RATIONALE entities as a "rationale" field.
    """
    result = []
    for eid, label, etype, desc, page in rows:
        rels_out = conn.execute(
            "SELECT r.relation_type, r.description, e.label, r.confidence "
            "FROM relationships r JOIN entities e ON r.target_id = e.id "
            "WHERE r.source_id = ? AND r.valid_until IS NULL",
            (eid,),
        ).fetchall()
        rels_in = conn.execute(
            "SELECT r.relation_type, r.description, e.label, r.confidence "
            "FROM relationships r JOIN entities e ON r.source_id = e.id "
            "WHERE r.target_id = ? AND r.valid_until IS NULL",
            (eid,),
        ).fetchall()

        entry: dict[str, object] = {
            "label": label,
            "type": etype,
            "description": desc or "",
            "wiki_page": page or "",
            "outgoing": [
                {"relation_type": rt, "target": tl, "description": rd or "", "confidence": conf or DEFAULT_CONFIDENCE}
                for rt, rd, tl, conf in rels_out[:10]
            ],
            "incoming": [
                {"relation_type": rt, "source": sl, "description": rd or "", "confidence": conf or DEFAULT_CONFIDENCE}
                for rt, rd, sl, conf in rels_in[:10]
            ],
        }

        # Phase 2B: attach rationale nodes (incoming ERKLÄRT_MIT/WEGEN/HINTET_AUF from RATIONALE entities)
        if with_rationale:
            rationale_rows = conn.execute(
                "SELECT r.relation_type, r.description, e.label, r.confidence "
                "FROM relationships r JOIN entities e ON r.source_id = e.id "
                "WHERE r.target_id = ? "
                "AND r.valid_until IS NULL "
                "AND r.relation_type IN ('ERKLÄRT_MIT', 'WEGEN', 'HINTET_AUF') "
                "AND e.entity_type = 'RATIONALE'",
                (eid,),
            ).fetchall()
            if rationale_rows:
                entry["rationale"] = [
                    {"relation_type": rt, "source": sl, "description": rd or "", "confidence": conf or DEFAULT_CONFIDENCE}
                    for rt, rd, sl, conf in rationale_rows[:5]
                ]

        result.append(entry)
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


def cmd_graph_search(
    query: str,
    as_json: bool = False,
    scopes: list[str] | None = None,
    with_rationale: bool = False,
) -> None:
    """Simple graph lookup: find entities matching query + their 1-hop relations.

    Args:
        query: Text to match against entity labels and descriptions.
        as_json: If True, emit a JSON array to stdout instead of log-lines.
        scopes: Optional list of scope prefixes (e.g. ["private", "family"])
                to filter entities by their wiki_page field.
        with_rationale: If True, also show incoming ERKLÄRT_MIT/WEGEN/HINTET_AUF
            relations from RATIONALE entities (Phase 2B Why-Nodes).
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
        result = _entities_to_json(conn, rows, with_rationale=with_rationale)
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
                "SELECT r.relation_type, r.description, e.label, r.confidence "
                "FROM relationships r JOIN entities e ON r.target_id = e.id "
                "WHERE r.source_id = ? AND r.valid_until IS NULL",
                (eid,),
            ).fetchall()

            rels_in = conn.execute(
                "SELECT r.relation_type, r.description, e.label, r.confidence "
                "FROM relationships r JOIN entities e ON r.source_id = e.id "
                "WHERE r.target_id = ? AND r.valid_until IS NULL",
                (eid,),
            ).fetchall()

            if rels_out:
                log("info", f"    \u2192 outgoing ({len(rels_out)}):")
                for rtype, rdesc, target_label, conf in rels_out[:10]:
                    conf_display = conf or DEFAULT_CONFIDENCE
                    log("info", f"      \u2514\u2500 {rtype} \u2192 {target_label} [confidence={conf_display}]")
                    if rdesc:
                        log("info", f"         {rdesc}")

            if rels_in:
                log("info", f"    \u2190 incoming ({len(rels_in)}):")
                for rtype, rdesc, source_label, conf in rels_in[:10]:
                    conf_display = conf or DEFAULT_CONFIDENCE
                    log("info", f"      \u2514\u2500 {source_label} {rtype} [confidence={conf_display}]")
                    if rdesc:
                        log("info", f"         {rdesc}")

            # Phase 2B: show rationale nodes
            if with_rationale:
                rationale_rows = conn.execute(
                    "SELECT r.relation_type, r.description, e.label, r.confidence "
                    "FROM relationships r JOIN entities e ON r.source_id = e.id "
                    "WHERE r.target_id = ? "
                    "AND r.valid_until IS NULL "
                    "AND r.relation_type IN ('ERKLÄRT_MIT', 'WEGEN', 'HINTET_AUF') "
                    "AND e.entity_type = 'RATIONALE'",
                    (eid,),
                ).fetchall()
                if rationale_rows:
                    log("info", f"    \u2753 rationale ({len(rationale_rows)}):")
                    for rtype, rdesc, source_label, conf in rationale_rows[:5]:
                        conf_display = conf or DEFAULT_CONFIDENCE
                        log("info", f"      \u2514\u2500 {source_label} {rtype} [confidence={conf_display}]")
                        if rdesc:
                            log("info", f"         {rdesc}")

            log("info", "")

        conn.close()


# ═══════════════════════════════════════════════════════════════════════
#  HTML Graph Export (Phase 2C — data layer only)
# ═══════════════════════════════════════════════════════════════════════


def _build_graph_data(
    db: sqlite3.Connection,
    query: str,
    min_confidence: str | None = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Build subgraph data for HTML export.

    1. Find seed entities matching *query*.
    2. Expand to 1-hop neighbours (both directions).
    3. Collect relationships among all discovered entities.

    Args:
        db: Open SQLite connection.
        query: Text to match against entity labels and descriptions.
        min_confidence: If set, filter relationships by confidence level
            (must be one of CONFIDENCE_ORDER keys).

    Returns:
        (entities_list, relationships_list) as plain Python lists of dicts.
    """
    like = f"%{query}%"

    # ── Step 1: Seed entities ────────────────────────────────────────
    seed_rows = db.execute(
        "SELECT id, label, entity_type, COALESCE(description, '') FROM entities "
        "WHERE (label LIKE ? OR description LIKE ?) COLLATE NOCASE AND valid_until IS NULL",
        (like, like),
    ).fetchall()

    seed_ids = {row[0] for row in seed_rows}
    if not seed_ids:
        return [], []

    seed_ids_str = ", ".join(f"'{sid}'" for sid in seed_ids)

    # ── Step 2: 1-hop neighbours (both directions) ──────────────────
    neighbour_rows = db.execute(
        f"SELECT e.id, e.label, e.entity_type, COALESCE(e.description, '') "
        "FROM entities e "
        "INNER JOIN relationships r ON (r.source_id = e.id OR r.target_id = e.id) "
        f"WHERE (r.source_id IN ({seed_ids_str}) OR r.target_id IN ({seed_ids_str})) "
        "AND e.valid_until IS NULL AND r.valid_until IS NULL "
        "UNION "
        f"SELECT id, label, entity_type, COALESCE(description, '') FROM entities "
        f"WHERE id IN ({seed_ids_str})"
    ).fetchall()

    all_ids = {row[0] for row in neighbour_rows}
    all_ids_str = ", ".join(f"'{eid}'" for eid in all_ids)

    # ── Step 3: Relationships among discovered entities ──────────────
    rel_where = f"source_id IN ({all_ids_str}) AND target_id IN ({all_ids_str}) AND valid_until IS NULL"
    if min_confidence and min_confidence in CONFIDENCE_ORDER:
        min_val = CONFIDENCE_ORDER[min_confidence]
        rel_where += (
            f" AND CASE confidence WHEN 'extracted' THEN 3 WHEN 'inferred' THEN 2 ELSE 1 END >= {min_val}"
        )

    rel_rows = db.execute(
        f"SELECT source_id, target_id, relation_type, COALESCE(confidence, '{DEFAULT_CONFIDENCE}') "
        f"FROM relationships WHERE {rel_where}"
    ).fetchall()

    # ── Build result structures ──────────────────────────────────────
    entities = []
    for eid, label, etype, desc in neighbour_rows:
        entities.append({
            "id": eid,
            "label": label,
            "type": etype,
            "description": desc[:500],
        })

    relationships = []
    for src, tgt, rtype, conf in rel_rows:
        relationships.append({
            "source_id": src,
            "target_id": tgt,
            "relation_type": rtype,
            "confidence": conf,
        })

    return entities, relationships


def _build_full_graph_data(
    db: sqlite3.Connection,
    min_confidence: str | None = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Build the full graph — all active entities and relationships.

    Args:
        db: Open SQLite connection.
        min_confidence: If set, filter relationships by confidence level.

    Returns:
        (entities_list, relationships_list) as plain Python lists of dicts.
    """
    # All active entities
    entity_rows = db.execute(
        "SELECT id, label, entity_type, COALESCE(description, '') FROM entities WHERE valid_until IS NULL"
    ).fetchall()

    # All active relationships
    rel_where = "valid_until IS NULL"
    if min_confidence and min_confidence in CONFIDENCE_ORDER:
        min_val = CONFIDENCE_ORDER[min_confidence]
        rel_where += (
            f" AND CASE confidence WHEN 'extracted' THEN 3 WHEN 'inferred' THEN 2 ELSE 1 END >= {min_val}"
        )

    rel_rows = db.execute(
        f"SELECT source_id, target_id, relation_type, COALESCE(confidence, '{DEFAULT_CONFIDENCE}') "
        f"FROM relationships WHERE {rel_where}"
    ).fetchall()

    # Count linkCount per entity and build relationship list
    link_counts: dict[str, int] = {}
    relationships = []
    for src, tgt, rtype, conf in rel_rows:
        relationships.append({
            "source_id": src,
            "target_id": tgt,
            "relation_type": rtype,
            "confidence": conf,
        })
        link_counts[src] = link_counts.get(src, 0) + 1
        link_counts[tgt] = link_counts.get(tgt, 0) + 1

    # Build entity list with linkCount
    entities = []
    for eid, label, etype, desc in entity_rows:
        entities.append({
            "id": eid,
            "label": label,
            "type": etype,
            "description": desc[:500],
            "linkCount": link_counts.get(eid, 0),
        })

    return entities, relationships



def _generate_html_graph(
    query: str,
    nodes_json: str,
    links_json: str,
    has_private: bool = False,
    skip_limits: bool = False,
) -> str:
    """Generate an interactive HTML graph file and return its path.

    Applies hard limits, writes the file, and sets restrictive permissions.
    The HTML template is loaded from assets/template.html.

    Args:
        query: The original search query (used for filename).
        nodes_json: JSON string of node data.
        links_json: JSON string of link data.
        has_private: Whether the graph contains private-scope entities.

    Returns:
        Absolute path to the generated HTML file.

    Raises:
        SystemExit: If node or edge count exceeds hard limits.
    """
    num_nodes = sum(1 for _ in nodes_json.split('"id":')) - 1
    num_edges = sum(1 for _ in links_json.split('"source_id":')) - 1

    if not skip_limits:
        if num_nodes > HTML_GRAPH_MAX_NODES:
            raise SystemExit(
                f"Graph too large: {num_nodes} nodes (max {HTML_GRAPH_MAX_NODES}). "
                "Use a more specific query."
            )
        if num_edges > HTML_GRAPH_MAX_EDGES:
            raise SystemExit(
                f"Graph too large: {num_edges} edges (max {HTML_GRAPH_MAX_EDGES}). "
                "Use a more specific query."
            )

    # Filename: graph_<md5(query)[:8]>.html
    query_hash = hashlib.md5(query.encode("utf-8")).hexdigest()[:8]
    _GRAPH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = _GRAPH_OUTPUT_DIR / f"graph_{query_hash}.html"

    # Load template from file
    if not _HTML_TEMPLATE_PATH.exists():
        raise SystemExit(f"HTML template not found: {_HTML_TEMPLATE_PATH}")
    template_text = _HTML_TEMPLATE_PATH.read_text(encoding="utf-8")

    # Build combined data JSON
    data_json = json.dumps({
        "nodes": json.loads(nodes_json),
        "links": json.loads(links_json),
    }, ensure_ascii=False)

    # Replace template variables
    title_safe = html.escape(query)
    html_content = template_text.replace("{{TITLE}}", f"ActiveWiki Graph — {title_safe}")
    html_content = html_content.replace("{{DATA_JSON}}", data_json)
    html_content = html_content.replace("{{HAS_PRIVATE}}", "true" if has_private else "false")

    output_path.write_text(html_content, encoding="utf-8")
    os.chmod(str(output_path), 0o600)

    return str(output_path)


def cmd_graph_html(query: str, min_confidence: str | None = None, all_graph: bool = False) -> None:
    """Generate an interactive HTML graph for a query and print the output path.

    Args:
        query: Text to search for seed entities (None = full graph).
        min_confidence: Optional minimum confidence level for relationships.
        all_graph: If True, include ALL entities/relationships regardless of limits.
    """
    if not _graph_has_tables():
        log(
            "error",
            " Graph not built yet. Run `vectordb.py graph build` first.",
            stderr=True,
        )
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    try:
        if all_graph:
            # Full graph: no query, all entities + relationships
            entities, relationships = _build_full_graph_data(conn, min_confidence)
        else:
            if not query:
                log("error", " Query required for subgraph. Use --all for the full graph.", stderr=True)
                sys.exit(1)
            entities, relationships = _build_graph_data(conn, query, min_confidence)
    finally:
        conn.close()

    if not entities:
        if all_graph:
            log("info", "Graph is empty. Run `vectordb.py graph build` first.")
        else:
            log("info", f"No entities matching '{query}'")
        return

    nodes_json = json.dumps(entities, ensure_ascii=False)
    links_json = json.dumps(relationships, ensure_ascii=False)

    # Check for private-scope entities
    has_private = any(
        e.get("type") == "PERSON" or e.get("type") == "PROPERTY"
        for e in entities
    )

    output_path = _generate_html_graph(
        query=query or "Full Graph",
        nodes_json=nodes_json,
        links_json=links_json,
        has_private=has_private,
        skip_limits=all_graph,
    )

    print(output_path)


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
            f"AND r.valid_until IS NULL"
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

    # graph deduplicate (entity-level, embedding-based)
    sp_gd = graph_sub.add_parser("deduplicate", help="Embedding-based entity deduplication (merge semantic duplicates)")
    sp_gd.add_argument("--threshold", type=float, default=0.95,
                       help="Cosine similarity threshold for clustering (default: 0.95)")
    sp_gd.add_argument("--string-threshold", type=float, default=0.66,
                       help="Jaccard similarity threshold for string-based matching (default: 0.66)")
    sp_gd.add_argument("--dry-run", action="store_true", help="Only preview merges without applying them")
    sp_gd.add_argument("--json", action="store_true", help="Output report as JSON")

    # graph deduplicate-rels (relationship-level, fact conflict resolution)
    graph_sub.add_parser("deduplicate-rels", help="Deduplicate conflicting relationships (older → valid_until)")

    # graph cleanup
    graph_sub.add_parser("cleanup", help="Clean up orphaned relationships (missing source/target)")

    # graph search
    sp_gs = graph_sub.add_parser("search", help="Search the knowledge graph")
    sp_gs.add_argument("query")
    sp_gs.add_argument("--json", action="store_true", help="Output as JSON")
    sp_gs.add_argument("--scopes", type=str, default=None,
        help="Comma-separated list of scopes to filter by")
    sp_gs.add_argument("--with-rationale", action="store_true",
        help="Also show Rationale nodes (Why-Nodes) explaining facts")

    # graph pages (vector→graph bridge: entities anchored on given wiki pages)
    sp_gp = graph_sub.add_parser("pages", help="Entities (+1-hop) for given wiki pages")
    sp_gp.add_argument("--pages", type=str, required=True,
        help="Comma-separated wiki_page refs ('scope/slug'), highest-priority first")
    sp_gp.add_argument("--json", action="store_true", help="Output as JSON")
    sp_gp.add_argument("--scopes", type=str, default=None,
        help="Comma-separated scope allow-list (defensive filter)")

    # graph god-nodes (top entities by connection count)
    sp_gn = graph_sub.add_parser("god-nodes", help="Top entities by connection count (most connected)")
    sp_gn.add_argument("--top", type=int, default=10,
                       help="Number of top entities to show (default: 10)")

    # graph communities
    sp_gc = graph_sub.add_parser("communities", help="Community detection & summaries (Phase 2)"
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

    # graph html
    sp_gh = graph_sub.add_parser("html", help="Export an interactive HTML graph")
    sp_gh.add_argument("query", nargs="?", default=None, help="Search query to find seed entities (omit for full graph)")
    sp_gh.add_argument(
        "--all", "--full-graph",
        action="store_true",
        default=False,
        dest="all_graph",
        help="Include ALL entities and relationships (ignores hard limits)",
    )
    sp_gh.add_argument(
        "--min-confidence",
        choices=["extracted", "inferred", "weak"],
        default=None,
        help="Only include relationships at or above this confidence level",
    )
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
        elif gc == "deduplicate":
            if not DB_PATH.exists():
                log("error", " Database not found. Run `vectordb.py graph build` first.", stderr=True)
                sys.exit(1)
            threshold = getattr(args, "threshold", 0.95)
            string_threshold = getattr(args, "string_threshold", 0.5)
            dry_run = getattr(args, "dry_run", False)
            as_json = getattr(args, "json", False)
            conn = sqlite3.connect(DB_PATH)
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            init_db(conn)
            report = deduplicate_entities(conn, threshold=threshold, string_threshold=string_threshold, dry_run=dry_run)
            conn.close()
            if as_json:
                print(json.dumps(report, indent=2, ensure_ascii=False))
            else:
                if dry_run:
                    log("info", f"[DRY RUN] Would merge {report.get('would_merge', 0)} entities in {report.get('would_clusters', 0)} clusters")
                else:
                    log("info", f"Entity deduplication complete (cosine_threshold={threshold}, jaccard_threshold={string_threshold})")
                log("info", f"  Entities loaded: {report.get('entities_loaded', '?')}")
                log("info", f"  Embeddable: {report.get('embeddable', '?')}")
                log("info", f"  Clusters: {report.get('clusters', 0)}")
                log("info", f"  Merged: {report.get('merged', 0)}")
                log("info", f"  Redirects: {report.get('redirects', 0)}")
                if report.get("merges"):
                    log("info", f"  Merges:")
                    for m in report["merges"]:
                        log("info", f"    '{m['duplicate_label']}' → '{m['canonical_label']}' "
                            f"(method={m['match_method']}, cos={m['cosine_similarity']}, jac={m['jaccard_similarity']})")
        elif gc == "deduplicate-rels":
            if not DB_PATH.exists():
                log("error", " Database not found. Run `vectordb.py graph build` first.", stderr=True)
                sys.exit(1)
            conn = sqlite3.connect(DB_PATH)
            init_db(conn)
            deduped = _deduplicate_relationships(conn)
            conn.commit()
            conn.close()
            log("info", f"Deduplicated {deduped} relationships")
        elif gc == "cleanup":
            if not DB_PATH.exists():
                log("error", " Database not found. Run `vectordb.py graph build` first.", stderr=True)
                sys.exit(1)
            conn = sqlite3.connect(DB_PATH)
            init_db(conn)
            cleaned = _cleanup_orphaned_relationships(conn)
            conn.commit()
            conn.close()
            log("info", f"Cleaned up {cleaned} orphaned relationships")
        elif gc == "search":
            scopes = None
            if getattr(args, "scopes", None):
                scopes = [s.strip() for s in args.scopes.split(",")]
            with_rat = getattr(args, "with_rationale", False)
            cmd_graph_search(args.query, as_json=getattr(args, "json", False), scopes=scopes, with_rationale=with_rat)
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
        elif gc == "god-nodes":
            if not _graph_has_tables():
                log("error", " Graph tables not found. Run `vectordb.py graph build` first.", stderr=True)
                sys.exit(1)
            top_n = getattr(args, "top", 10)
            cmd_graph_god_nodes(top_n=top_n)
        elif gc == "html":
            min_conf = getattr(args, "min_confidence", None)
            all_flag = getattr(args, "all_graph", False)
            cmd_graph_html(args.query, min_confidence=min_conf, all_graph=all_flag)
        else:
            ap.print_help()

    return 0


if __name__ == "__main__":
    sys.exit(main())

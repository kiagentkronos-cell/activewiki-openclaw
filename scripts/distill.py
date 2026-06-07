#!/usr/bin/env python3
"""ActiveWiki: distill Docling sources into hierarchical wiki pages (scope-isolated).

Scope (private|family|public) is determined by the parent dir of the source
(sources/<scope>/<source-id>/). Distillation is strictly scope-isolated: a
private source is never merged into a family or public wiki page, and vice
versa. Each scope maintains its own wiki/ tree and index files.

Hierarchical structure (bottom-up):
  - Each inbox folder (depth >= 1 under scope) gets its own wiki page.
  - Leaf folders: direct sources → wiki page (distill phase).
  - Parent folders: child wiki pages → parent wiki page (rollup phase).
  - Sources without inbox_path fall back to legacy flat-page mode.

Phases:
  1. DISTILL: Process each unprocessed source → target folder's wiki page.
     Only the target page (not all wiki pages) is loaded into the LLM prompt.
  2. ROLLUP: Bottom-up traversal of inbox folder hierarchy.
     Each folder's wiki page is synthesized from its direct children's wiki pages.

Chunking: Large sources (>80 000 chars) are split into ~50 000-char chunks
and processed serially through the LLM so no information is lost.

Configuration via activewiki.json (see activewiki.example.json).
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import hashlib
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from datetime import date, timezone
from pathlib import Path
from typing import Any

import yaml

# ── Config loading ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    load_config, get,
    wikis_root, scopes,
    llm_model, llm_url, llm_temperature, llm_max_tokens,
)

_CONFIG = load_config()
_WIKIS_ROOT = wikis_root(_CONFIG)
_SCOPES = scopes(_CONFIG)

# Tolerate filenames/strings carrying PEP 383 surrogate escapes.
sys.stdout.reconfigure(errors="backslashreplace")
sys.stderr.reconfigure(errors="backslashreplace")

# ── Paths (derived from config) ─────────────────────────────────────────────
SOURCES = _WIKIS_ROOT / "sources"
WIKI = _WIKIS_ROOT / "wiki"
INBOX = _WIKIS_ROOT / "inbox"

# ── LLM Config (from activewiki.json, overridable via env) ──────────────────
MODEL = os.environ.get("ACTIVEWIKI_MODEL", llm_model(_CONFIG))
LLM_BASE_URL = os.environ.get("OLLAMA_URL", llm_url(_CONFIG))
HTTP_TIMEOUT = int(os.environ.get("ACTIVEWIKI_HTTP_TIMEOUT", "3600"))

# Chunking thresholds for large sources
CHUNK_SIZE = int(os.environ.get("ACTIVEWIKI_CHUNK_SIZE", "50000"))
CHUNK_TRIGGER = int(os.environ.get("ACTIVEWIKI_CHUNK_TRIGGER", "80000"))

# Auto-Split: page-size limit that triggers creation of a companion page
WIKI_MAX_PAGE_CHARS = int(os.environ.get("WIKI_MAX_PAGE_CHARS", "25000"))
WIKI_SPLIT_ENABLED = os.environ.get("WIKI_SPLIT_ENABLED", "true").lower() in ("true", "1", "yes")

# ── System Prompts ──────────────────────────────────────────────────────────

SYSTEM_PROMPT_DISTILL_PAGE = """Du bist ein Redakteur für ein persönliches Wissens-Repo.

Eingabe:
- Eine existierende Wiki-Seite (oder der Hinweis, dass sie neu erstellt wird).
- Volltext einer neuen Quelle (Docling-Output eines Dokuments).

Aufgabe: Integriere die neue Quelle in die Wiki-Seite.
- Wenn die Seite existiert: Erweitere den Inhalt um die neuen Informationen.
  Behalte den bestehenden Inhalt bei — nichts weglassen.
- Wenn die Seite neu erstellt wird: Erstelle eine gut strukturierte Wiki-Seite
  aus der neuen Quelle.

Regeln:
- slug: kebab-case (a-z, 0-9, Bindestrich), 3-60 Zeichen, beschreibend.
  Verwende den übergebenen Ziel-Slug.
- topics: 1-5 kurze Tags in kebab-case.
- Sprache: Deutsch, neutraler Sachton, keine Spekulation außerhalb der Quellen.
- Struktur: Markdown mit Überschriften (##), Listen, Tabellen nach Bedarf.
- Keine Frontmatter in content_md — die setzt das Skript.

Antworte ausschließlich mit einem JSON-Objekt:
{"type": "create"|"update",
 "slug": "...",
 "title": "...",
 "summary": "...",
 "topics": [...],
 "content_md": "..."}
"""

# Appended to SYSTEM_PROMPT_DISTILL_PAGE when the target page has reached
# WIKI_MAX_PAGE_CHARS and a companion page should be created.
SYSTEM_PROMPT_DISTILL_PAGE_SPLIT_INSTRUCTION = """

⚠️ SEITEN-SPLIT AKTIVIERT ⚠️
Die Ziel-Seite hat das maximale Seitenvolumen erreicht (≥ 20.000 Zeichen).
Statt die bestehende Seite weiter aufzublähen, erstelle eine Ergänzungsseite.

Verfahren:
1. Komprimiere die bestehende Seite (Seite 1) auf max. 18.000 Zeichen:
   - Behalte die wichtigsten Fakten und Struktur
   - Entferne redundante Details, fasse zusammen
   - Füge am Ende einen "Siehe auch:"-Link zur neuen Seite ein
2. Erstelle eine neue Ergänzungsseite (Seite 2) mit dem neuen Inhalt
   und allen Details die nicht in die komprimierte Seite 1 passen
3. Die neue Seite bekommt denselben Slug mit "-2" Suffix (z.B. "mein-thema-2")
4. Auf Seite 2 ebenfalls einen "Siehe auch:"-Link zurück zu Seite 1

WICHTIG: Seite 1 MUSST nach deiner Bearbeitung unter 20.000 Zeichen bleiben.
Das ist eine harte Grenze. Wenn du das nicht schaffst, ist der Split fehlgeschlagen.

Antworte mit einem JSON-Objekt das ein "actions"-Array enthält:
{"actions": [
  {"type": "update", "slug": "...", "title": "...", "summary": "...",
   "topics": [...], "content_md": "...", "related_slugs": ["...-2"]},
  {"type": "create", "slug": "...-2", "title": "...", "summary": "...",
   "topics": [...], "content_md": "...", "related_slugs": ["..."]}  ]}
"""

SYSTEM_PROMPT_DISTILL_LEGACY = """Du bist ein Redakteur für ein persönliches Wissens-Repo.

Eingabe:
- Liste existierender Wiki-Seiten im selben Scope (Slug, Titel, Summary, Topics, Volltext).
- Volltext einer neuen Quelle (Docling-Output eines Dokuments).

Aufgabe: Entscheide pro Thema, ob der neue Inhalt zu einer existierenden Seite
passt (UPDATE) oder ein neues Thema ist (CREATE). Eine Quelle kann mehrere
Actions auslösen, wenn sie thematisch mehrere Seiten berührt.

Regeln:
- UPDATE: content_md muss das bisherige Material der Seite vollständig enthalten
  und um die neuen Informationen ergänzt sein. Nichts weglassen.
- CREATE: slug ist kebab-case (a-z, 0-9, Bindestrich), 3-60 Zeichen, beschreibend.
- UPDATE: slug ist exakt der Slug der existierenden Seite.
- topics: 1-5 kurze Tags in kebab-case.
- Sprache: Deutsch, neutraler Sachton, keine Spekulation außerhalb der Quellen.
- Struktur: Markdown mit Überschriften (##), Listen, Tabellen nach Bedarf.
- Keine Frontmatter in content_md — die setzt das Skript.

Antworte ausschließlich mit einem JSON-Objekt:
{"actions": [
  {"type": "create"|"update",
   "slug": "...",
   "title": "...",
   "summary": "...",
   "topics": [...],
   "content_md": "..."}
]}
"""

SYSTEM_PROMPT_ROLLUP = """Du bist ein Redakteur für ein persönliches Wissens-Repo.

Eingabe:
- Wiki-Seiten der direkten Unterordner (Children) eines Ordners.
- Eventuell Zusammenfassungen von Quellen, die direkt in diesem Ordner liegen.

Aufgabe: Synthesisiere eine übergeordnete Wiki-Seite für den Elternordner.
Diese Seite fasst die Inhalte der Child-Seiten zusammen — auf einer höheren
Abstraktionsebene.

Regeln:
- Dies ist KEINE einfache Konkatenation der Child-Seiten.
- Erzeuge eine kohärente Zusammenfassung auf Parent-Ebene.
- Wichtige Fakten, Zahlen, Namen hervorheben.
- Querverweise auf verwandte Themen einbauen.
- Wenn es nur eine Child-Seite gibt: fasse deren Inhalt kompakter zusammen.
- topics: 1-5 kurze Tags in kebab-case.
- Sprache: Deutsch, neutraler Sachton.
- Struktur: Markdown mit Überschriften (##), Listen nach Bedarf.
- Keine Frontmatter in content_md — die setzt das Skript.

Antworte ausschließlich mit einem JSON-Objekt:
{"slug": "...",
 "title": "...",
 "summary": "...",
 "topics": [...],
 "content_md": "..."}
"""


# ── Timestamped Logging ─────────────────────────────────────────────────────

def log(level: str, message: str, *, stderr: bool = False) -> None:
    """Print a log line with ISO-8601 Berlin timestamp."""
    from zoneinfo import ZoneInfo
    ts = datetime.datetime.now(ZoneInfo("Europe/Berlin")).strftime("%Y-%m-%dT%H:%M:%S %Z")
    line = f"[{ts}] [{level}] {message}"
    print(line, file=sys.stderr if stderr else sys.stdout)


# ── Serial Chunking ─────────────────────────────────────────────────────────

def split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    """Split a large markdown document into logical chunks.

    Prefers cutting at ``## `` heading boundaries so each chunk starts at a
    section boundary.  Falls back to paragraph boundaries, then hard splits.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    remaining = text

    while remaining:
        if len(remaining) <= chunk_size:
            chunks.append(remaining)
            break

        # Target cut position (leave some margin for safety)
        cut = chunk_size - 200
        if cut < 100:
            cut = chunk_size

        # Strategy 1: next ## heading within ±200 chars of cut
        best = -1
        for m in re.finditer(r"^##\s", remaining[cut - 200:cut + 400], re.MULTILINE):
            pos = cut - 200 + m.start()
            if best < 0 or abs(pos - cut) < abs(best - cut):
                best = pos

        if best >= 0:
            # Cut *before* the heading so it starts the next chunk
            chunks.append(remaining[:best].rstrip())
            remaining = remaining[best:]
            continue

        # Strategy 2: blank-line (paragraph) boundary near cut
        para = remaining.rfind("\n\n", 0, cut + 100)
        if para >= cut - 200:
            chunks.append(remaining[:para].rstrip())
            remaining = remaining[para:].lstrip("\n")
            continue

        # Strategy 3: single newline
        nl = remaining.rfind("\n", 0, cut + 100)
        if nl >= cut - 200:
            chunks.append(remaining[:nl].rstrip())
            remaining = remaining[nl:].lstrip("\n")
            continue

        # Strategy 4: hard cut (worst case)
        chunks.append(remaining[:cut])
        remaining = remaining[cut:]

    return chunks


def distill_source_chunked(
    scope: str,
    source_id: str,
    source_md: str,
    target_slug: str | None,
    existing_page: dict[str, Any] | None,
    inbox_path: list[str],
    original_name: str | None,
) -> tuple[list[dict[str, Any]], str]:
    """Distill a source using serial chunking for large documents.

    - If ``len(source_md) <= CHUNK_TRIGGER``: single LLM call (normal path).
    - Otherwise: split into chunks, process serially.
      Chunk 1 → CREATE page; Chunk 2..N → UPDATE page incrementally.

    **Chunked-Split:** If the accumulated wiki body exceeds WIKI_MAX_PAGE_CHARS
    and more chunks remain, the next chunks spill over into a new companion page
    (slug with ``-2``, ``-3`` … suffix). All split pages are cross-linked via
    ``related_slugs`` in frontmatter and "Siehe auch:" links in the body.

    Returns ``(actions, final_body)``.  ``actions`` is populated for the
    single-call path so the caller can apply them normally.  In chunked mode
    the page is written after each chunk, and ``final_body`` holds the
    accumulated content of the *last* page written.
    """
    if len(source_md) <= CHUNK_TRIGGER:
        # ── Single call (normal) ──
        if target_slug and existing_page:
            prompt = build_prompt_page_update(
                target_slug=target_slug,
                target_title=existing_page["frontmatter"].get("title", ""),
                target_body=existing_page["body"],
                source_id=source_id,
                source_md=source_md,
                original_name=original_name,
                inbox_path=inbox_path,
            )
        elif target_slug:
            prompt = build_prompt_page_create(
                target_slug=target_slug,
                source_id=source_id,
                source_md=source_md,
                original_name=original_name,
                inbox_path=inbox_path,
            )
        else:
            # Legacy: caller must supply pages dict separately
            raise ValueError("Legacy mode requires pages dict — use distill_source_legacy()")

        result = call_llm(SYSTEM_PROMPT_DISTILL_PAGE, prompt, source_id=source_id)
        atype = result.get("type", "update" if existing_page else "create")
        if atype not in ("create", "update"):
            atype = "update" if existing_page else "create"
        return [result], result.get("content_md", "")

    # ── Chunked processing ──
    chunks = split_into_chunks(source_md)
    num_chunks = len(chunks)
    log("info", f"chunking {scope}/{source_id}: {num_chunks} chunks ({len(source_md):,} chars total)")

    wiki_body = ""
    last_result: dict[str, Any] | None = None
    first_title = ""
    today = date.today().isoformat()

    # ── Chunked-Split state ──
    # current_slug tracks which page we're currently writing to. Starts as
    # target_slug; when a split fires it advances to -2, -3, …
    current_slug: str = target_slug or "unknown"
    # Collect all slugs involved in this chunked run for cross-linking
    split_slugs: list[str] = [current_slug]

    for ci, chunk in enumerate(chunks):
        chunk_label = f"{source_id}::chunk{ci + 1}/{num_chunks}"
        log("info", f"processing chunk {ci + 1}/{num_chunks} for {scope}/{source_id} ({len(chunk):,} chars)")

        if ci == 0:
            # CREATE from first chunk
            prompt = build_prompt_page_create(
                target_slug=current_slug,
                source_id=chunk_label,
                source_md=chunk,
                original_name=original_name,
                inbox_path=inbox_path,
            )
            result = call_llm(SYSTEM_PROMPT_DISTILL_PAGE, prompt, source_id=chunk_label)
            wiki_body = result.get("content_md", "").rstrip()
            first_title = (result.get("title") or "").strip()
        else:
            # UPDATE: append new chunk to accumulated page
            prompt = build_prompt_chunk_update(
                target_slug=current_slug,
                existing_body=wiki_body,
                source_id=chunk_label,
                source_md=chunk,
                original_name=original_name,
                inbox_path=inbox_path,
            )
            result = call_llm(SYSTEM_PROMPT_DISTILL_PAGE, prompt, source_id=chunk_label)
            wiki_body = result.get("content_md", "").rstrip()

        last_result = result

        # ── Write page to disk after each chunk ──
        # Chunk 1 creates the page; chunks 2..N overwrite with accumulated content.
        # This guarantees the page exists on disk when the caller processes actions.
        # BUGFIX 3: Split pages get a suffixed title ("Titel (Teil 2)")
        base_title = first_title if first_title else (result.get("title") or "").strip()
        if current_slug != (target_slug or "unknown"):
            # Extract part number from slug suffix (-2 → Teil 2)
            m_suffix = re.search(r"-(\d+)$", current_slug)
            part_num = int(m_suffix.group(1)) if m_suffix else 2
            cur_title = f"{base_title} (Teil {part_num})"
        else:
            cur_title = base_title
        cur_summary = " ".join((result.get("summary") or "").split())
        cur_topics = [
            str(t).strip() for t in (result.get("topics") or [])
            if str(t).strip()
        ]

        # Compute related_slugs: all OTHER slugs in this run (not the current one)
        related = [s for s in split_slugs if s != current_slug]

        # 🔴 FIX: Load existing sources and merge — don't overwrite!
        existing_page_for_sources = load_page(scope, current_slug)
        old_sources = list((existing_page_for_sources["frontmatter"].get("sources") or []) if existing_page_for_sources else [])
        if source_id not in old_sources:
            old_sources.append(source_id)

        write_page(
            scope, current_slug,
            title=cur_title, summary=cur_summary, topics=cur_topics,
            sources=old_sources, created=today, updated=today,
            body=wiki_body,
            folder_path="/".join(inbox_path) if inbox_path else None,
            related_slugs=related if related else None,
        )
        log("info",
            f"chunk {ci + 1}/{num_chunks} done → {len(wiki_body):,} chars "
            f"(written to wiki/{scope}/{current_slug}.md)")

        # ── Chunked-Split: check if we need to split before next chunk ──
        # Only trigger if: split enabled, body too large, AND more chunks follow
        if ci < num_chunks - 1 and _page_needs_split(wiki_body):
            next_slug = _next_split_slug(current_slug, scope)
            log("split",
                f"chunked-split triggered after chunk {ci + 1}/{num_chunks}: "
                f"{len(wiki_body):,} chars (≥ {WIKI_MAX_PAGE_CHARS:,}) → "
                f"next chunks go to {next_slug}")
            current_slug = next_slug
            split_slugs.append(current_slug)
            wiki_body = ""  # reset body for the new companion page

    # ── BUGFIX 1: Final size check for last chunk ──
    # The loop above skips the split check for the last chunk (ci == num_chunks - 1).
    # If the final body still exceeds the limit, warn but don't auto-split
    # (no chunks left to distribute).
    if num_chunks > 1 and _page_needs_split(wiki_body):
        log("warn",
            f"final chunk of {scope}/{source_id} produced oversized page "
            f"({len(wiki_body):,} chars ≥ {WIKI_MAX_PAGE_CHARS:,}); "
            f"cannot split further — page will exceed limit")

    # ── Final cross-link pass: add "Siehe auch:" links to ALL split pages ──
    if len(split_slugs) > 1:
        for slug in split_slugs:
            page = load_page(scope, slug)
            if page is None:
                continue
            other_slugs = [s for s in split_slugs if s != slug]
            if not other_slugs:
                continue
            new_body = _add_related_links(page["body"], other_slugs)
            if new_body != page["body"]:
                fm = page["frontmatter"]
                # Merge related_slugs into frontmatter
                existing_related = list(fm.get("related_slugs") or [])
                for rs in other_slugs:
                    if rs not in existing_related:
                        existing_related.append(rs)
                write_page(
                    scope, slug,
                    title=fm.get("title", slug),
                    summary=" ".join((fm.get("summary") or "").split()),
                    topics=fm.get("topics") or [],
                    sources=fm.get("sources") or [],
                    created=fm.get("created") or today,
                    updated=today,
                    body=new_body,
                    folder_path=fm.get("folder_path"),
                    related_slugs=existing_related if existing_related else None,
                )
                log("info", f"cross-linked {scope}/{slug} ↔ {other_slugs}")

    # Return a special action type so the caller knows the page is already
    # on disk — it just needs to register the source_id and refresh its
    # in-memory pages dict.
    # Note: final_body is the body of the LAST page written (may be a split page).
    # The caller should use target_slug for the primary page registration.
    return ([{"type": "chunked_complete"}] if last_result else []), wiki_body


def build_prompt_chunk_update(
    target_slug: str,
    existing_body: str,
    source_id: str,
    source_md: str,
    original_name: str | None = None,
    inbox_path: list[str] | None = None,
) -> str:
    """Build prompt for incrementally extending an existing wiki page with a new chunk."""
    parts: list[str] = []
    parts.append(f"## Ziel-Wiki-Seite erweitern: {target_slug}")
    if inbox_path:
        parts.append(f"**Ordner-Pfad:** `/{'/'.join(inbox_path)}/`")
    parts.append("")
    parts.append("### Aktueller Inhalt der Seite (bereits aus früheren Chunks):")
    parts.append("\n```markdown")
    parts.append(existing_body.strip())
    parts.append("```\n")

    parts.append(f"\n## Neuer Quell-Chunk: {source_id}\n")
    if original_name:
        parts.append(f"**Originaldatei:** {original_name}")
    parts.append("\n```markdown")
    parts.append(source_md.strip())
    parts.append("```")
    return "\n".join(parts)


# ── Auto-Split Helpers ──────────────────────────────────────────────────────

def _pick_target_companion(base_slug: str, pages: dict) -> str:
    """Return the slug with the smallest body among base_slug and its companions.

    When a primary page has been split into companions (slug-2, slug-3…),
    new sources should be appended to the *smallest* existing page rather
    than always inflating the primary. This prevents the primary from
    exceeding LLM context limits.

    Parameters
    ----------
    base_slug :
        The canonical slug derived from inbox_path (e.g. "mein-thema").
    pages :
        Already-loaded wiki pages dictionary (slug → page dict).

    Returns
    -------
    str
        Slug of the page with the smallest body.
    """
    candidates = [(base_slug, pages.get(base_slug))]
    n = 2
    while True:
        s = f"{base_slug}-{n}"
        p = pages.get(s)
        if p is None:
            break  # no more companions
        candidates.append((s, p))
        n += 1

    # Filter out missing pages and pick the one with smallest body
    valid = [(s, pg) for s, pg in candidates if pg is not None]
    if len(valid) == 0:
        return base_slug  # no page exists yet → caller will create one
    if len(valid) == 1:
        return base_slug  # only primary exists

    best_slug = min(valid, key=lambda pair: len(pair[1].get("body", "")))[0]
    return best_slug


def _next_split_slug(base_slug: str, scope: str, pages: dict | None = None) -> str:
    """Compute the next flat split-slug for a base slug within a single scope.

    Finds the ROOT page (everything before the first `-<digit>` suffix) and
    scans all existing companion pages (filesystem + in-memory ``pages`` dict)
    to find the highest -N suffix, then returns root-(N+1).

    Flat numbering: ``haus-musterstr-32-anleitungen`` → ``-2``, ``-3``, ``-4``, …
    NOT tree: ``-3-2``, ``-3-2-2``, …

    Examples:
        "mein-thema"          → "mein-thema-2"
        "mein-thema-2"        → "mein-thema-3"  (if -2 exists)
        "mein-thema-5"        → "mein-thema-6"  (flat, not "mein-thema-5-2")
    """
    # 1. Extract root slug: everything before the first `-<digit>` suffix
    root_match = re.match(r"^(.+)-\d+$", base_slug)
    if root_match:
        root_slug = root_match.group(1)
    else:
        root_slug = base_slug

    # 2. Scan for highest -N suffix on the root
    pattern = re.compile(rf"^{re.escape(root_slug)}-(\d+)$")
    max_n = 1  # start at 2

    # Check filesystem
    scope_path = WIKI / scope
    if scope_path.exists():
        for p in scope_path.glob("*.md"):
            m = pattern.match(p.stem)
            if m:
                n = int(m.group(1))
                if n >= max_n:
                    max_n = n

    # Also check in-memory pages dict for recently-created companions
    if pages is not None:
        for slug_key in pages:
            m = pattern.match(slug_key)
            if m:
                n = int(m.group(1))
                if n >= max_n:
                    max_n = n

    return f"{root_slug}-{max_n + 1}"


def _add_related_links(body: str, related_slugs: list[str]) -> str:
    """Append or update 'Siehe auch:' cross-links at the bottom of a wiki page body.

    FIX #3: Idempotent — if a 'Siehe auch:' block already exists, it is replaced
    rather than duplicated on repeated calls.
    """
    if not related_slugs:
        return body

    links = " ".join(f"[{s}]({s})" for s in related_slugs)
    new_block = f"\n\n---\n\n**Siehe auch:** {links}\n"

    # FIX #3: Remove any existing "Siehe auch:" block before appending
    # Tolerant pattern: handles "**Siehe auch :**", "### Siehe auch", etc.
    body_clean = re.sub(
        r'(?:\n+(?:---|#+\s*)?\s*\n?\s*\*\*Siehe auch\s*:\*\*[^\n]*)\s*\n*$',
        "",
        body.rstrip(),
    )

    if not body_clean.endswith("\n"):
        body_clean += "\n"

    return body_clean + new_block


def _page_needs_split(body: str) -> bool:
    """Check whether a wiki page body exceeds the split threshold."""
    if not WIKI_SPLIT_ENABLED:
        return False
    return len(body) >= WIKI_MAX_PAGE_CHARS


def _parse_split_actions(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract action list from an LLM response that may be single-action or split-actions.

    Returns a list of action dicts.  If the response has an "actions" key,
    uses that; otherwise wraps the single response in a list.
    """
    if "actions" in result and isinstance(result["actions"], list):
        return result["actions"]
    return [result]


def _get_action_body_for_slug(actions: list[dict[str, Any]], slug: str) -> str | None:
    """Find the body content for a given slug in a list of actions.

    Returns the body string if found, None otherwise.
    Checks both 'content_md' and 'body' keys.
    """
    for a in actions:
        a_slug = (a.get("slug") or "").strip()
        if safe_slug(a_slug) == slug:
            return (a.get("content_md") or a.get("body") or "").rstrip()
    return None


# ── Regex Patterns ──────────────────────────────────────────────────────────

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n?(.*)$", re.DOTALL)
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,118}[a-z0-9]$")


def safe_slug(slug: str, max_len: int = 120) -> str:
    """Ensure slug fits SLUG_RE. Sanitize and truncate with hash if needed.

    Never rejects — always returns a valid slug.
    """
    # Strip trailing dashes (regex requires [a-z0-9] at end)
    slug = slug.rstrip("-")
    if len(slug) <= max_len and SLUG_RE.match(slug):
        return slug
    if len(slug) <= max_len:
        # Valid length but bad chars — should not happen with proper slugify,
        # but guard anyway
        return slug
    # Too long: truncate with hash suffix
    h = hashlib.sha256(slug.encode()).hexdigest()[:8]
    suffix = f"--{h}"  # '--' is valid in [a-z0-9-]; underscores are not
    prefix_len = max_len - len(suffix)
    prefix = slug[:prefix_len].rstrip("-")
    if len(prefix) + len(suffix) < 2:
        prefix = "a"
    return f"{prefix}{suffix}"


# ── Page I/O ────────────────────────────────────────────────────────────────

def parse_page(path: Path) -> tuple[dict[str, Any], str]:
    """Parse a wiki .md file into (frontmatter_dict, body_string)."""
    text = path.read_text(encoding="utf-8")
    m = FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    fm = yaml.safe_load(m.group(1)) or {}
    return fm, m.group(2)


def load_page(scope: str, slug: str) -> dict[str, Any] | None:
    """Load a single wiki page by slug. Returns None if not found."""
    path = WIKI / scope / f"{slug}.md"
    if not path.exists():
        return None
    fm, body = parse_page(path)
    return {"frontmatter": fm, "body": body}


def write_page(
    scope: str,
    slug: str,
    *,
    title: str,
    summary: str,
    topics: list[str],
    sources: list[str],
    created: str,
    updated: str,
    body: str,
    folder_path: str | None = None,
    related_slugs: list[str] | None = None,
    rollup_hash: str | None = None,
) -> None:
    """Write a wiki page with frontmatter + markdown body.

    folder_path is always stored as a '/'-joined string in frontmatter.
    related_slugs stores cross-references to split companion pages.
    rollup_hash stores the SHA-256 of rollup inputs for change detection.
    """
    frontmatter: dict[str, Any] = {
        "title": title,
        "summary": summary,
        "topics": topics,
        "sources": sources,
        "scope": scope,
        "created": created,
        "updated": updated,
    }
    if folder_path is not None:
        frontmatter["folder_path"] = folder_path
    if related_slugs:
        frontmatter["related_slugs"] = related_slugs
    if rollup_hash:
        frontmatter["rollup_hash"] = rollup_hash
    fm_yaml = yaml.safe_dump(
        frontmatter, allow_unicode=True, sort_keys=False, default_flow_style=False
    )
    content = f"---\n{fm_yaml}---\n{body.rstrip()}\n"
    (WIKI / scope / f"{slug}.md").write_text(content, encoding="utf-8")


def load_wiki_pages(scope: str) -> dict[str, dict[str, Any]]:
    """Load ALL wiki pages of a scope. Used for legacy mode and index rebuilds."""
    pages: dict[str, dict[str, Any]] = {}
    scope_dir = WIKI / scope
    if not scope_dir.exists():
        return pages
    for p in sorted(scope_dir.glob("*.md")):
        fm, body = parse_page(p)
        pages[p.stem] = {"frontmatter": fm, "body": body}
    return pages


# ── Slug Helpers ────────────────────────────────────────────────────────────

def normalize_slug_component(name: str) -> str:
    """Convert a folder/file name to a kebab-case slug component.

    Handles unicode (ö→oe, ü→ue, ß→ss), strips non-alphanumeric chars.
    """
    german_map = str.maketrans({
        "ö": "oe", "Ö": "Oe",
        "ü": "ue", "Ü": "Ue",
        "ä": "ae", "Ä": "Ae",
        "ß": "ss",
        "é": "e",  "É": "E",
        "è": "e",  "È": "E",
        "ê": "e",  "Ê": "E",
        "ô": "o",  "Ô": "O",
        "î": "i",  "Î": "I",
        "û": "u",  "Û": "U",
        "ç": "c",  "Ç": "C",
        "ñ": "n",  "Ñ": "N",
    })
    mapped = name.translate(german_map)

    nfd = unicodedata.normalize("NFD", mapped)
    ascii_str = nfd.encode("ascii", "ignore").decode("ascii")

    slug = re.sub(r"[^a-z0-9]+", "-", ascii_str.lower()).strip("-")
    return slug or "untitled"


def inbox_path_to_slug(inbox_path: list[str]) -> str:
    """Convert an inbox_path list to a hierarchical wiki slug.

    E.g. ["Immobilieninvestments", "Musterort", "Bilder", "Drohne"]
         → "immobilieninvestments-musterort-bilder-drohne"
    """
    if not inbox_path:
        return ""
    components = [normalize_slug_component(p) for p in inbox_path]
    return "-".join(components)


# ── Source Metadata ─────────────────────────────────────────────────────────

def get_source_metadata(source_id: str, scope: str) -> dict[str, Any]:
    """Read metadata.json for a source."""
    meta_path = SOURCES / scope / source_id / "metadata.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {}


def load_all_source_metadata(scope: str) -> dict[str, dict[str, Any]]:
    """Load metadata for ALL sources in a scope. Cached to avoid O(N) scans."""
    result: dict[str, dict[str, Any]] = {}
    scope_dir = SOURCES / scope
    if not scope_dir.exists():
        return result
    for meta_path in sorted(scope_dir.glob("*/metadata.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        sid = meta.get("source_id", meta_path.parent.name)
        result[sid] = meta
    return result


def get_folder_slug_for_source(source_id: str, scope: str) -> str | None:
    """Determine the target wiki page slug from a source's inbox_path.

    Returns None if the source has no inbox_path (legacy mode).
    """
    meta = get_source_metadata(source_id, scope)
    inbox_path = meta.get("inbox_path")
    if not inbox_path:
        return None
    return inbox_path_to_slug(inbox_path)


def write_distilled_metadata(
    source_id: str,
    scope: str,
    target_page: str | None,
    distilled: bool = True,
) -> None:
    """Write distilled status to a source's metadata.json.

    Atomic write: backs up the original file, writes new content to a
    temporary file in the same directory, then ``os.replace()`` for
    atomic swap. On interrupt the original file remains intact.
    """
    meta_path = SOURCES / scope / source_id / "metadata.json"
    if not meta_path.exists():
        return

    # Read, merge
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["distilled"] = distilled
    if distilled:
        meta["distilled_at"] = datetime.datetime.now(timezone.utc).isoformat()
    if target_page is not None:
        meta["target_page"] = target_page

    # Atomic write: temp file in same directory → os.replace
    temp_path = meta_path.with_suffix(".tmp")
    temp_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temp_path), str(meta_path))


def _migrate_existing_metadata(scope: str) -> tuple[int, int]:
    """One-time migration: populate ``distilled`` field in all metadata.json files.

    For each source without a ``distilled`` field, checks whether it is
    referenced in any wiki page's frontmatter ``sources`` list.

    Returns ``(distilled_count, not_distilled_count)``.
    """
    pages = load_wiki_pages(scope)
    scope_dir = SOURCES / scope
    if not scope_dir.exists():
        return (0, 0)

    migrated = 0
    distilled_count = 0
    not_distilled_count = 0

    for meta_path in sorted(scope_dir.glob("*/metadata.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        # Skip if already has distilled field
        if "distilled" in meta:
            continue

        source_id = meta.get("source_id", meta_path.parent.name)

        # Check if referenced in any wiki page
        is_referenced = any(
            source_id in (p["frontmatter"].get("sources") or [])
            for p in pages.values()
        )

        # Find target page slug if referenced
        target_page = None
        if is_referenced:
            for slug, p in pages.items():
                if source_id in (p["frontmatter"].get("sources") or []):
                    target_page = slug
                    break



        # Update metadata
        meta["distilled"] = is_referenced
        if is_referenced:
            meta["distilled_at"] = datetime.datetime.now(timezone.utc).isoformat()
            distilled_count += 1
        else:
            not_distilled_count += 1
        if target_page is not None:
            meta["target_page"] = target_page

        # Atomic write: temp file in same directory → os.replace
        temp_path = meta_path.with_suffix(".tmp")
        temp_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(str(temp_path), str(meta_path))
        migrated += 1

    if migrated > 0:
        log("migration",
            f"{scope}: migrated {migrated} metadata.json files "
            f"({distilled_count} distilled, {not_distilled_count} not distilled)")

    return (distilled_count, not_distilled_count)


def _any_source_needs_migration(scope: str) -> bool:
    """Check if any source in scope lacks a ``distilled`` field."""
    scope_dir = SOURCES / scope
    if not scope_dir.exists():
        return False
    for meta_path in scope_dir.glob("*/metadata.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if "distilled" not in meta:
            return True
    return False


# ── Prompt Builders ─────────────────────────────────────────────────────────

def build_prompt_legacy(
    source_id: str,
    source_md: str,
    pages: dict[str, dict],
    inbox_path: list[str] | None = None,
    original_name: str | None = None,
) -> str:
    """Build prompt for legacy mode (all wiki pages + new source)."""
    parts: list[str] = ["## Existierende Wiki-Seiten\n"]
    if pages:
        for slug, p in pages.items():
            fm = p["frontmatter"]
            parts.append(f"### slug: {slug}")
            parts.append(f"title: {fm.get('title', '')}")
            parts.append(f"summary: {fm.get('summary', '')}")
            parts.append(f"topics: {fm.get('topics', [])}")
            parts.append(f"updated: {fm.get('updated', '')}")
            parts.append("\n```markdown")
            parts.append(p["body"].strip())
            parts.append("```\n")
    else:
        parts.append("(noch keine Seiten)\n")
    parts.append(f"\n## Neue Quelle: {source_id}\n")
    if original_name:
        parts.append(f"**Originaldatei:** {original_name}")
    if inbox_path:
        parts.append(
            f"**Quellen-Ordner-Hinweis:** `/{'/'.join(inbox_path)}/` — "
            "Pfad innerhalb der Inbox; nutze das als Hinweis auf Thema/"
            "Kategorie/Jahr, wenn der Dokumentinhalt selbst unklar ist. "
            "Ignorieren, wenn der Ordnername offensichtlich irrelevant ist."
        )
    parts.append("\n```markdown")
    parts.append(source_md.strip())
    parts.append("```")
    result = "\n".join(parts)

    if len(result) > 200_000:
        log("warn",
            f"legacy prompt for {source_id} is {len(result):,} chars "
            f">200 KB — LLM may truncate or degrade quality",
            stderr=True)

    return result


def _truncate_existing_body(body: str, max_chars: int = 25000) -> str:
    """Truncate existing wiki body to keep the prompt manageable.

    Keeps the beginning (structure/headings) and end (most recent additions),
    cutting the middle. This way the LLM sees the page's outline and its
    latest content without needing to regenerate megabytes of context.
    """
    if len(body) <= max_chars:
        return body
    half = (max_chars - 200) // 2  # leave room for ellipsis header
    return (
        body[:half]
        + f"\n\n... [({len(body) - max_chars:,} weitere Zeichen ausgelassen)] ...\n\n"
        + body[-(max_chars - half - 200):]
    )


def build_prompt_page_update(
    target_slug: str,
    target_title: str,
    target_body: str,
    source_id: str,
    source_md: str,
    original_name: str | None = None,
    inbox_path: list[str] | None = None,
    max_chars: int | None = 25000,
) -> str:
    """Build prompt for updating an existing wiki page with a new source.

    Args:
        max_chars: Maximum chars for existing body in prompt. Pass None to
                   include the full body (used in split mode).
    """
    parts: list[str] = []
    parts.append(f"## Ziel-Wiki-Seite: {target_slug}")
    parts.append(f"**Titel:** {target_title}")
    if inbox_path:
        parts.append(f"**Ordner-Pfad:** `/{'/'.join(inbox_path)}/`")
    parts.append("")
    parts.append("### Aktueller Inhalt der Seite (kann gekürzt sein):")
    parts.append("\n```markdown")
    if max_chars is not None:
        truncated = _truncate_existing_body(target_body, max_chars)
        parts.append(truncated.strip())
        if len(target_body) > max_chars:
            parts.insert(len(parts) - 1,
                f"[Inhalt gekürzt auf {max_chars:,} von {len(target_body):,} Zeichen]")
    else:
        parts.append(target_body.strip())
    parts.append("```\n")

    parts.append(f"\n## Neue Quelle: {source_id}\n")
    if original_name:
        parts.append(f"**Originaldatei:** {original_name}")
    parts.append("\n```markdown")
    parts.append(source_md.strip())
    parts.append("```")
    return "\n".join(parts)


def build_prompt_page_create(
    target_slug: str,
    source_id: str,
    source_md: str,
    original_name: str | None = None,
    inbox_path: list[str] | None = None,
) -> str:
    """Build prompt for creating a new wiki page from a source."""
    parts: list[str] = []
    parts.append(f"## Neue Wiki-Seite erstellen")
    parts.append(f"**Ziel-Slug:** {target_slug}")
    if inbox_path:
        parts.append(f"**Ordner-Pfad:** `/{'/'.join(inbox_path)}/`")
    parts.append("")
    parts.append(f"\n## Quelle: {source_id}\n")
    if original_name:
        parts.append(f"**Originaldatei:** {original_name}")
    parts.append("\n```markdown")
    parts.append(source_md.strip())
    parts.append("```")
    return "\n".join(parts)


def build_prompt_rollup(
    folder_slug: str,
    folder_path: list[str],
    child_pages: list[tuple[str, str, str]],
    source_summaries: list[tuple[str, str, str]],
) -> str:
    """Build prompt for rolling up child wiki pages into a parent page.

    child_pages: list of (slug, title, body_or_summary)
    source_summaries: list of (source_id, original_name, preview)
    """
    parts: list[str] = []
    parts.append(f"## Rollup für Ordner: {folder_slug}")
    if folder_path:
        parts.append(f"**Ordner-Pfad:** `/{'/'.join(folder_path)}/`")
    parts.append("")

    if child_pages:
        parts.append("### Wiki-Seiten der direkten Unterordner:\n")
        for slug, title, body in child_pages:
            parts.append(f"#### {slug} — {title}")
            parts.append("\n```markdown")
            parts.append(body.strip())
            parts.append("```\n")
    else:
        parts.append("### Wiki-Seiten der direkten Unterordner:\n(nicht vorhanden)\n")

    if source_summaries:
        parts.append("### Quellen direkt in diesem Ordner:\n")
        for sid, name, preview in source_summaries:
            parts.append(f"- **{sid}** ({name}): {preview[:300]}")
        parts.append("")
    else:
        parts.append("### Quellen direkt in diesem Ordner:\n(nicht vorhanden)\n")

    return "\n".join(parts)


# JSON Schemas (structured outputs)

_SCHEMA_DISTILL_PAGE = {
    'type': 'json_schema',
    'json_schema': {
        'name': 'wiki_update',
        'strict': False,
        'schema': {
            'type': 'object',
            'properties': {
                'type': {'type': 'string', 'enum': ['create', 'update']},
                'slug': {'type': 'string'},
                'title': {'type': 'string'},
                'summary': {'type': 'string'},
                'topics': {'type': 'array', 'items': {'type': 'string'}},
                'content_md': {'type': 'string'},
            },
            'required': ['type', 'slug', 'title', 'summary', 'topics', 'content_md'],
            'additionalProperties': False,
            '$schema': 'http://json-schema.org/draft-07/schema#',
        },
    },
}

_SCHEMA_LEGACY = {
    'type': 'json_schema',
    'json_schema': {
        'name': 'wiki_chunk_legacy',
        'strict': False,
        'schema': {
            'type': 'object',
            'properties': {
                'actions': {
                    'type': 'array',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'type': {'type': 'string', 'enum': ['create', 'update']},
                            'slug': {'type': 'string'},
                            'title': {'type': 'string'},
                            'summary': {'type': 'string'},
                            'topics': {'type': 'array', 'items': {'type': 'string'}},
                            'content_md': {'type': 'string'},
                        },
                        'required': ['type', 'slug', 'title', 'summary', 'topics', 'content_md'],
                        'additionalProperties': False,
                        '$schema': 'http://json-schema.org/draft-07/schema#',
                    },
                },
            },
            'required': ['actions'],
            'additionalProperties': False,
            '$schema': 'http://json-schema.org/draft-07/schema#',
        },
    },
}

_SCHEMA_ROLLUP = {
    'type': 'json_schema',
    'json_schema': {
        'name': 'wiki_rollup',
        'strict': False,
        'schema': {
            'type': 'object',
            'properties': {
                'slug': {'type': 'string'},
                'title': {'type': 'string'},
                'summary': {'type': 'string'},
                'topics': {'type': 'array', 'items': {'type': 'string'}},
                'content_md': {'type': 'string'},
            },
            'required': ['slug', 'title', 'summary', 'topics', 'content_md'],
            'additionalProperties': False,
            '$schema': 'http://json-schema.org/draft-07/schema#',
        },
    },
}

# ── LLM Client ──────────────────────────────────────────────────────────────

class SourceSkipped(Exception):
    """Raised when a source is intentionally skipped (e.g., too large even after truncation).
    Catchable by callers that want to continue processing other sources."""
    pass


def call_llm(
    system: str,
    user: str,
    *,
    source_id: str | None = None,
    json_schema_override: dict | None = None,
) -> dict[str, Any]:
    """Call the LLM API and parse the JSON response.

    FIX #3: Full error handling with retry + exponential backoff.
    FIX #4: Robust JSON parsing with regex fallback for invalid responses.
    FIX #10: source_id for better error messages.
    """
    max_retries = 3
    last_error: Exception | None = None
    source_label = source_id if source_id else "unknown"

    for attempt in range(max_retries):
        try:
            payload = {
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": llm_temperature(_CONFIG),
                "max_tokens": 65536,
                "response_format": json_schema_override or _SCHEMA_DISTILL_PAGE,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            req = urllib.request.Request(
                f"{LLM_BASE_URL}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
                raw = resp.read().decode("utf-8")
                body = json.loads(raw)
            text = body["choices"][0]["message"]["content"]
            if text is None:
                log("warn",
                    f"LLM returned content=null (reasoning bug) for {source_label} "
                    f"(attempt {attempt + 1}/{max_retries}) — will retry next run",
                    stderr=True)
                raise ValueError("content is null (reasoning bug)")

            # FIX #4: Robust JSON extraction (non-greedy, balanced braces)
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                blocks = []
                depth = 0
                start = None
                for i, c in enumerate(text):
                    if c == '{':
                        if depth == 0:
                            start = i
                        depth += 1
                    elif c == '}':
                        depth -= 1
                        if depth == 0 and start is not None:
                            blocks.append(text[start:i + 1])
                for block in sorted(blocks, key=len, reverse=True):
                    try:
                        return json.loads(block)
                    except json.JSONDecodeError:
                        continue
                log("warn",
                    f"LLM returned invalid JSON for {source_label} "
                    f"(attempt {attempt + 1}/{max_retries}):\n"
                    f"  first 200 chars: {text[:200]!r}",
                    stderr=True)
                raise  # trigger retry

        except urllib.error.HTTPError as e:
            last_error = e
            status = e.code if hasattr(e, 'code') else 'unknown'
            log("retry",
                f"LLM HTTP error {status} for {source_label} "
                f"(attempt {attempt + 1}/{max_retries})",
                stderr=True)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_error = e
            log("retry",
                f"LLM connection error for {source_label}: {e} "
                f"(attempt {attempt + 1}/{max_retries})",
                stderr=True)
        except json.JSONDecodeError as e:
            last_error = e
            log("retry",
                f"LLM response not valid JSON for {source_label}: {e} "
                f"(attempt {attempt + 1}/{max_retries})",
                stderr=True)
        except ValueError as e:
            last_error = e
            log("warn",
                f"Skipping {source_label}: {e} — will retry next nightly run",
                stderr=True)
            break  # exit retry loop, source NOT marked processed
        except Exception as e:
            raise RuntimeError(f"Unexpected LLM error: {e}") from e

        if attempt < max_retries - 1:
            backoff = (attempt + 1) * 5  # 5s, 10s
            log("retry", f"waiting {backoff}s before retry…", stderr=True)
            time.sleep(backoff)

    raise RuntimeError(
        f"LLM call failed after {max_retries} attempts for {source_label}: {last_error}"
    ) from last_error


# ── Phase 1: Distill ────────────────────────────────────────────────────────

def source_already_distilled(source_id: str, pages: dict[str, dict]) -> bool:
    """Check if a source_id is already referenced by wiki pages.

    O(1) fast path: checks ``distilled`` field in the source's metadata.json.
    Falls back to scanning all wiki pages' frontmatter for backward
    compatibility with sources that haven't been migrated yet.
    """
    # Determine scope from pages dict (pages are scope-isolated)
    # We need scope to read metadata — derive from first page's scope field
    scope: str | None = None
    for p in pages.values():
        s = p["frontmatter"].get("scope")
        if s:
            scope = s
            break

    # Fast path: check metadata.json
    if scope is not None:
        meta = get_source_metadata(source_id, scope)
        if "distilled" in meta:
            return meta["distilled"]

    # Slow path: scan all wiki pages' frontmatter (backward compat)
    return any(
        source_id in (p["frontmatter"].get("sources") or [])
        for p in pages.values()
    )


def distill_source(
    scope: str,
    source_id: str,
    pages: dict[str, dict] | None = None,
) -> int:
    """Distill a single source into the wiki.

    Uses hierarchical mode if the source has an inbox_path,
    otherwise falls back to legacy mode (all pages in prompt).

    Large sources are automatically chunked (see distill_source_chunked).

    Returns the number of actions applied.
    """
    src_dir = SOURCES / scope / source_id
    md_path = src_dir / "document.md"
    if not md_path.exists():
        log("skip", f"{scope}/{source_id}: no document.md", stderr=True)
        return 0

    md = md_path.read_text(encoding="utf-8")
    meta = get_source_metadata(source_id, scope)
    inbox_path = meta.get("inbox_path") or []
    original_name = meta.get("original_name") or None

    # Determine target folder slug (hierarchical mode)
    folder_slug = inbox_path_to_slug(inbox_path) if inbox_path else None

    # FIX #5: Use provided pages or load once
    if pages is None:
        pages = load_wiki_pages(scope)
        owned = True
    else:
        owned = False

    # ── Companion-Aware Target Selection ──
    # If companion pages exist (slug-2, slug-3…), use the smallest one
    # to avoid inflating an already-large primary page past LLM limits.
    if folder_slug:
        folder_slug = _pick_target_companion(folder_slug, pages)

    # Check if already distilled
    if source_already_distilled(source_id, pages):
        log("skip", f"{scope}/{source_id}: already referenced by wiki pages")
        return 0

    original_size = len(md)
    folder_hint = f" folder=/{'/'.join(inbox_path)}/" if inbox_path else ""
    chunk_note = f", chunked" if original_size > CHUNK_TRIGGER else ""
    log("info",
        f"distilling {scope}/{source_id} "
        f"({original_size:,} chars, model={MODEL}{folder_hint}{chunk_note})")

    # ── Hierarchical mode (target page known) ──
    if folder_slug:
        target_page = load_page(scope, folder_slug)
        if target_page is None and folder_slug in pages:
            target_page = pages[folder_slug]

        # ── Auto-Split Detection ──
        # If the existing page body exceeds WIKI_MAX_PAGE_CHARS and the source
        # is small enough for a single LLM call, trigger split mode.
        # In split mode, the LLM creates a companion page instead of inflating
        # the existing one further.
        do_split = (
            WIKI_SPLIT_ENABLED
            and target_page is not None
            and _page_needs_split(target_page["body"])
            and original_size <= CHUNK_TRIGGER  # only for non-chunked sources
        )

        if do_split:
            split_slug = _next_split_slug(folder_slug, scope, pages)
            log("split",
                f"page {scope}/{folder_slug} has {len(target_page['body']):,} chars "
                f"(≥ {WIKI_MAX_PAGE_CHARS:,}) → mechanical split to {split_slug}")

            # ── Mechanical Split (replaces broken LLM-based split) ──
            # The old approach fed the entire existing page (up to 40K) + new source
            # to the LLM and asked it to compress+split. The LLM produced output of
            # ~the same size, so the split never actually reduced page size → infinite
            # retry loop. Instead: distill the new source into a companion page and
            # cross-link, same pattern the chunked pipeline uses.

            # 1. Distill new source into the companion page (CREATE, no existing body)
            prompt = build_prompt_page_create(
                target_slug=split_slug,
                source_id=source_id,
                source_md=md,
                original_name=original_name,
                inbox_path=inbox_path,
            )
            result = call_llm(SYSTEM_PROMPT_DISTILL_PAGE, prompt, source_id=source_id)
            companion_body = (result.get("content_md") or "").rstrip()

            if not companion_body:
                log("warn",
                    f"mechanical split for {scope}/{folder_slug}: LLM returned "
                    f"empty content_md — writing placeholder companion (source "
                    f"NOT registered to avoid reprocessing)",
                    stderr=True)
                # FIX: Don't register source_id — mark as permanently unprocessable
                # via placeholder text. Source stays unprocessed but won't cause
                # infinite retry since the companion page now exists.
                today = date.today().isoformat()
                fallback_body = (
                    f"\n\n## Hinweis\n\n"
                    f"Quelle `{source_id}` konnte nicht verarbeitet werden "
                    f"(LLM lieferte keinen Inhalt). Diese Quelle kann nicht "
                    f"automatisch distilliert werden.\n"
                )
                fallback_body = _add_related_links(fallback_body, [folder_slug])
                write_page(
                    scope, split_slug,
                    title=f"{(target_page['frontmatter'].get('title') or folder_slug)} (Teil 2)",
                    summary="(Verarbeitung fehlgeschlagen — Platzhalter)",
                    topics=[],
                    sources=[], created=today, updated=today,
                    body=fallback_body,
                    folder_path="/".join(inbox_path) if inbox_path else None,
                    related_slugs=[folder_slug],
                )
                # Update primary with cross-link
                primary_fm = target_page["frontmatter"]
                primary_body = _add_related_links(target_page["body"], [split_slug])
                existing_related = list(primary_fm.get("related_slugs") or [])
                if split_slug not in existing_related:
                    existing_related.append(split_slug)
                write_page(
                    scope, folder_slug,
                    title=primary_fm.get("title", folder_slug),
                    summary=" ".join((primary_fm.get("summary") or "").split()),
                    topics=primary_fm.get("topics") or [],
                    sources=primary_fm.get("sources") or [],
                    created=primary_fm.get("created") or today,
                    updated=today,
                    body=primary_body,
                    folder_path=primary_fm.get("folder_path"),
                    related_slugs=existing_related if existing_related else None,
                )
                for s in (folder_slug, split_slug):
                    updated = load_page(scope, s)
                    if updated:
                        pages[s] = updated
                # Record distilled metadata (fallback case)
                write_distilled_metadata(source_id, scope, split_slug)
                return 2  # counted as processed (fallback companion + primary link)

        if do_split:
            # companion_body is guaranteed non-empty here
            today = date.today().isoformat()
            companion_title = (result.get("title") or "").strip()
            companion_summary = " ".join((result.get("summary") or "").split())
            companion_topics = [
                str(t).strip() for t in (result.get("topics") or [])
                if str(t).strip()
            ]

            # Extract part number for companion title suffix
            m_suffix = re.search(r"-(\d+)$", split_slug)
            part_num = int(m_suffix.group(1)) if m_suffix else 2
            if not companion_title:
                companion_title = f"{(target_page['frontmatter'].get('title') or folder_slug)} (Teil {part_num})"
            elif not companion_title.endswith(f"(Teil {part_num})"):
                companion_title = f"{companion_title} (Teil {part_num})"

            # 2. Write companion page with cross-link to primary
            companion_body = _add_related_links(companion_body, [folder_slug])
            write_page(
                scope, split_slug,
                title=companion_title, summary=companion_summary,
                topics=companion_topics,
                sources=[source_id], created=today, updated=today,
                body=companion_body,
                folder_path="/".join(inbox_path) if inbox_path else None,
                related_slugs=[folder_slug],
            )
            log("ok", f"create wiki/{scope}/{split_slug}.md (mechanical split companion, {len(companion_body):,} chars)")

            # 3. Update primary page: add cross-link to companion in body + frontmatter
            # FIX #1: Source belongs ONLY on the companion page — not on the primary.
            # The primary page is already oversized; adding the source_id there
            # caused double-registration (source appeared in both pages' sources).
            primary_fm = target_page["frontmatter"]
            primary_body = _add_related_links(target_page["body"], [split_slug])

            # Merge related_slugs into frontmatter
            existing_related = list(primary_fm.get("related_slugs") or [])
            if split_slug not in existing_related:
                existing_related.append(split_slug)

            write_page(
                scope, folder_slug,
                title=primary_fm.get("title", folder_slug),
                summary=" ".join((primary_fm.get("summary") or "").split()),
                topics=primary_fm.get("topics") or [],
                sources=primary_fm.get("sources") or [],  # FIX #1: preserve existing sources, don't add source_id
                created=primary_fm.get("created") or today,
                updated=today,
                body=primary_body,
                folder_path=primary_fm.get("folder_path"),
                related_slugs=existing_related if existing_related else None,
            )
            log("ok", f"update wiki/{scope}/{folder_slug}.md (added cross-link to {split_slug})")

            # Refresh pages dict
            for s in (folder_slug, split_slug):
                updated = load_page(scope, s)
                if updated:
                    pages[s] = updated

            # Record distilled metadata
            write_distilled_metadata(source_id, scope, split_slug)
            return 2  # primary updated + companion created

        # ── Normal path (no split) ──
        # Use chunked distillation
        actions, final_body = distill_source_chunked(
            scope=scope,
            source_id=source_id,
            source_md=md,
            target_slug=folder_slug,
            existing_page=target_page,
            inbox_path=inbox_path,
            original_name=original_name,
        )

        # In chunked mode (>1 chunk), the page was already written incrementally.
        # We still need to register the source in the pages dict and apply the
        # final result through the normal action pipeline.
        is_chunked = original_size > CHUNK_TRIGGER

        # ── Apply final action ──
        today = date.today().isoformat()
        applied = 0

        for a in actions:
            atype = a.get("type")
            slug = (a.get("slug") or "").strip()
            title = (a.get("title") or "").strip()
            summary = " ".join((a.get("summary") or "").split())
            topics = [str(t).strip() for t in (a.get("topics") or []) if str(t).strip()]
            # In chunked mode, use the accumulated body
            body = final_body if is_chunked else (a.get("content_md") or "").rstrip()

            original_slug = slug
            slug = safe_slug(slug)
            if slug != original_slug:
                log("warn", f"slug truncated to 120 chars: {original_slug!r} → {slug!r}", stderr=True)
            if not body:
                log("warn", f"empty content_md for {slug}, skipping", stderr=True)
                continue

            if atype == "create":
                if slug in pages:
                    log("warn", f"create {scope}/{slug}: page exists, skipping", stderr=True)
                    continue
                write_page(
                    scope, slug,
                    title=title, summary=summary, topics=topics,
                    sources=[source_id], created=today, updated=today, body=body,
                    folder_path="/".join(inbox_path) if inbox_path else None,
                )
                log("ok", f"create wiki/{scope}/{slug}.md")
            elif atype == "chunked_complete":
                # Page was already written to disk by distill_source_chunked()
                # after each chunk. Just register the source_id and refresh
                # the in-memory pages dict.
                # With chunked-split, there may be companion pages (-2, -3, …).
                # Register the source_id on ALL of them.
                primary_slug = folder_slug or "unknown"
                # Discover all split pages: primary + any -N companions
                split_pattern = re.compile(rf"^{re.escape(primary_slug)}(-\d+)?$")
                all_slugs: list[str] = [primary_slug]
                scope_dir = WIKI / scope
                if scope_dir.exists():
                    for p in scope_dir.glob("*.md"):
                        if split_pattern.match(p.stem) and p.stem != primary_slug:
                            all_slugs.append(p.stem)
                all_slugs.sort()  # deterministic order

                for slug in all_slugs:
                    # BUGFIX 4: Skip if source_id already registered
                    existing_page = pages.get(slug)
                    if existing_page is not None:
                        existing_sources = existing_page["frontmatter"].get("sources") or []
                        if source_id in existing_sources:
                            log("info", f"chunked_complete {scope}/{slug}: source already registered, skipping rewrite")
                            continue

                    updated_page = load_page(scope, slug)
                    if updated_page is None:
                        log("warn",
                            f"chunked_complete {scope}/{slug}: page not found on disk!",
                            stderr=True)
                        continue
                    fm = updated_page["frontmatter"]
                    old_sources = list(fm.get("sources") or [])
                    if source_id not in old_sources:
                        old_sources.append(source_id)
                    write_page(
                        scope, slug,
                        title=(fm.get("title") or slug).strip(),
                        summary=" ".join((fm.get("summary") or "").split()),
                        topics=fm.get("topics") or [],
                        sources=old_sources,
                        created=fm.get("created") or today,
                        updated=today,
                        body=updated_page["body"],
                        folder_path=fm.get("folder_path"),
                        related_slugs=fm.get("related_slugs"),
                    )
                    pages[slug] = load_page(scope, slug)

                log("ok", f"chunked_complete wiki/{scope}/{primary_slug}.md "
                    f"({len(all_slugs)} page(s), {len(final_body):,} chars last, "
                    f"source registered on all)")
            elif atype == "update":
                if slug not in pages:
                    log("warn", f"update {scope}/{slug}: page missing, skipping", stderr=True)
                    continue
                fm = pages[slug]["frontmatter"]
                old_sources = list(fm.get("sources") or [])
                if source_id not in old_sources:
                    old_sources.append(source_id)
                existing_folder_path = fm.get("folder_path")
                write_page(
                    scope, slug,
                    title=title, summary=summary, topics=topics,
                    sources=old_sources,
                    created=fm.get("created") or today, updated=today, body=body,
                    folder_path=existing_folder_path,
                )
                log("ok", f"update wiki/{scope}/{slug}.md")
            else:
                log("warn", f"unknown action type {atype!r}, skipping", stderr=True)
                continue

            updated_page = load_page(scope, slug)
            if updated_page:
                pages[slug] = updated_page
            applied += 1

        # Record distilled metadata on success
        if applied > 0:
            target_slug = None
            for slug, p in pages.items():
                if source_id in (p["frontmatter"].get("sources") or []):
                    target_slug = slug
                    break
            write_distilled_metadata(source_id, scope, target_slug)
        return applied

    else:
        # ── Legacy mode (all pages in prompt) ──
        # For large sources in legacy mode, we still chunk, but we need a
        # slightly different approach since legacy mode decides CREATE vs UPDATE.
        # We use the single-call path for simplicity (LLM handles truncation).
        prompt = build_prompt_legacy(
            source_id, md, pages,
            inbox_path=inbox_path, original_name=original_name,
        )
        result = call_llm(
            SYSTEM_PROMPT_DISTILL_LEGACY, prompt,
            source_id=source_id,
            json_schema_override=_SCHEMA_LEGACY,
        )
        actions = result.get("actions") or []

        # ── Apply actions ──
        today = date.today().isoformat()
        applied = 0

        for a in actions:
            atype = a.get("type")
            slug = (a.get("slug") or "").strip()
            title = (a.get("title") or "").strip()
            summary = " ".join((a.get("summary") or "").split())
            topics = [str(t).strip() for t in (a.get("topics") or []) if str(t).strip()]
            body = (a.get("content_md") or "").rstrip()

            original_slug = slug
            slug = safe_slug(slug)
            if slug != original_slug:
                log("warn", f"slug truncated to 120 chars: {original_slug!r} → {slug!r}", stderr=True)
            if not body:
                log("warn", f"empty content_md for {slug}, skipping", stderr=True)
                continue

            if atype == "create":
                if slug in pages:
                    log("warn", f"create {scope}/{slug}: page exists, skipping", stderr=True)
                    continue
                write_page(
                    scope, slug,
                    title=title, summary=summary, topics=topics,
                    sources=[source_id], created=today, updated=today, body=body,
                    folder_path="/".join(inbox_path) if inbox_path else None,
                )
                log("ok", f"create wiki/{scope}/{slug}.md")
            elif atype == "update":
                if slug not in pages:
                    log("warn", f"update {scope}/{slug}: page missing, skipping", stderr=True)
                    continue
                fm = pages[slug]["frontmatter"]
                old_sources = list(fm.get("sources") or [])
                if source_id not in old_sources:
                    old_sources.append(source_id)
                existing_folder_path = fm.get("folder_path")
                write_page(
                    scope, slug,
                    title=title, summary=summary, topics=topics,
                    sources=old_sources,
                    created=fm.get("created") or today, updated=today, body=body,
                    folder_path=existing_folder_path,
                )
                log("ok", f"update wiki/{scope}/{slug}.md")
            else:
                log("warn", f"unknown action type {atype!r}, skipping", stderr=True)
                continue

            updated_page = load_page(scope, slug)
            if updated_page:
                pages[slug] = updated_page
            applied += 1

        # Record distilled metadata on success
        if applied > 0:
            target_slug = None
            for slug, p in pages.items():
                if source_id in (p["frontmatter"].get("sources") or []):
                    target_slug = slug
                    break
            write_distilled_metadata(source_id, scope, target_slug)
        return applied


# ── Phase 2: Rollup ─────────────────────────────────────────────────────────

SCRIPTS_DIR = Path(__file__).resolve().parent
ROLLUP_HASH_FILE = SCRIPTS_DIR / "rollup-hashes.json"  # keyed by folder_path


def _load_rollup_hashes() -> dict[str, str]:
    """Load persisted rollup hashes (folder_path -> hash)."""
    if not ROLLUP_HASH_FILE.exists():
        return {}
    try:
        return json.loads(ROLLUP_HASH_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_rollup_hashes(hashes: dict[str, str]) -> None:
    """Persist rollup hashes to JSON (atomic write)."""
    temp_path = ROLLUP_HASH_FILE.with_suffix(".tmp")
    temp_path.write_text(
        json.dumps(hashes, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temp_path), str(ROLLUP_HASH_FILE))


def rollup_input_hash(
    child_pages: list[tuple[str, str, str]],
    source_summaries: list[tuple[str, str, str]],
) -> str:
    """Compute a SHA-256 hash of rollup inputs (child pages + source previews).

    Used to detect whether rollup inputs actually changed since last run.
    """
    h = hashlib.sha256()
    for slug, title, content in child_pages:
        h.update(f"page:{slug}:{title}:{content}".encode("utf-8"))
        h.update(b"\x00")  # separator
    for sid, orig, preview in source_summaries:
        h.update(f"src:{sid}:{orig}:{preview}".encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def rollup_folder(
    folder_slug: str,
    folder_path: list[str],
    scope: str,
    source_metadata: dict[str, dict[str, Any]] | None = None,
) -> bool:
    """Roll up a single folder: child wiki pages + direct sources → parent page.

    FIX #5: Uses frontmatter summary for child pages (not truncated body).
    FIX #8: Accepts pre-loaded source_metadata dict to avoid O(N) file scans.
    FIX #9: Skips LLM call if rollup inputs haven't changed (hash comparison).

    Returns True if a page was written or already up-to-date, False if skipped (nothing to roll up).
    """
    if folder_path:
        inbox_folder = INBOX / scope
        for part in folder_path:
            inbox_folder = inbox_folder / part
    else:
        inbox_folder = INBOX / scope
    child_pages: list[tuple[str, str, str]] = []

    if inbox_folder.exists() and inbox_folder.is_dir():
        for child_dir in sorted(inbox_folder.iterdir()):
            if not child_dir.is_dir() or child_dir.name.startswith("."):
                continue
            child_rel = list(folder_path) + [child_dir.name]
            child_slug = inbox_path_to_slug(child_rel)
            child_page = load_page(scope, child_slug)
            if child_page:
                child_fm = child_page["frontmatter"]
                child_content = child_fm.get("summary") or child_page["body"][:8000]
                child_pages.append((
                    child_slug,
                    child_fm.get("title", child_slug),
                    child_content,
                ))

    source_summaries: list[tuple[str, str, str]] = []
    if source_metadata is not None:
        for sid, meta in source_metadata.items():
            if (meta.get("inbox_path") or []) == folder_path:
                orig = meta.get("original_name", sid)
                md_file = SOURCES / scope / sid / "document.md"
                preview = ""
                if md_file.exists():
                    preview = md_file.read_text(encoding="utf-8")[:500]
                source_summaries.append((sid, orig, preview))
    else:
        sources_dir = SOURCES / scope
        if sources_dir.exists():
            for meta_path in sorted(sources_dir.glob("*/metadata.json")):
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    continue
                if (meta.get("inbox_path") or []) == folder_path:
                    sid = meta.get("source_id", meta_path.parent.name)
                    orig = meta.get("original_name", sid)
                    md_file = meta_path.parent / "document.md"
                    preview = ""
                    if md_file.exists():
                        preview = md_file.read_text(encoding="utf-8")[:500]
                    source_summaries.append((sid, orig, preview))

    if not child_pages and not source_summaries:
        return False

    # FIX #9 (enhanced): Skip if rollup inputs haven't changed.
    # FIX #10: Store hash by folder_path in rollup-hashes.json (not in wiki page)
    # so slug drift from LLM doesn't cause phantom updates.
    current_hash = rollup_input_hash(child_pages, source_summaries)
    stored_hashes = _load_rollup_hashes()
    folder_key = "/".join(folder_path) if folder_path else scope
    if stored_hashes.get(folder_key) == current_hash:
        log("rollup-skip", f"{scope}/{folder_slug} inputs unchanged, skipping")
        return True  # Page is up-to-date, count as processed

    prompt = build_prompt_rollup(
        folder_slug=folder_slug,
        folder_path=folder_path,
        child_pages=child_pages,
        source_summaries=source_summaries,
    )

    child_count = len(child_pages)
    source_count = len(source_summaries)
    log("rollup",
        f"{scope}/{folder_slug} "
        f"({child_count} child page(s), {source_count} direct source(s))")

    result = call_llm(SYSTEM_PROMPT_ROLLUP, prompt, source_id=f"rollup:{folder_slug}", json_schema_override=_SCHEMA_ROLLUP)

    # FIX #10: Always use folder_slug as canonical slug — prevents LLM drift
    # from creating zombie pages with different names each run.
    slug = safe_slug(folder_slug)
    title = (result.get("title") or folder_slug).strip()
    summary = " ".join((result.get("summary") or "").split())
    topics = [str(t).strip() for t in (result.get("topics") or []) if str(t).strip()]
    body = (result.get("content_md") or "").rstrip()

    if not body:
        log("warn", f"rollup empty content_md for {slug}, skipping", stderr=True)
        return False

    all_sources: list[str] = []
    seen_sources: set[str] = set()
    for child_slug, _, _ in child_pages:
        child_page = load_page(scope, child_slug)
        if child_page:
            for s in (child_page["frontmatter"].get("sources") or []):
                if s not in seen_sources:
                    all_sources.append(s)
                    seen_sources.add(s)
    for sid, _, _ in source_summaries:
        if sid not in seen_sources:
            all_sources.append(sid)
            seen_sources.add(sid)

    today = date.today().isoformat()

    existing = load_page(scope, slug)
    if existing:
        old_fm = existing["frontmatter"]
        old_sources = list(old_fm.get("sources") or [])
        merged_sources = list(dict.fromkeys(old_sources + all_sources))
        write_page(
            scope, slug,
            title=title, summary=summary, topics=topics,
            sources=merged_sources,
            created=old_fm.get("created") or today, updated=today, body=body,
            folder_path="/".join(folder_path) if folder_path else None,
        )
        log("rollup-ok", f"update wiki/{scope}/{slug}.md")
    else:
        write_page(
            scope, slug,
            title=title, summary=summary, topics=topics,
            sources=all_sources, created=today, updated=today, body=body,
            folder_path="/".join(folder_path) if folder_path else None,
        )
        log("rollup-ok", f"create wiki/{scope}/{slug}.md")

    # FIX #10: Persist hash by folder_path for next-run change detection
    stored_hashes[folder_key] = current_hash
    _save_rollup_hashes(stored_hashes)

    return True


def rollup_all(scope: str) -> int:
    """Perform bottom-up rollup of the entire inbox folder hierarchy for a scope.

    Walks the inbox/<scope>/ tree, collects all folders sorted by depth
    (deepest first), then rolls each one up.

    Returns the number of folders processed.
    """
    inbox_scope = INBOX / scope
    if not inbox_scope.exists():
        log("rollup", f"no inbox/{scope}, skipping")
        return 0

    folders: list[tuple[int, list[str]]] = []
    for dirpath, dirnames, filenames in os.walk(inbox_scope):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        rel = Path(dirpath).relative_to(inbox_scope)
        parts = list(rel.parts)
        if parts:
            folders.append((len(parts), parts))

    folders.sort(key=lambda x: x[0], reverse=True)

    source_metadata = load_all_source_metadata(scope)

    processed = 0
    for depth, parts in folders:
        if past_deadline():
            remaining = len(folders) - processed
            log("rollup-stop", f"deadline reached, skipped {remaining} remaining folder(s)")
            break
        folder_slug = inbox_path_to_slug(parts)
        try:
            if rollup_folder(folder_slug, parts, scope, source_metadata=source_metadata):
                processed += 1
        except Exception as e:
            log("rollup-fail", f"{scope}/{folder_slug}: {e}", stderr=True)
            processed += 1

    log("rollup-done", f"{scope}: {processed}/{len(folders)} folder(s) processed")
    return processed


def rollup_ancestors(folder_path: list[str], scope: str,
                     source_metadata: dict[str, dict[str, Any]] | None = None) -> int:
    """Roll up only the ancestor folders of a given folder (selective rollup).

    FIX #1: Works directly with inbox_path list — no slug-string splitting.
    FIX #2: Accepts pre-loaded source_metadata to avoid repeated full scans.

    Used after distilling a single source to update the hierarchy chain.
    Returns the number of ancestor folders processed.
    """
    if len(folder_path) <= 1:
        return 0

    processed = 0
    for i in range(len(folder_path) - 1, 0, -1):
        ancestor_path = folder_path[:i]
        ancestor_slug = inbox_path_to_slug(ancestor_path)
        if past_deadline():
            log("rollup-stop", "deadline reached during ancestor rollup")
            break
        try:
            if rollup_folder(ancestor_slug, ancestor_path, scope,
                             source_metadata=source_metadata):
                processed += 1
        except Exception as e:
            log("rollup-fail", f"{scope}/{ancestor_slug}: {e}", stderr=True)
            processed += 1

    return processed


# ── Index Rebuild ───────────────────────────────────────────────────────────

def rebuild_wiki_index(scope: str) -> None:
    """Rebuild .index.json for a scope's wiki pages."""
    scope_dir = WIKI / scope
    scope_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for slug, p in sorted(load_wiki_pages(scope).items()):
        fm = p["frontmatter"]
        entry: dict[str, Any] = {
            "slug": slug,
            "scope": scope,
            "title": fm.get("title", ""),
            "summary": fm.get("summary", ""),
            "topics": fm.get("topics") or [],
            "sources": fm.get("sources") or [],
            "created": fm.get("created", ""),
            "updated": fm.get("updated", ""),
        }
        if fm.get("folder_path"):
            entry["folder_path"] = fm["folder_path"]
        index.append(entry)
    (scope_dir / ".index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def rebuild_sources_index(scope: str) -> None:
    """Rebuild .index.json for a scope's sources."""
    scope_dir = SOURCES / scope
    scope_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for meta_path in sorted(scope_dir.glob("*/metadata.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        md_path = meta_path.parent / "document.md"
        try:
            preview = md_path.read_text(encoding="utf-8")[:500] if md_path.exists() else ""
        except OSError:
            preview = ""
        entries.append(
            {
                "source_id": meta.get("source_id", meta_path.parent.name),
                "scope": scope,
                "original_name": meta.get("original_name", ""),
                "ingested_at": meta.get("ingested_at", ""),
                "size_bytes": meta.get("size_bytes", 0),
                "preview": preview,
            }
        )
    (scope_dir / ".index.json").write_text(
        json.dumps(entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


# ── Source Discovery ────────────────────────────────────────────────────────

def unprocessed_sources(scope: str) -> list[str]:
    """Find source IDs not yet referenced by any wiki page."""
    pages = load_wiki_pages(scope)
    scope_dir = SOURCES / scope
    if not scope_dir.exists():
        return []
    all_ids = sorted(p.name for p in scope_dir.iterdir() if p.is_dir())
    return [sid for sid in all_ids if not source_already_distilled(sid, pages)]


def locate_source_scope(source_id: str) -> str | None:
    """Find which scope contains a given source_id."""
    for scope in _SCOPES:
        if (SOURCES / scope / source_id).is_dir():
            return scope
    return None


# ── Deadline Check ──────────────────────────────────────────────────────────

def past_deadline() -> bool:
    """Check if we've passed the WIKIS_DEADLINE environment variable.

    WIKIS_DEADLINE is in LOCAL time (HH:MM), set by run_inbox.sh.
    Matches the shell's past_deadline() which uses `date +%H:%M` (local).
    """
    deadline = os.environ.get("WIKIS_DEADLINE")
    if not deadline:
        return False
    try:
        hh, mm = deadline.split(":")
        now = datetime.datetime.now()
        dh, dm = int(hh), int(mm)
        if (now.hour, now.minute) >= (dh, dm):
            return True
        return False
    except (ValueError, AttributeError):
        return False


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Distill Docling sources into hierarchical wiki pages."
    )
    ap.add_argument("source_id", nargs="?", help="Source ID to distill")
    ap.add_argument("--scope", choices=_SCOPES, help="Limit to scope")
    ap.add_argument(
        "--all", action="store_true",
        help="Distill all unprocessed sources, then roll up hierarchy",
    )
    ap.add_argument(
        "--rollup", action="store_true",
        help="Roll up the entire wiki hierarchy (phase 2 only)",
    )
    ap.add_argument(
        "--rebuild-indices", action="store_true",
        help="Rebuild indices only",
    )
    args = ap.parse_args()

    target_scopes = (args.scope,) if args.scope else _SCOPES

    for scope in target_scopes:
        (WIKI / scope).mkdir(parents=True, exist_ok=True)

    # ── Rebuild indices only ──
    if args.rebuild_indices:
        for scope in target_scopes:
            rebuild_wiki_index(scope)
            rebuild_sources_index(scope)
        log("ok", "indices rebuilt")
        return 0

    # Always rebuild sources index first
    for scope in target_scopes:
        rebuild_sources_index(scope)

    # ── Mode: --all (distill + full rollup) ──
    if args.all:
        # One-time migration: populate distilled field in metadata.json
        for scope in target_scopes:
            if _any_source_needs_migration(scope):
                _migrate_existing_metadata(scope)

        targets: list[tuple[str, str]] = []
        for scope in target_scopes:
            for sid in unprocessed_sources(scope):
                targets.append((scope, sid))

        if not targets:
            log("info", "no unprocessed sources")
        else:
            total = 0
            processed_count = 0
            scope_pages: dict[str, dict[str, dict]] = {
                s: load_wiki_pages(s) for s in target_scopes
            }
            for scope, sid in targets:
                if past_deadline():
                    remaining = len(targets) - processed_count
                    log("stop", f"deadline reached, skipped {remaining} remaining source(s)")
                    break
                try:
                    # distill_source() already calls write_distilled_metadata() internally
                    result = distill_source(scope, sid, pages=scope_pages[scope])
                    total += result
                    processed_count += 1
                except SourceSkipped as e:
                    log("skip", f"{scope}/{sid}: {e}", stderr=True)
                    processed_count += 1
                except Exception as e:
                    log("fail", f"{scope}/{sid}: {e}", stderr=True)
                    processed_count += 1
            log("distill-done",
                f"{total} action(s) applied across "
                f"{processed_count}/{len(targets)} source(s)")

        # Phase 2: full rollup of hierarchy
        for scope in target_scopes:
            rollup_all(scope)

    # ── Mode: --rollup only ──
    elif args.rollup:
        for scope in target_scopes:
            rollup_all(scope)

    # ── Mode: single source_id ──
    elif args.source_id:
        scope = args.scope or locate_source_scope(args.source_id)
        if scope is None:
            log("fail", f"source-id {args.source_id} not found in any scope", stderr=True)
            return 2
        if scope not in target_scopes:
            log("fail",
                f"source-id {args.source_id} found in {scope}, "
                f"but --scope limits to {target_scopes}",
                stderr=True)
            return 2

        total = distill_source(scope, args.source_id)

        meta = get_source_metadata(args.source_id, scope)
        inbox_path = meta.get("inbox_path") or []
        if inbox_path and len(inbox_path) > 1:
            metadata_cache = load_all_source_metadata(scope)
            rollup_ancestors(inbox_path, scope, source_metadata=metadata_cache)

        log("done", f"{total} action(s) applied for {scope}/{args.source_id}")

    else:
        ap.error("pass source_id, --all, --rollup, or --rebuild-indices")
        return 2

    # Rebuild wiki indices
    for scope in target_scopes:
        rebuild_wiki_index(scope)
        rebuild_sources_index(scope)

    return 0


if __name__ == "__main__":
    sys.exit(main())

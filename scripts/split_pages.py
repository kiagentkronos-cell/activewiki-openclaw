#!/usr/bin/env python3
"""ActiveWiki: thematic splitting of oversized wiki pages.

Scans all wiki pages (all scopes) for pages exceeding --max-chars
and splits them thematically into multiple smaller pages.

Uses the same LLM stack as distill.py (configured via activewiki.json).

Usage:
    python3 split_pages.py --dry-run          # What would happen?
    python3 split_pages.py                    # Go ahead
    python3 split_pages.py --max-chars 15000  # Lower threshold
    python3 split_pages.py --scope private    # Only one scope
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
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
    load_config,
    wikis_root, scopes,
    llm_model, llm_url,
)

_CONFIG = load_config()
_WIKIS_ROOT = wikis_root(_CONFIG)
_SCOPES = scopes(_CONFIG)

# ── Surrogate-escape safety ─────────────────────────────────────────────────
try:
    sys.stdout.reconfigure(errors="backslashreplace")
    sys.stderr.reconfigure(errors="backslashreplace")
except (AttributeError, ValueError):
    pass  # reconfigure not available (e.g., redirected stdout)

# ── Paths (derived from config) ─────────────────────────────────────────────
WIKI = _WIKIS_ROOT / "wiki"

# ── LLM Config (from activewiki.json, overridable via env) ──────────────────
MODEL = os.environ.get("ACTIVEWIKI_MODEL", llm_model(_CONFIG))
LLM_BASE_URL = os.environ.get("OLLAMA_URL", llm_url(_CONFIG)).rstrip("/")
HTTP_TIMEOUT = int(os.environ.get("ACTIVEWIKI_HTTP_TIMEOUT", "3600"))

# ── Regex ────────────────────────────────────────────────────────────────────
FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n?(.*)$", re.DOTALL)
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,118}[a-z0-9]$")


# ── Logging ──────────────────────────────────────────────────────────────────

def log(level: str, message: str, *, stderr: bool = False) -> None:
    ts = datetime.datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"[{ts}] [{level}] {message}"
    print(line, file=sys.stderr if stderr else sys.stdout)


# ── Slug Helpers ─────────────────────────────────────────────────────────────

def safe_slug(slug: str, max_len: int = 120) -> str:
    slug = slug.rstrip("-")
    if len(slug) <= max_len and SLUG_RE.match(slug):
        return slug
    if len(slug) <= max_len:
        return slug
    h = hashlib.sha256(slug.encode()).hexdigest()[:8]
    suffix = f"--{h}"
    prefix_len = max_len - len(suffix)
    prefix = slug[:prefix_len].rstrip("-")
    if len(prefix) + len(suffix) < 2:
        prefix = "a"
    return f"{prefix}{suffix}"


def normalize_slug_component(name: str) -> str:
    german_map = str.maketrans({
        "ö": "oe", "Ö": "Oe", "ü": "ue", "Ü": "Ue",
        "ä": "ae", "Ä": "Ae", "ß": "ss",
    })
    mapped = name.translate(german_map)
    nfd = unicodedata.normalize("NFD", mapped)
    ascii_str = nfd.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_str.lower()).strip("-")
    return slug or "untitled"


# ── Page I/O ─────────────────────────────────────────────────────────────────

def parse_page(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    m = FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    fm = yaml.safe_load(m.group(1)) or {}
    return fm, m.group(2)


def load_page(scope: str, slug: str) -> dict[str, Any] | None:
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
) -> None:
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
    fm_yaml = yaml.safe_dump(
        frontmatter, allow_unicode=True, sort_keys=False, default_flow_style=False
    )
    content = f"---\n{fm_yaml}---\n{body.rstrip()}\n"
    (WIKI / scope / f"{slug}.md").write_text(content, encoding="utf-8")


def archive_page(scope: str, slug: str) -> Path:
    """Move a page to _archive/split/YYYY-MM-DD/ directory."""
    src = WIKI / scope / f"{slug}.md"
    if not src.exists():
        return src
    archive_dir = WIKI / "_archive" / "split" / date.today().isoformat()
    archive_dir.mkdir(parents=True, exist_ok=True)
    dst = archive_dir / f"{slug}.md"
    src.rename(dst)
    return dst


def resolve_slug_collision(scope: str, slug: str) -> str:
    """Check if slug already exists in scope; if so, append -1, -2, … until free."""
    candidate = slug
    counter = 1
    while (WIKI / scope / f"{candidate}.md").exists():
        candidate = f"{slug}-{counter}"
        counter += 1
    return candidate


def update_backlinks(scope: str, old_slug: str, new_primary_slug: str) -> int:
    """Scan all other wiki pages in the same scope for links to the old page.
    
    Updates various link formats:
      - [title](old-slug) → [title](new-primary-slug)
      - [title](old-slug.md) → [title](new-primary-slug.md)
      - [[old-slug]] → [[new-primary-slug]]
      - [title][old-slug] → [title][new-primary-slug]
      
    Returns number of files modified.
    """
    scope_dir = WIKI / scope
    if not scope_dir.exists():
        return 0
    
    modified = 0
    # Patterns: (regex, replacement)
    patterns = [
        # Markdown links: [text](old-slug) but NOT [text](old-slug.something)
        (r'\[([^\]]*)\]\(' + re.escape(old_slug) + r'(?![a-zA-Z0-9_-])',
         rf'[\1]({new_primary_slug})'),
        # Markdown links with .md: [text](old-slug.md)
        (r'\[([^\]]*)\]\(' + re.escape(old_slug) + r'\.md\)',
         rf'[\1]({new_primary_slug}.md)'),
        # Obsidian wiki-links: [[old-slug]] or [[old-slug|text]]
        (r'\[\[' + re.escape(old_slug) + r'(?:\|[^\]]*)?\]\]',
         rf'[[{new_primary_slug}]]'),
        # Reference-style links: [text][old-slug]
        (r'\[([^\]]*)\]\[' + re.escape(old_slug) + r'\]',
         rf'[\1][{new_primary_slug}]'),
    ]
    
    for p in sorted(scope_dir.glob("*.md")):
        target_slug = p.stem
        if target_slug == old_slug or target_slug == new_primary_slug:
            continue
        
        try:
            content = p.read_text(encoding="utf-8")
        except Exception:
            continue
        
        new_content = content
        changed = False
        for pattern, replacement in patterns:
            if re.search(pattern, content):
                new_content = re.sub(pattern, replacement, new_content)
                changed = True
        
        if changed:
            p.write_text(new_content, encoding="utf-8")
            log("info", f"  updated backlinks in {scope}/{target_slug} → {new_primary_slug}")
            modified += 1
    
    return modified


# ── LLM Client ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT_SPLIT = """Du bist ein Redakteur für ein Wissens-Repo.

Deine Aufgabe: Eine zu große Wiki-Seite thematisch auf mehrere kleinere Seiten aufteilen.

Regeln:
1. Behalte ALLEN Inhalt bei — nichts weglassen.
2. Gruppiere verwandte Themen zusammen (z.B. alle Baurechts-Themen, alle technischen Unterlagen).
3. Erstelle pro Thema eine eigene Seite mit eigenem Slug, Titel, Summary und Body.
4. Jeder Slug: kebab-case (a-z, 0-9, Bindestrich), 3-60 Zeichen, beschreibend.
5. Füge 'Siehe auch:'-Links zu den anderen Teilen ein (am Ende jedes Body).
6. topics: 1-5 kurze Tags pro Seite in kebab-case.
7. Sprache: Deutsch, neutraler Sachton.
8. Struktur: Markdown mit Überschriften (##), Listen, Tabellen nach Bedarf.
9. Die Seiten sollten jeweils unter {max_chars} Zeichen Body-Content haben.
10. sources: aus der Original-Frontmatter übernehmen und verteilen.

Antworte ausschließlich mit einem JSON-Objekt:
{{"pages": [
  {{"slug": "thematischer-slug",
    "title": "Titel",
    "summary": "Kurzbeschreibung",
    "topics": ["tag1", "tag2"],
    "body_md": "Markdown-Inhalt mit 'Siehe auch:'-Links",
    "sources": ["source-id-1", "source-id-2"]}},
  ...
]}}
"""


def call_llm(system: str, user: str, *, source_id: str | None = None) -> dict[str, Any]:
    """Call the LLM API and parse the JSON response. Same pattern as distill.py."""
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
                "temperature": 0.2,
                "max_tokens": 32768,
                "response_format": {"type": "json_object"},
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
                raise

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
        except Exception as e:
            raise RuntimeError(f"Unexpected LLM error: {e}") from e

        if attempt < max_retries - 1:
            backoff = (attempt + 1) * 5
            log("retry", f"waiting {backoff}s before retry…", stderr=True)
            time.sleep(backoff)

    raise RuntimeError(
        f"LLM call failed after {max_retries} attempts for {source_label}: {last_error}"
    ) from last_error


# ── Core Logic ───────────────────────────────────────────────────────────────

def find_large_pages(scopes: tuple[str, ...], max_chars: int) -> list[tuple[str, str, int]]:
    """Find all wiki pages exceeding max_chars in body content.

    Returns list of (scope, slug, body_char_count) sorted by size descending.
    """
    result: list[tuple[str, str, int]] = []
    for scope in scopes:
        scope_dir = WIKI / scope
        if not scope_dir.exists():
            continue
        for p in sorted(scope_dir.glob("*.md")):
            try:
                fm, body = parse_page(p)
            except Exception as e:
                log("warn", f"could not parse {p}: {e}", stderr=True)
                continue
            body_len = len(body)
            if body_len >= max_chars:
                result.append((scope, p.stem, body_len))
    result.sort(key=lambda x: x[2], reverse=True)
    return result


def build_split_prompt(
    slug: str,
    title: str,
    body: str,
    sources: list[str],
    max_chars: int,
) -> str:
    """Build the user prompt for the LLM split request."""
    parts: list[str] = []
    parts.append(f"## Wiki-Seite zum Aufteilen: {slug}")
    parts.append(f"**Titel:** {title}")
    parts.append(f"**Größe:** {len(body):,} Zeichen (Limit: {max_chars:,})")
    parts.append(f"**Sources:** {', '.join(sources) if sources else '(keine)'}")
    parts.append("")
    parts.append("### Vollständiger Inhalt der Seite:")
    parts.append("\n```markdown")
    parts.append(body.strip())
    parts.append("```")
    return "\n".join(parts)


def split_page(
    scope: str,
    slug: str,
    fm: dict[str, Any],
    body: str,
    max_chars: int,
    dry_run: bool = False,
) -> int:
    """Split a single oversized page thematically via LLM.

    Returns the number of new pages created (0 on failure).
    """
    title = fm.get("title", slug)
    sources = fm.get("sources") or []
    folder_path = fm.get("folder_path")
    created_date = fm.get("created", date.today().isoformat())
    today = date.today().isoformat()

    if dry_run:
        log("dry-run", f"WOULD split {scope}/{slug} ({len(body):,} chars) via LLM")
        log("dry-run", f"  title: {title}")
        log("dry-run", f"  sources: {len(sources)} source(s)")
        log("dry-run", f"  WOULD archive {scope}/{slug}")
        # Estimate number of split pages
        est_pages = max(2, len(body) // max_chars)
        return est_pages

    log("info", f"splitting {scope}/{slug} ({len(body):,} chars) → LLM…")

    system = SYSTEM_PROMPT_SPLIT.format(max_chars=max_chars)
    user = build_split_prompt(slug, title, body, sources, max_chars)

    try:
        result = call_llm(system, user, source_id=slug)
    except RuntimeError as e:
        log("error", f"LLM split failed for {scope}/{slug}: {e}", stderr=True)
        return 0

    pages = result.get("pages")
    if not pages or not isinstance(pages, list):
        log("error", f"LLM returned no pages array for {scope}/{slug}", stderr=True)
        return 0

    if len(pages) < 2:
        log("warn",
            f"LLM returned only {len(pages)} page(s) for {scope}/{slug} "
            f"— splitting into 1 page makes no sense, skipping",
            stderr=True)
        return 0

    # Validate: check that each page's body is under max_chars
    oversized = [p.get("slug", "?") for p in pages if len(p.get("body_md", "")) >= max_chars]
    if oversized:
        log("warn",
            f"LLM produced oversized split pages for {scope}/{slug}: {oversized}",
            stderr=True)
        # Continue anyway — better than nothing

    all_slugs = [p["slug"] for p in pages]
    log("info", f"  → {len(pages)} new pages: {', '.join(all_slugs)}")

    # Phase 1: Resolve all slugs (safe + collision-free)
    final_slugs_map: dict[str, str] = {}  # llm_slug → final_slug
    for p in pages:
        p_slug = safe_slug(p["slug"])
        final_slug = resolve_slug_collision(scope, p_slug)
        final_slugs_map[p["slug"]] = final_slug

    # Phase 2: Build page data with correct cross-links
    pages_to_write: list[dict[str, Any]] = []
    for p in pages:
        p_title = p.get("title", p["slug"])
        p_summary = p.get("summary", "")
        p_topics = p.get("topics", [])
        p_body = p.get("body_md", "")
        p_sources = p.get("sources", sources)
        if not isinstance(p_sources, list):
            p_sources = [p_sources]

        # Cross-links using final (collision-resolved) slugs
        other_final_slugs = [final_slugs_map[s] for s in all_slugs if s != p["slug"]]
        if other_final_slugs and "**Siehe auch:**" not in p_body:
            links = " ".join(f"[{s}]({s}.md)" for s in other_final_slugs)
            p_body = p_body.rstrip() + f"\n\n---\n\n**Siehe auch:** {links}\n"

        pages_to_write.append({
            "slug": final_slugs_map[p["slug"]],
            "title": p_title,
            "summary": p_summary,
            "topics": p_topics,
            "body": p_body,
            "sources": p_sources,
            "created": created_date,
            "updated": today,
            "folder_path": folder_path,
            "related_slugs": other_final_slugs,
        })

    # Phase 3: Write all new pages
    for pw in pages_to_write:
        write_page(
            scope, pw["slug"],
            title=pw["title"],
            summary=pw["summary"],
            topics=pw["topics"],
            sources=pw["sources"],
            created=pw["created"],
            updated=pw["updated"],
            body=pw["body"],
            folder_path=pw["folder_path"],
            related_slugs=pw["related_slugs"] if pw["related_slugs"] else None,
        )
        log("info", f"  wrote {scope}/{pw['slug']} ({len(pw['body']):,} chars)")

    # Archive original
    dst = archive_page(scope, slug)
    log("info", f"  archived {scope}/{slug} → {dst.relative_to(ROOT)}")

    # Update backlinks in other wiki pages
    if pages_to_write:
        primary_slug = pages_to_write[0]["slug"]
        bl_count = update_backlinks(scope, slug, primary_slug)
        if bl_count > 0:
            log("info", f"  updated backlinks in {bl_count} page(s) → {primary_slug}")
        else:
            log("info", f"  no backlinks found for {slug} in {scope}")

    return len(pages)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Thematische Aufteilung großer Wiki-Seiten"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Nur anzeigen, was passieren würde — nichts schreiben"
    )
    parser.add_argument(
        "--max-chars", type=int, default=20000,
        help="Body-Char-Limit pro Seite (Default: 20000)"
    )
    parser.add_argument(
        "--scope", choices=_SCOPES, default=None,
        help="Nur diesen Scope scannen (Default: alle)"
    )
    parser.add_argument(
        "--page", type=str, default=None,
        help="Nur diese Seite aufteilen (slug, ohne .md). Benötigt --scope."
    )
    args = parser.parse_args()

    if args.page and not args.scope:
        log("error", "--page erfordert --scope", stderr=True)
        sys.exit(1)

    scopes = (args.scope,) if args.scope else _SCOPES

    # ── Single-page mode ──
    if args.page:
        scope = args.scope  # guaranteed by check above
        page_data = load_page(scope, args.page)
        if page_data is None:
            log("error", f"Seite nicht gefunden: {scope}/{args.page}", stderr=True)
            sys.exit(1)
        body_len = len(page_data["body"])
        if body_len < args.max_chars:
            log("info",
                f"{scope}/{args.page} hat {body_len:,} chars "
                f"< {args.max_chars:,} — kein Split nötig")
            sys.exit(0)
        log("info", f"Split-Anforderung: {scope}/{args.page} ({body_len:,} chars)")
        count = split_page(
            scope, args.page,
            page_data["frontmatter"], page_data["body"],
            args.max_chars, args.dry_run
        )
        if count:
            log("info", f"Fertig: {args.page} → {count} neue Seiten")
        else:
            log("error", f"Split fehlgeschlagen für {args.page}", stderr=True)
            sys.exit(1)
        return

    # ── Scan mode ──
    log("info", f"Scanne {len(scopes)} Scope(s) für Seiten ≥ {args.max_chars:,} Body-Chars…")
    large = find_large_pages(scopes, args.max_chars)

    if not large:
        log("info", "Keine zu großen Seiten gefunden.")
        return

    log("info", f"Gefunden: {len(large)} zu große Seite(n)")
    for scope, slug, size in large:
        log("info", f"  {scope}/{slug}: {size:,} chars")

    if args.dry_run:
        log("info", "--- DRY-RUN Modus ---")

    total_created = 0
    for scope, slug, size in large:
        page_data = load_page(scope, slug)
        if page_data is None:
            log("warn", f"Seite verschwunden während Scan: {scope}/{slug}")
            continue

        count = split_page(
            scope, slug,
            page_data["frontmatter"], page_data["body"],
            args.max_chars, args.dry_run
        )
        total_created += count

        # Brief pause between LLM calls
        if count > 0 and not args.dry_run:
            time.sleep(2)

    log("info", f"=== Fertig: {total_created} neue Seiten erstellt ===")


if __name__ == "__main__":
    main()

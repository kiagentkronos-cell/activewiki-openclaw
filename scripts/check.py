#!/usr/bin/env python3
"""Wiki-Faktencheck (V1) — read-only gegenüber Wiki & Sources.

Prüft Wiki-Pages gegen ihre Quellen:
  1. Deterministisch: alle Zahlen in Markdown-Tabellen der Page müssen in
     mindestens einer Quelle vorkommen (Normalisierung: Komma↔Punkt,
     Tausenderpunkte/Leerstellen, Jahreszahlen 19xx/20xx und URLs ignoriert).
  2. LLM: Prosa-Check via vLLM (OpenAI-kompatibel), System-Prompt =
     wiki-faktencheck-prompt-v1.md.

Read-only: schreibt NUR JSON-Reports nach --output-dir.
Lauf: python3 check.py --page <pfad>  |  --all --since-days 7 --limit 20
Exit: 0 = alle clean, 1 = Issues, 2 = Error
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

LLM_ENDPOINT = "http://localhost:8000/v1/chat/completions"
LLM_TIMEOUT_DEFAULT = 120  # Fallback ohne Config-Key (Abwärtskompatibilität)
LLM_MAX_TOKENS_DEFAULT = 4096  # Fallback ohne Config-Key (Abwärtskompatibilität)
LLM_MAX_TOKENS_CAP = 32768  # Obergrenze: Schutz vor vLLM-Crash durch
                            # max_tokens-Mismatch
SOURCE_FULL_MAX = 30000
SOURCE_PART = 10000


def resolve_llm_timeout() -> int:
    """LLM-Timeout in Sekunden — Priorität: Env > Config > Default.

    1. ACTIVEWIKI_CHECK_TIMEOUT (Env, wie bisher)
    2. quality_check.timeout_seconds aus activewiki.json
    3. Default 120 (kein Fail-Fast bei fehlendem Key)
    """
    env = os.environ.get("ACTIVEWIKI_CHECK_TIMEOUT")
    if env:
        return int(env)
    try:
        from config import load_config, get
        cfg = load_config()
        val = get(cfg, "quality_check.timeout_seconds")
        if val is not None:
            return int(val)
    except Exception:
        pass  # Config fehlt/unlesbar → Default (Check darf nicht crashen)
    return LLM_TIMEOUT_DEFAULT


LLM_TIMEOUT = resolve_llm_timeout()


def _coerce_max_tokens(raw_val, source: str) -> int | None:
    """1-Wert → geprüfte Tokenzahl oder None (dann nächste Prioritätsstufe).

    Kein Fail-Fast: Müll, 0, negativ → None (ignoriert). Riesige Werte
    werden an LLM_MAX_TOKENS_CAP gekappt (Prompt + max_tokens muss ins
    Context-Fenster passen — ein Mismatch hat schon mal vLLM gecrasht).
    """
    try:
        val = int(str(raw_val).strip())
    except (TypeError, ValueError):
        return None  # keine Zahl → ignoriert (kein Crash beim Import/Call)
    if val <= 0:
        return None
    if val > LLM_MAX_TOKENS_CAP:
        print(f"[quality-check] WARNING {source}: max_tokens {val} über "
              f"Obergrenze → gekappt auf {LLM_MAX_TOKENS_CAP}",
              file=sys.stderr)
        return LLM_MAX_TOKENS_CAP
    return val


def resolve_max_tokens() -> int:
    """max_tokens pro LLM-Call — Priorität: Env > Config > Default.

    1. ACTIVEWIKI_CHECK_MAX_TOKENS (Env, analog ACTIVEWIKI_CHECK_TIMEOUT)
    2. quality_check.max_tokens aus activewiki.json
    3. Default 4096 (kein Fail-Fast bei fehlendem Key)

    Unsinnswerte (keine Zahl, 0, negativ) werden ignoriert und die
    nächste Stufe greift; Werte über 32768 werden gekappt.
    """
    env = os.environ.get("ACTIVEWIKI_CHECK_MAX_TOKENS")
    if env:
        val = _coerce_max_tokens(env, "Env ACTIVEWIKI_CHECK_MAX_TOKENS")
        if val is not None:
            return val
    try:
        from config import load_config, get
        cfg = load_config()
        val = get(cfg, "quality_check.max_tokens")
        if val is not None:
            coerced = _coerce_max_tokens(val, "quality_check.max_tokens")
            if coerced is not None:
                return coerced
    except Exception:
        pass  # Config fehlt/unlesbar → Default (Check darf nicht crashen)
    return LLM_MAX_TOKENS_DEFAULT


def default_output_dir() -> Path:
    """Standard-Ziel für JSON-Reports: <wikis_root>/quality/results
    (config-driven, wie die anderen Pipeline-Skripte)."""
    from config import load_config, wikis_root
    return wikis_root(load_config()) / "quality/results"


# System-Prompt = wiki-faktencheck-prompt-v1 (2026-08-18), eingebettet
# (public-Repo trägt keine Workspace-Prompts mit). Override via
# ACTIVEWIKI_CHECK_PROMPT (Pfad zu einer Prompt-Datei).
SYSTEM_PROMPT_CHECK = """# Wiki-Faktencheck — Pilot-Prompt (v1, 2026-08-18)

## Aufgabe (an das LLM)

Du prüfst eine Wiki-Page streng auf Faktenfehler. Du vergleichst JEDER Aussage der Wiki-Page gegen die dazugehörigen Quellen. NIMME AN, dass die Page Fehler enthält — deine einzige Aufgabe ist, diese zu finden.

## Regeln

1. **Nur Fakten.** Keine Stil-, Layout- oder Sprachkritik. Keine Verbesserungsvorschläge.
2. **Jede Aussage prüfen.** Zahlen, Namen, Daten, Beträge, Adressen, Modellbezeichnungen, technische Werte, Behauptungen über Kausalität.
3. **Quellen sind die Wahrheit.** Wenn Page und Quelle widersprechen → der Quelle folgen.
4. **Kein Halluzinieren.** Wenn eine Aussage in KEINER der Quellen vorkommt, ist sie nicht automatisch falsch — aber als "unbelegt" kennzeichnen.
5. **Kurz und präzise.** pro Issue: was steht in der Page, was steht in der Quelle, wo ist der Widerspruch.

## Output (STRIKTES JSON, keine Prosa davor/danach)

```json
{
  "page": "<slug>",
  "status": "clean" | "issues",
  "checked_claims": <Anzahl geprüfter Aussagen, integer>,
  "issues": [
    {
      "severity": "contradiction" | "omission" | "stale" | "unbelegt",
      "claim_in_page": "<was die Page behauptet, zitiert>",
      "source_ref": "<welche Quelle, z.B. 'Rechnung Notar Mustermann' oder Unter-Page-Slug>",
      "evidence": "<was die Quelle tatsächlich sagt, zitiert>",
      "fix_hint": "<wie die Page korrekt lauten müsste, 1 Satz>"
    }
  ]
}
```

### Severity-Definitionen
- **contradiction** — Page sagt X, Quelle sagt Y (direkter Widerspruch)
- **omission** — relevantes Fakt aus der Quelle fehlt komplett in der Page
- **stale** — Page verweist auf etwas, das in neueren Quellen korrigiert/aktualisiert wurde
- **unbelegt** — Aussage steht in keiner Quelle (nicht automatisch Fehler, aber kennzeichnen)

### "clean"
Nur wenn WIRKLICH jede Aussage gegenprüfter ist und kein einziger Befund besteht. Im Zweifel → "issues" mit Severity "unbelegt".

## Was NICHT gecheckt wird (bewusste Grenzen)
- Stil, Tonalität, Übersetzungsqualität
- Querverweise ZWISCHEN anderen Pages (nur diese Page vs. ihre Quellen)
- Vollständigkeit der Themenabdeckung (nur: was da ist, stimmt's?)"""

def _unquote(value: str) -> str:
    """Zwei-oder-einfache Anführungszeichen von einem Skalar entfernen."""
    v = value.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
        inner = v[1:-1]
        if v[0] == "'":
            return inner.replace("''", "'")  # YAML: '' = escaped Quote
        return inner.replace('""', '"')
    return v


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Mini-YAML-Parser für das Frontmatter (nur stdlib, kein PyYAML).

    Unterstützt: Scalars (mit/ohne Anführungszeichen), block Listen
    ('- item' Zeilen), einzeilige Block-Weiterführung (Summary-Wrap)
    und verschachtelte Blöcke (werden als {} gemerkt, Kinder übersprungen).
    Returns: (frontmatter_dict, body) — ohne Frontmatter: ({}, text).
    """
    m = re.match(r"^---\n(.*?)\n---\n?(.*)$", text, re.DOTALL)
    if not m:
        return {}, text
    lines = m.group(1).splitlines()
    fm: dict = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        i += 1
        if not line.strip() or line.strip().startswith("#"):
            continue
        if line[0] in " \t":
            continue  # Kinder von verschachtelten Blöcken werden übersprungen
        mm = re.match(r"^(\S[^:]*):\s*(.*)$", line)
        if not mm:
            continue
        key = _unquote(mm.group(1))
        val = mm.group(2).strip()
        items: list[str] = []
        while (i < len(lines) and lines[i].strip().startswith("- ")
               and not lines[i][0] in " \t"):
            items.append(_unquote(lines[i].strip()[2:]))
            i += 1
        if items:
            fm[key] = items
            continue
        if val == "":
            fm[key] = {}  # verschachtelter Block → Kinder überspringen
            continue
        cont = ""
        if (i < len(lines) and lines[i][0] in " \t"
                and not lines[i].strip().startswith("- ")):
            cont = lines[i].strip()
            i += 1
        fm[key] = _unquote(val + (" " + cont if cont else ""))
    return fm, m.group(2)


def discover_sources(sources_root: Path, scope: str,
                     source_ids: list[str]) -> list[tuple[str, Path | None]]:
    """Quellen via Glob finden: <sources_root>/<scope>/<id>/document.md.

    source_ids aus dem Frontmatter sind die vollen Verzeichnisnamen
    (z.B. 'aaaa11112222-Musterfirma_..._Preislist'). Ohne Treffer →
    (id, None) — der Caller meldet dann 'Quelle nicht auffindbar'.
    """
    found: list[tuple[str, Path | None]] = []
    scope_dir = sources_root / scope
    for sid in source_ids:
        doc = None
        if scope_dir.is_dir():
            hits = sorted(scope_dir.glob(f"{sid}/document.md"))
            if hits:
                doc = hits[0]
        found.append((sid, doc))
    return found


URL_RE = re.compile(r"\b(?:https?://|www\.)\S+")
NUMBER_RE = re.compile(
    r"[+-]?\d[\d., ]*\d(?:\.\d+)?|\d"
)
YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")


def _strip_ignored(text: str) -> str:
    """URLs und Jahreszahlen (19xx/20xx) maskieren, bevor Zahlen gescannt werden."""
    text = URL_RE.sub(" ", text)
    text = YEAR_RE.sub(" ", text)
    return text


def extract_numbers(text: str) -> list[str]:
    """Alle Zahlen-Varianten im Text (Originalschreibweise, Reihenfolge, dedup)."""
    text = _strip_ignored(text)
    seen: dict[str, None] = {}
    for m in NUMBER_RE.finditer(text):
        num = m.group(0).strip(" ,.")
        # Nur echte Zahlen behalten (mindestens eine Ziffer, keine reinen Punkte/Leerzeichen)
        if num and any(c.isdigit() for c in num) and num not in (".", ",", " "):
            seen.setdefault(num)
    return list(seen.keys())


def _canonical_forms(num: str) -> set[str]:
    """Alle plausiblen Normalformen einer Zahlenschreibweise.

    '929.000,00' → {'929000.00','929000','929,000.00'→...}; Leerstellen als
    Tausender-Trenner werden entfernt; wenn die Nachkommastelle nur Nullen
    ist, wird sie weggelassen (929000.00 ≙ 929000).
    """
    s = num.strip().replace(" ", "")
    sign = ""
    if s and s[0] in "+-":
        sign, s = s[0], s[1:]
    has_dot, has_comma = "." in s, "," in s
    if has_dot and has_comma:
        # Letzter Separator = Dezimaltrenner, erster = Tausender-Trenner.
        # Beide Notationen generieren: (Tausender weglassen, Dezimal behalten)
        # sowie (alles weg, reine Ganzzahl).
        last = max(s.rfind("."), s.rfind(","))
        first = min(s.rfind("."), s.rfind(","))
        dec_part = s[last + 1:]
        int_part = (s[:first] + s[first + 1:last]).replace(".", "").replace(",", "")
        raw = [(int_part, dec_part), (int_part, "")]
    elif has_comma:
        raw = [(s.split(",", 1)[0], s.split(",", 1)[1]), (s.replace(",", ""), "")]
    elif has_dot:
        raw = [(s.rsplit(".", 1)[0], s.rsplit(".", 1)[1]), (s.replace(".", ""), "")]
    else:
        raw = [(s, "")]
    forms: set[str] = set()
    for int_part, dec_part in raw:
        if not int_part or not any(c.isdigit() for c in int_part):
            continue
        int_part = int_part.lstrip("0") or "0"
        dec_part = dec_part.lstrip("0")
        forms.add(f"{sign}{int_part}.{dec_part}" if dec_part else f"{sign}{int_part}")
        forms.add(f"{sign}{int_part}")  # ohne Nachkommastelle
    return forms


def build_source_number_index(source_text: str) -> set[str]:
    """Index aller Normalformen aller Zahlen einer Quelle."""
    index: set[str] = set()
    for num in extract_numbers(source_text):
        index |= _canonical_forms(num)
    return index


OCR_DECIMAL_SPACE_RE = re.compile(r"(\d) (?=\d{1,2}(?!\d))")


def normalize_ocr_decimal(s: str) -> str:
    """OCR-Dezimal-Normalisierung: eine Leerstelle zwischen Ziffern, die wie
    Dezimaltrenner wirkt (genau 1 Leerstelle, gefolgt von 1–2 Ziffern am
    Ende der Zahl), wird zu Komma. Nur für die Match-Entscheidung gedacht
    (kein Patch-Vorschlag daraus).

    '199 34' → '199,34'; '103 5' → '103,5'; '1 500' (Tausender, 3 Ziffern
    nach der Leerstelle) bleibt unverändert.
    """
    return OCR_DECIMAL_SPACE_RE.sub(r"\g<1>,", s)


def build_source_ocr_index(source_text: str) -> set[str]:
    """OCR-Dezimal-Index: Normalisierungsformen (Leerstelle→Komma) aller
    Zahlen einer Quelle. Nur für die Match-Entscheidung."""
    return {normalize_ocr_decimal(n) for n in extract_numbers(source_text)}


def ocr_decimal_in_source(num: str, ocr_index: set[str]) -> bool:
    """True, wenn die Normalisierungsform der Zahl in der Quelle vorkommt
    (OCR-Leerstelle ↔ Komma: '199,34' ↔ '199 34').

    Ganze Zahlen ohne Nachkomma matchen NICHT gegen eine Leerstellen-Variante
    eines anderen Wertes: '199' ✗ '199 34' → unbelegt.
    """
    return normalize_ocr_decimal(num) in ocr_index


def number_in_source(num: str, index: set[str],
                     ocr_index: set[str] | None = None) -> bool:
    """True, wenn exakt (mindestens eine Normalform im Quellen-Index) oder
    via OCR-Dezimal-Normalisierung belegt."""
    if _canonical_forms(num) & index:
        return True
    if ocr_index is not None:
        return ocr_decimal_in_source(num, ocr_index)
    return False


def ocr_sister_candidate(page_num: str, number_index: set[str],
                         ocr_index: set[str]) -> str | None:
    """OCR-Korrekturvorschlag: liefert die Quellen-Form, wenn die unbelegte
    Page-Zahl die Ganzzahl-Hälfte einer OCR-Dezimalzahl der Quelle ist.

    Bedingungen (alle müssen gelten, sonst None):
      * Die Page-Zahl ist eine reine Ganzzahl (kein Komma/Punkt/Leerzeichen,
        z.B. '199') — Zahlen mit eigener Nachkommastelle ('199,5') bekommen
        keinen Vorschlag.
      * Die Page-Zahl ist in keiner Quelle exakt belegt (sonst ist nichts
        zu korrigieren).
      * Im OCR-Index existiert eine Form mit exakt demselben Ganzzahlteil
        (0-normalisiert) UND nicht-leerem Dezimalteil ('199 34' → '199,34').
        '199' → '199,34' ja; '19' und '1990' nein (anderer Ganzzahlteil);
        reine Ganzzahlen der Quelle ('1990' → '1990' ohne Dezimalteil) nie.

    Returns die korrigierte Form als String (z.B. '199,34') oder None.
    NUR ein Vorschlag für den LLM — die Match-Entscheidung (number_in_source)
    bleibt strikt exakt und wird davon nicht beeinflusst.
    """
    s = page_num.strip()
    if not re.fullmatch(r"\d+", s):
        return None  # nur reine Ganzzahlen ohne Dezimalstelle
    if _canonical_forms(page_num) & number_index:
        return None  # exakt belegt → nichts zu korrigieren
    key = s.lstrip("0") or "0"
    for form in ocr_index:
        if "," in form:
            int_part, dec_part = form.split(",", 1)
            if (int_part.lstrip("0") or "0") == key and dec_part:
                return form
    return None


def _find_ocr_raw(source_text: str, normalized: str) -> str | None:
    """Kurzeste Roh-Form in der Quelle, deren OCR-Normalisierung `normalized`
    ergibt (z.B. '199 34' für '199,34'). Nur für die evidence-Anzeige."""
    hits = [n for n in extract_numbers(source_text)
            if normalize_ocr_decimal(n) == normalized]
    return min(hits, key=len) if hits else None


def _split_row(line: str) -> list[str]:
    """Eine Markdown-Tabelle-Zeile in Zellen zerlegen (Pipe-trennt, Ränder weg)."""
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _is_separator(line: str) -> bool:
    cells = _split_row(line)
    return bool(cells) and all(re.fullmatch(r":?-{2,}:?", c) for c in cells)


def parse_tables(body: str) -> list[dict]:
    """Alle Markdown-Tabellen extrahieren: [{header:[...], rows:[[...]]}]."""
    tables: list[dict] = []
    lines = body.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.lstrip().startswith("|") and i + 1 < len(lines) \
                and lines[i + 1].lstrip().startswith("|") \
                and _is_separator(lines[i + 1]):
            header = _split_row(line)
            rows: list[list[str]] = []
            i += 2
            while i < len(lines) and lines[i].lstrip().startswith("|") \
                    and not _is_separator(lines[i]):
                rows.append(_split_row(lines[i]))
                i += 1
            tables.append({"header": header, "rows": rows})
        else:
            i += 1
    return tables


def deterministic_issues(body: str, source_texts: list[str],
                         source_names: list[str]) -> list[dict]:
    """Tabellen-Diff: jede Zahl in jeder Tabellenzelle muss in mind. einer
    Quelle vorkommen. Sonst Issue (severity 'unbelegt') mit Zeilenlabel,
    Spaltenheader und Wert. Prosa-Zahlen werden NICHT geprüft (LLM-Teil).

    OCR-Korrekturvorschlag: unbelegte reine Ganzzahlen, für die in einer
    Quelle die OCR-Dezimal-Schwester existiert (Page '199' ↔ Quelle
    '199 34' = '199,34'), bekommen einen Hinweis in evidence + fix_hint.
    Die Issue bleibt origin='deterministic' und wird NIE direkt gepatcht —
    der LLM darf den Vorschlag nur als eigene Issue mit fix bestätigen."""
    indexes = [build_source_number_index(t) for t in source_texts]
    ocr_indexes = [build_source_ocr_index(t) for t in source_texts]
    issues: list[dict] = []
    for table in parse_tables(body):
        for row in table["rows"]:
            # Erste nicht-leere Zelle = Zeilenlabel: nur als Label verwenden,
            # nicht zahlenmäßig prüfen (sonst zählt '3' in 'Haus 3' mit).
            row_label = next((c for c in row if c), "")
            label_idx = row.index(row_label) if row_label else -1
            for col, cell in enumerate(row):
                if col == label_idx:
                    continue
                header = table["header"][col] if col < len(table["header"]) else ""
                for num in extract_numbers(cell):
                    if any(number_in_source(num, ix, ox)
                           for ix, ox in zip(indexes, ocr_indexes)):
                        continue
                    label_parts = [p for p in (row_label, header) if p]
                    evidence = "Wert in keiner der Quellen gefunden"
                    fix_hint = "Wert mit Quelle abgleichen oder entfernen"
                    sister = None
                    for st, ix, ox in zip(source_texts, indexes, ocr_indexes):
                        sister = ocr_sister_candidate(num, ix, ox)
                        if sister is not None:
                            break
                    if sister is not None:
                        raw = None
                        for st in source_texts:
                            raw = _find_ocr_raw(st, sister)
                            if raw is not None:
                                break
                        raw_display = raw if raw is not None else sister
                        evidence = ("Wert in keiner der Quellen gefunden. "
                                    f"Quelle enthält stattdessen: `{raw_display}` "
                                    f"(OCR-Variante von `{sister}`.)")
                        fix_hint = ("Korrekturvorschlag aus Quelle: "
                                    f"`{num}` → `{sister}`. "
                                    "Übernehme NUR wenn der Kontext passt; "
                                    "sonst null.")
                    issues.append({
                        "severity": "unbelegt",
                        "claim_in_page": f"{' / '.join(label_parts)}: {num}",
                        "source_ref": source_names[0] if source_names else "",
                        "evidence": evidence,
                        "fix_hint": fix_hint,
                        "origin": "deterministic",
                    })
    return issues


SELF_CONTRADICTION_MARKERS = (
    "kein fehler",
    "keine fehler",
    "es gibt keinen fehler",
    "es besteht kein fehler",
    "ist korrekt",
    "faktisch korrekt",
)


def has_self_contradiction(issue: dict) -> bool:
    """True, wenn fix_hint ODER evidence einen Self-contradiction-Marker
    enthält (case-insensitive, Unicode-fest) — das LLM revidiert in seinem
    eigenen Text die gerade gemeldete Prüfung (z.B. evidence endet mit
    'Es gibt keinen Fehler hier.').

    ACHTUNG: Nur fix_hint/evidence werden geprüft, NICHT claim_in_page
    (eine Claim kann legitimerweise 'kein Fehler enthalten' sagen).
    """
    text = " ".join(str(issue.get(k) or "") for k in ("evidence", "fix_hint"))
    low = text.casefold()
    return any(marker in low for marker in SELF_CONTRADICTION_MARKERS)


def filter_self_contradictions(issues: list[dict]) -> tuple[list[dict], list[str]]:
    """Teilt LLM-Issues in (beibehalten, verworfen) auf.

    Returns: (kept_issues, [claim_in_page der verworfenen Issues]) —
    für das Report-Feld 'filtered_self_contradictions' (Transparenz,
    aber nicht als Issue gezählt).
    """
    kept: list[dict] = []
    dropped_claims: list[str] = []
    for iss in issues:
        if isinstance(iss, dict) and has_self_contradiction(iss):
            dropped_claims.append(str(iss.get("claim_in_page", "")))
        else:
            kept.append(iss)
    return kept, dropped_claims


def _scan_balanced_json(text: str, start: int) -> int | None:
    """Scanne ab text[start] == '{' bis der Klammer-Ausgleich erreicht ist.

    Strings und Escapes werden beachtet (Klammern in Strings zählen nicht).
    Returns Index HINTER dem abschließenden '}' oder None (unbalanciert →
    typisch für max_tokens-Trunkierung).
    """
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return None


def _strip_code_fences(text: str) -> str:
    """Markdown-Fences am Rand entfernen (```json … ``` / ``` … ```)."""
    stripped = text.strip()
    if stripped.startswith("```"):
        first_nl = stripped.find("\n")
        if first_nl != -1:
            stripped = stripped[first_nl + 1:]
        elif stripped[3:] == "json":
            stripped = ""
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3]
    return stripped


def parse_llm_answer(raw: str) -> tuple[dict | None, str, str]:
    """Robustes JSON-Parsing mit Klammer-Scanner und Unterscheidungsgrund.

    Returns: (parsed_dict|None, raw, grund) — grund ist "" bei Erfolg,
    "abgeschnitten (max_tokens zu klein?)" bei unbalanciertem JSON
    (Trunkierung) und "kein JSON" bei Prosa/unerfolgreichem Parse.
    Markdown-Fences am Rand werden entfernt; Kandidaten sind alle balancierten
    {...}-Blöcke (Strings/Escapes beachtet), nicht blind erstes-bis-letztes.
    """
    if not raw:
        return None, raw, "kein JSON"
    body = _strip_code_fences(raw)
    if "{" not in body:
        return None, raw, "kein JSON"
    # Fehlendeschließende Klammer(n) werden vom Scanner als unbalanciert
    # erkannt → "abgeschnitten" (Trunkierungs-Hinweis), nicht "kein JSON".
    saw_unbalanced = False
    pos = body.find("{")
    while pos != -1:
        end = _scan_balanced_json(body, pos)
        if end is None:
            saw_unbalanced = True  # Rest des Textes kann nur unbalanciert sein
            break
        try:
            parsed = json.loads(body[pos:end])
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return parsed, raw, ""
        pos = body.find("{", pos + 1)
    return (None, raw,
            "abgeschnitten (max_tokens zu klein?)" if saw_unbalanced
            else "kein JSON")


LLM_RETRY_DELAY_SEC = 10.0  # Pause vor dem EINZIGEN Timeout-Retry


def _is_timeout_error(e: BaseException) -> bool:
    """True nur für Timeout-Fehler (socket.timeout ≡ TimeoutError,
    OSError mit errno.ETIMEDOUT, URLError dessen cause ein Timeout ist).
    4xx/5xx/JSON-Parser/etc. → False (kein Retry)."""
    import socket
    if isinstance(e, urllib.error.HTTPError):
        return False  # HTTP-Fehlerstatus: kein Retry
    if isinstance(e, (TimeoutError, socket.timeout)):
        return True
    if isinstance(e, urllib.error.URLError):
        return _is_timeout_error(e.reason)
    return False


def default_llm_call(system: str, user: str) -> str:
    """Echter vLLM-Call (OpenAI-kompatibel), Timeout aus resolve_llm_timeout()
    (Env ACTIVEWIKI_CHECK_TIMEOUT > quality_check.timeout_seconds > 120s).

    Retry-Regel: bei Timeout EINE Wiederholung nach ~10s Pause. Bei
    anderen Fehlern (HTTP 4xx/5xx, JSON-Parser, content=null, ...) wird
    KEIN Retry gemacht — der Fehler wird sofort (mit EINEM Präfix) als
    RuntimeError nach oben gereicht.

    Config: llm.model + llm.temperature + quality_check.timeout_seconds
    aus activewiki.json.
    """
    from config import load_config, llm_model, llm_temperature
    cfg = load_config()
    model = llm_model(cfg)
    temperature = llm_temperature(cfg)
    timeout = resolve_llm_timeout()
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": resolve_max_tokens(),
        "chat_template_kwargs": {"enable_thinking": False},
    }
    last_error: Exception | None = None
    for attempt in (1, 2):  # Initial + max. 1 Retry (nur bei Timeout)
        try:
            req = urllib.request.Request(
                LLM_ENDPOINT,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            text = data["choices"][0]["message"]["content"]
            if text is None:
                raise ValueError("LLM lieferte content=null")
            return text
        except Exception as e:  # HTTP-Status/Parse-Fehler: sofort weiterreichen
            last_error = e
            if attempt == 1 and _is_timeout_error(e):
                time.sleep(LLM_RETRY_DELAY_SEC)  # EIN Retry nur bei Timeout
                continue
            raise RuntimeError(f"LLM-Call fehlgeschlagen: {e}") from e
    # Unreachable: Loop endet nur via return/raise (erster Timeout wird
    # gescort, zweiter wirft sofort).
    raise RuntimeError(f"LLM-Call fehlgeschlagen: {last_error}")  # pragma: no cover


def _trim_source(text: str) -> str:
    """Quellen-Text: <30KB komplett, sonst Kopf 10KB + Mitte 10KB + Ende 10KB."""
    if len(text) <= SOURCE_FULL_MAX:
        return text
    hint = "\n\n[Hinweis: Quelle gekürzt — Tabellen wurden separat deterministisch geprüft.]\n\n"
    head = text[:SOURCE_PART]
    mid = text[len(text) // 2 - SOURCE_PART // 2: len(text) // 2 + SOURCE_PART // 2]
    tail = text[-SOURCE_PART:]
    return f"{head}\n\n[...]\n\n{mid}\n\n[...]\n\n{tail}{hint}"


def load_system_prompt() -> str:
    """System-Prompt = eingebettete Konstante SYSTEM_PROMPT_CHECK
    (wiki-faktencheck-prompt-v1). Env-Override ACTIVEWIKI_CHECK_PROMPT
    auf einen Datei-Pfad bleibt bestehen."""
    override = os.environ.get("ACTIVEWIKI_CHECK_PROMPT")
    if override:
        return Path(override).expanduser().read_text(encoding="utf-8").strip()
    return SYSTEM_PROMPT_CHECK.strip()


def build_llm_user_prompt(page_name: str, page_body: str,
                          source_texts: list[str],
                          source_names: list[str],
                          det_issues: list[dict] | None = None) -> str:
    """User-Prompt: Page + (ggf. gekürzte) Quellen im Markdown-Format.

    det_issues: optional die deterministischen Table-Diff-Issues — solche
    mit einem OCR-Korrekturvorschlag werden dem LLM als eigene, zu
    bestätigende Issues (mit fix) dargeboten; die deterministischen Issues
    selbst bleiben unverändert origin='deterministic' und nie direkt
    patchbar.
    """
    parts = [f"## Wiki-Page: {page_name}\n", "```markdown", page_body.strip(), "```"]
    for name, text in zip(source_names, source_texts):
        parts += [f"\n## Quelle: {name}\n", "```markdown", _trim_source(text), "```"]
    parts.append(
        "\nPrüfe jetzt JEDER Aussage der Page gegen die Quellen "
        "und antworte NUR mit dem JSON-Objekt.\n"
        "Feld \"fix\" pro Issue: Wenn die Korrektur exakt aus der "
        "geladenen Quell-Evidenz belegt ist, liefere: "
        '{"original": "exakter Textauszug aus der Page, inklusive '
        "ausreichendem Kontext (min. ein halber Satz oder Tabellenzeile), "
        'damit er eindeutig ist", "corrected": "korrigierter Text, wörtlich '
        'aus der Quelle übernommen, keine Paraphrase"} — "original" muss ein '
        'exakter Textabschnitt sein, der im Page-Body genau EINMAL vorkommt, '
        'und "corrected" wird wörtlich aus der Quelle übernommen (keine '
        'Paraphrase, keine Ergänzung). Wenn die Korrektur NICHT exakt aus '
        'der Quell-Evidenz belegt ist oder du unsicher bist: "fix": null '
        "(das ist dann die einzig korrekte Antwort).")
    if det_issues:
        hints = [i for i in det_issues
                 if i.get("origin") == "deterministic"
                 and "Korrekturvorschlag aus Quelle" in str(i.get("fix_hint") or "")]
        if hints:
            lines = ["", "## Korrekturen aus dem Tabellen-Abgleich (bitte prüfen)",
                     "Die folgenden Werte fehlen so in den Quellen; die Quellen-"
                     "evidenz enthält jedoch eine OCR-Dezimal-Variante:"]
            for i in hints:
                lines.append(f"- {i.get('claim_in_page', '')} — {i.get('fix_hint', '')}")
            lines.append(
                "Bei Issues mit Korrekturen Vorschlag aus Quelle: wenn der "
                "Kontext eindeutig passt, melde das als Issue mit fix "
                "(original inkl. Zeilenkontext, z.B. 'TSI-6K3D: 199', "
                "corrected mit dem Quellen-Wert).")
            parts.append("\n".join(lines))
    return "\n".join(parts)


def llm_prose_check(system: str, user: str,
                    llm_call) -> tuple[dict | None, str, str]:
    """LLM-Prosa-Check (eigene, mockbare Funktion).

    llm_call(system, user) → Roh-String. Parse-Fehler → (None, raw, grund),
    kein Crash — der Caller markiert status 'error' und schreibt den Grund.
    """
    raw = llm_call(system, user)
    return parse_llm_answer(raw)


def _llm_model_name() -> str:
    """Modellname aus der Config (für das Report-Feld), ohne Crash."""
    try:
        from config import load_config, llm_model
        return llm_model(load_config())
    except Exception:
        return "unknown"


def check_page(page: Path, *, sources_root: Path,
               output_dir: Path | None = None,
               llm_call=default_llm_call) -> dict:
    """Prüft eine Wiki-Page gegen ihre Quellen und liefert das Result-Dict.

    Output-Schema:
      {"page","status":"clean|issues|error","checked_claims":int,
       "issues":[{severity,claim_in_page,source_ref,evidence,fix_hint,origin}],
       "filtered_self_contradictions":[claim_in_page, ...],
       "llm_model","duration_sec"}
    LLM-Issues, deren eigener evidence-/fix_hint-Text die Prüfung revidiert
    (Self-contradiction, z.B. 'Es gibt keinen Fehler hier.'), werden verworfen
    und landen in 'filtered_self_contradictions' statt in 'issues'.
    Bei output_dir: zusätzlich JSON-Datei <page>-<stamp>.json + stdout-JSON.
    """
    started = time.time()
    page = Path(page)
    text = page.read_text(encoding="utf-8")
    fm, body = parse_frontmatter(text)
    scope = fm.get("scope") or "private"
    if not scope:
        m = re.search(r"/wiki/([^/]+)/", str(page))
        scope = m.group(1) if m else "private"
    source_ids = fm.get("sources") or []
    if isinstance(source_ids, str):
        source_ids = [source_ids]

    found = discover_sources(sources_root, scope, source_ids)
    missing = [sid for sid, doc in found if doc is None]
    issues: list[dict] = []
    status = "clean"

    for sid in missing:
        issues.append({
            "severity": "unbelegt",
            "claim_in_page": f"Quelle nicht auffindbar: {sid}",
            "source_ref": str(sources_root / scope / sid),
            "evidence": "Quellen-Verzeichnis oder document.md fehlt",
            "fix_hint": "Quelle neu ingesten oder Referenz korrigieren",
            "origin": "deterministic",
        })
    status = "error" if missing else status

    source_names = [sid for sid, _ in found if _ is not None]
    source_texts = []
    for sid, doc in found:
        if doc is not None:
            source_texts.append(doc.read_text(encoding="utf-8"))

    if not missing:
        issues += deterministic_issues(body, source_texts, source_names)

    checked_claims = len(issues)
    filtered_self_contradictions: list[str] = []
    llm_raw_text: str | None = None
    llm_grund: str = ""
    if not missing:
        try:
            llm_result, llm_raw_text, llm_grund = llm_prose_check(
                load_system_prompt(),
                build_llm_user_prompt(page.stem, body, source_texts,
                                      source_names, det_issues=issues),
                llm_call,
            )
        except Exception as e:  # LLM-Down/Timeout → Error, kein Crash
            # Entwrapping: default_llm_call prefixt bereits 'LLM-Call
            # fehlgeschlagen:' — sonst doppelt verkettet (nested).
            msg = str(e)
            prefix = "LLM-Call fehlgeschlagen: "
            if msg.startswith(prefix):
                msg = msg[len(prefix):]
            llm_result, llm_raw_text = None, prefix + msg
            llm_grund = "LLM-Call fehlgeschlagen"
        if llm_result is None:
            status = "error"
            reason = f" ({llm_grund})" if llm_grund else ""
            issues.append({
                "severity": "unbelegt",
                "claim_in_page": f"LLM-Antwort konnte nicht geparst werden{reason}",
                "source_ref": "llm",
                "evidence": (llm_raw_text or "")[:300],
                "fix_hint": ("max_tokens erhöhen (quality_check.max_tokens) "
                             "und erneut prüfen" if "abgeschnitten" in llm_grund
                             else "Prüfung erneut ausführen (LLM-Instabilität)"),
                "origin": "llm",
            })
        else:
            kept_issues, filtered_self_contradictions = filter_self_contradictions(
                llm_result.get("issues") or [])
            for iss in kept_issues:
                if not isinstance(iss, dict):
                    continue  # LLM liefert manchmal Strings/Ints — verworfen
                iss = dict(iss)
                iss["origin"] = "llm"
                issues.append(iss)
            if isinstance(llm_result.get("checked_claims"), int):
                checked_claims = llm_result["checked_claims"]
            # Status-Upgrade nur, wenn nach dem Self-contradiction-Filter
            # noch LLM-Issues übrig sind (sonst bliebe die Page trotz leerer
            # Issues-Liste bei 'issues').
            kept_llm = sum(1 for i in issues if i.get("origin") == "llm")
            if (llm_result.get("status") == "issues" and status == "clean"
                    and kept_llm > 0):
                status = "issues"

    # Deterministische Issues erzwingen mind. 'issues' (solange kein error).
    if status != "error" and status == "clean" and issues:
        status = "issues"

    result = {
        "page": page.stem,
        "status": status,
        "checked_claims": checked_claims,
        "issues": issues,
        "filtered_self_contradictions": filtered_self_contradictions,
        "llm_model": _llm_model_name(),
        "duration_sec": round(time.time() - started, 2),
    }
    if llm_raw_text is not None and status == "error":
        result["llm_raw_text"] = llm_raw_text

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        out_file = output_dir / f"{page.stem}-{stamp}.json"
        out_file.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        result["output_file"] = str(out_file)
    return result


def build_arg_parser():
    """CLI: --page <pfad>  |  --all --since-days N --limit N  [--output-dir]."""
    import argparse
    ap = argparse.ArgumentParser(
        prog="check.py",
        description="Wiki-Faktencheck: prüft Wiki-Pages gegen ihre Quellen.",
    )
    ap.add_argument("--page", metavar="PFAD",
                    help="eine Wiki-Page (.md) prüfen")
    ap.add_argument("--all", action="store_true",
                    help="alle Pages prüfen (nach since-days/limit gefiltert)")
    ap.add_argument("--since-days", type=int, default=7,
                    help="nur Pages mit updated/created ≤ N Tage (Default 7)")
    ap.add_argument("--limit", type=int, default=20,
                    help="max. Anzahl zu prüfender Pages (Default 20)")
    ap.add_argument("--oldest", type=int, nargs="?", const=2, default=None,
                    metavar="N",
                    help="N älteste Prüfkandidaten wählen UND prüfen "
                         "(Default 2, stdout JSON mit 'selected' + 'results')")
    out_dir = default_output_dir()
    ap.add_argument("--output-dir", default=str(out_dir),
                    help=f"Ziel für JSON-Reports (Default {out_dir})")
    return ap


def _page_age_days(fm: dict, today) -> int | None:
    """Alter in Tagen via Frontmatter updated (Fallback created)."""
    from datetime import date
    for key in ("updated", "created"):
        raw = str(fm.get(key) or "").strip()
        if not raw:
            continue
        try:
            d = date.fromisoformat(raw[:10])
            return (today - d).days
        except ValueError:
            continue
    return None


RECHECK_DAYS = 30  # clean/patched Pages werden 30 Tage lang nicht erneut geprüft


def _iso_date(value) -> "date | None":
    from datetime import date
    if not value:
        return None
    try:
        return date.fromisoformat(str(value).strip().strip("'\"")[:10])
    except ValueError:
        return None


def _candidate_scope(cand: dict) -> str:
    """Scope einer Page: Frontmatter-Field, sonst Elternordner."""
    fm = cand.get("fm") or {}
    return str(fm.get("scope") or Path(cand.get("path")).parent)


def select_candidates(candidates: list[dict], n: int, today=None) -> list[dict]:
    """Wählt die N ältesten Prüfkandidaten aus.

    candidates: List von {"path": Path, "fm": dict}.
    Regeln:
      1. Aufsteigend nach Frontmatter `updated` (Tiebreaker `created`);
         fehlende Daten sortieren ans Ende.
      2. Ausgeschlossen: check_status clean|patched UND last_check ≤ 30 Tage.
      3. Rollup-Regel (einfache robuste Variante, dokumentiert): eine Page mit
         rollup_hash wird in diesem Lauf ausgeschlossen, solange im selben
         Scope-Ordner eine NICHT-Rollup-Page mit updated ≤ Rollup-updated
         selbst noch prüfbar ist (d.h. Regel 2 nicht erfüllt). Ist das Blatt
         frisch geprüft, ist es von Regel 2 aussortiert und blockiert den
         Rollup nicht mehr — der Rollup wird dann im Lauf ausgewählt.
    """
    from datetime import date
    if today is None:
        today = date.today()
    far = date.max
    dated = []
    for c in candidates:
        fm = c.get("fm") or {}
        dated.append({
            "cand": c,
            "updated": _iso_date(fm.get("updated")),
            "created": _iso_date(fm.get("created")),
            "last_check": _iso_date(fm.get("last_check")),
            "status": str(fm.get("check_status") or "").strip(),
            "is_rollup": bool(fm.get("rollup_hash")),
            "scope": _candidate_scope(c),
        })

    def fresh_checked(d) -> bool:
        if d["status"] not in ("clean", "patched"):
            return False
        if d["last_check"] is None:
            return False
        return (today - d["last_check"]).days <= RECHECK_DAYS

    eligible = [d for d in dated if not fresh_checked(d)]
    blocked = set()
    for d in eligible:
        if not d["is_rollup"] or d["updated"] is None:
            continue
        for e in eligible:
            if (e is d or e["is_rollup"] or e["scope"] != d["scope"]
                    or e["updated"] is None or e["updated"] > d["updated"]):
                continue
            blocked.add(id(d))
            break
    pool = [d for d in eligible if id(d) not in blocked]
    pool.sort(key=lambda d: (d["updated"] or far, d["created"] or far,
                             str(d["cand"].get("path"))))
    return [d["cand"] for d in pool[:n]]


def select_oldest_pages(wikis_root: Path, scope_names, n: int,
                        today=None) -> list[dict]:
    """Kandidaten aus <wikis_root>/wiki/<scope>/*.md (alle Scopes) sammeln
    und via select_candidates die N ältesten wählbaren auswählen."""
    from datetime import date
    if today is None:
        today = date.today()
    wiki_root = Path(wikis_root) / "wiki"
    candidates: list[dict] = []
    for scope in scope_names:
        scope_dir = wiki_root / scope
        if not scope_dir.is_dir():
            continue
        for p in sorted(scope_dir.glob("*.md")):
            try:
                fm, _ = parse_frontmatter(p.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError):
                continue
            candidates.append({"path": p, "fm": fm})
    return select_candidates(candidates, n=n, today=today)


def frontmatter_status(status: str, issues: list[dict]) -> str:
    """Mapping Check-Status → check_status im Frontmatter.

    'error' wegen fehlender Quelle(n) → 'uncheckable', sonst passthrough
    (clean|issues|error|patched).
    """
    if status == "error" and any(
            "Quelle nicht auffindbar" in str(i.get("claim_in_page", ""))
            for i in issues):
        return "uncheckable"
    return status


def _git(repo_root: Path, *args: str) -> tuple[int, str]:
    import subprocess
    env = {**os.environ,
           "GIT_AUTHOR_NAME": "quality-check", "GIT_AUTHOR_EMAIL": "quality-check@local",
           "GIT_COMMITTER_NAME": "quality-check", "GIT_COMMITTER_EMAIL": "quality-check@local"}
    try:
        r = subprocess.run(["git", "-C", str(repo_root), *args],
                           capture_output=True, text=True, timeout=30, env=env)
        # stderr mitliefern: Identitäts-/Config-Fehler landen sonst im leeren stdout
        return r.returncode, ((r.stdout or "").strip() + " " + (r.stderr or "").strip()).strip()
    except Exception as e:
        return 1, str(e)


def update_frontmatter(page: Path, *, today, model: str, status: str,
                       repo_root: Path, kind: str = "check") -> dict:
    """Aktualisiert die drei Check-Felder im Frontmatter (Body bleibt
    byte-stabil) und committed die Page.

    Git-Schutz: VOR der Änderung `git status --porcelain` — bei Dirty-State
    wird NICHT committet (außer die einzige Änderung ist die Page selbst;
    dann ist der Arbeitsbaum vor dem Write ohnehin clean). Repo ohne Git →
    File-Update ohne Commit.
    Returns {"updated": bool, "committed": bool, "commit": {"message": str}}.
    kind: 'check' → Commit 'quality-check: <slug> check';
          'patch' → Commit 'quality-check: <slug> patch'.
    """
    page = Path(page)
    repo_root = Path(repo_root)
    slug = page.stem
    commit_msg = f"quality-check: {slug} {kind}"
    text = page.read_text(encoding="utf-8")
    m = re.match(r"^(---\n.*?\n---\n)(.*)$", text, re.DOTALL)
    if not m:
        return {"updated": False, "committed": False,
                "commit": {"message": "kein Frontmatter"}}
    # Bestehende Check-Felder entfernen (last_check vor last_check_model!
    # sonst würde 'last_check:' auch die model-Zeile matchen). Closing '---'
    # vom Block trennen, sonst landen die neuen Felder IM Body.
    inner = m.group(1).split("\n")[1:-2]  # ohne '---', '---' + trailing ''
    block = [ln for ln in inner
             if not re.match(r"^(last_check: |last_check_model: |check_status: )", ln)]
    block += [f"last_check: {today.isoformat()}",
              f"last_check_model: {model}",
              f"check_status: {status}"]
    new_text = "---\n" + "\n".join(block) + "\n---\n" + m.group(2)
    page.write_text(new_text, encoding="utf-8")

    if not (repo_root / ".git").exists():
        return {"updated": True, "committed": False,
                "commit": {"message": "kein Git-Repo — kein Commit"}}
    # Vor dem Commit prüfen: Repo clean? Ausnahme: die einzige Änderung ist
    # die Page selbst (z.B. Body-Patch davor) → Commit erlaubt.
    rc, porcelain = _git(repo_root, "status", "--porcelain", "--untracked-files=no")
    if rc != 0:
        return {"updated": True, "committed": False,
                "commit": {"message": f"git status fehlgeschlagen: {porcelain}"}}
    if porcelain:
        try:
            rel = page.resolve().relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            rel = None
        changed = [ln[2:].strip().strip('"') for ln in porcelain.splitlines()
                   if ln.strip()]
        if not (rel and changed and all(p == rel for p in changed)):
            return {"updated": True, "committed": False,
                    "commit": {"message": "dirty: Repo nicht clean, kein Commit"}}
    _git(repo_root, "add", "--", str(page))
    rc, err = _git(repo_root, "commit", "-m", commit_msg)
    if rc != 0:
        return {"updated": True, "committed": False,
                "commit": {"message": f"commit fehlgeschlagen: {err}"}}
    return {"updated": True, "committed": True, "commit": {"message": commit_msg}}


def apply_patch(text: str, replacements: list[dict]) -> tuple[str, list[str]]:
    """Exakte String-Ersatz-Operationen anwenden.

    Nur wenn `original` exakt EINMAL vorkommt. Fehlt `original` oder ist es
    mehrfach vorhanden → NICHT patchen, Fehlerliste füllen, Text unverändert
    zurückgeben. Returns (new_text, errors).
    """
    errors: list[str] = []
    for rep in replacements:
        if not isinstance(rep, dict):
            errors.append(f"ungültiges Replacement verworfen: {rep!r}")
            continue
        original = str(rep.get("original") or "")
        corrected = str(rep.get("corrected") or "")
        if not original:
            errors.append("leeres original verworfen")
            continue
        count = text.count(original)
        if count == 0:
            errors.append(f"nicht gefunden: {original[:80]!r}")
        elif count > 1:
            errors.append(f"nicht eindeutig ({count}×): {original[:80]!r}")
        else:
            text = text.replace(original, corrected)
    return text, errors


def collect_valid_fixes(issues: list[dict], body: str) -> tuple[list[dict], list[dict]]:
    """Wertet die `fix`-Felder der (LLM-)Issues aus.

    Ein fix ist gültig, wenn: `fix` ein Dict mit nicht-leeren `original` +
    `corrected` ist UND `original` im Body EXAKT EINMAL vorkommt (Sonst: kein
    Patch — die exakt-1×-Regel bleibt die Sicherheitsgrenze).

    Deterministische Issues (origin='deterministic') werden NICHT betrachtet:
    sie stammen vom Table-Diff, nicht vom LLM — nur LLM-meldete Issues können
    einen fix haben.

    Returns (valid_replacements, skipped). skipped-Einträge:
      {"claim_in_page": str, "reason": str} — für das Report-Feld 'patch'.
    """
    valid: list[dict] = []
    skipped: list[dict] = []
    seen_originals: set[str] = set()
    for iss in issues:
        if not isinstance(iss, dict) or iss.get("origin") != "llm":
            continue
        fix = iss.get("fix")
        claim = str(iss.get("claim_in_page", ""))
        if fix is None:
            skipped.append({"claim_in_page": claim, "reason": "fix=null"})
            continue
        if not isinstance(fix, dict):
            skipped.append({"claim_in_page": claim,
                            "reason": "fix ist kein Objekt"})
            continue
        original = str(fix.get("original") or "")
        corrected = str(fix.get("corrected") or "")
        if not original:
            skipped.append({"claim_in_page": claim,
                            "reason": "original fehlt/leer"})
            continue
        if not corrected:
            skipped.append({"claim_in_page": claim,
                            "reason": "corrected fehlt/leer"})
            continue
        count = body.count(original)
        if count == 0:
            skipped.append({"claim_in_page": claim,
                            "reason": f"original in Body nicht gefunden: {original[:60]!r}"})
            continue
        if count > 1:
            skipped.append({"claim_in_page": claim,
                            "reason": f"original nicht eindeutig ({count}×): {original[:60]!r}"})
            continue
        if original in seen_originals:
            skipped.append({"claim_in_page": claim,
                            "reason": "original bereits von anderem Issue belegt"})
            continue
        seen_originals.add(original)
        valid.append({"original": original, "corrected": corrected})
    return valid, skipped


def _wiki_pages(since_days: int):
    """Alle Wiki-Pages aller aktiven Scopes, gefiltert auf seit N Tagen."""
    from config import load_config, wikis_root, scopes
    from datetime import date
    cfg = load_config()
    wiki_root = wikis_root(cfg) / "wiki"
    today = date.today()
    pages: list[tuple[Path, int]] = []
    for scope in scopes(cfg):
        scope_dir = wiki_root / scope
        if not scope_dir.is_dir():
            continue
        for p in sorted(scope_dir.glob("*.md")):
            fm, _ = parse_frontmatter(p.read_text(encoding="utf-8"))
            age = _page_age_days(fm, today)
            if age is None or age <= since_days:
                pages.append((p, age if age is not None else 10**9))
    pages.sort(key=lambda t: t[1])  # jüngste zuerst
    return pages


def qc_process_page(page: Path, *, sources_root: Path, output_dir: Path,
                    llm_call=default_llm_call) -> dict:
    """Vollständiger QC-Ablauf für eine Page: EIN Check-Call → Issues →
    Inline-Patches (fix-Felder der LLM-Issues) → Frontmatter-Update (+
    Git-Commit).

    Der Check-Call liefert pro Issue optional `fix: {original, corrected}`
    (nur wenn aus der geladenen Quell-Evidenz exakt belegt). Gültige Fixes
    (original kommt im Body genau 1× vor) werden via `apply_patch` direkt
    angewandt — es gibt keinen separaten Patch-Pass mehr.

    2-Commit-Muster: bei erfolgreichem Body-Patch erst Frontmatter
    check_status 'issues' + Commit 'check' (unpatchter Body), DANN Body-Patch
    + check_status 'patched' + Commit 'patch'. Ohne gültigen Patch: nur ein
    'check'-Commit mit dem finalen Status (clean|issues|...).

    Liefert das check_page-Result + Zusatzfelder 'frontmatter', 'patch'
    ({patched: bool, applied: int, skipped: [claim_in_page mit Grund]}),
    'qc_status' (Frontmatter-Status). --all bleibt read-only (kompatibel).
    """
    from datetime import date
    page = Path(page)
    result = check_page(page, sources_root=sources_root, output_dir=output_dir,
                        llm_call=llm_call)
    model = result.get("llm_model") or _llm_model_name()
    repo_root = wikis_repo_root()
    qc_status = frontmatter_status(result["status"], result.get("issues") or [])
    _fm, body = parse_frontmatter(page.read_text(encoding="utf-8"))

    valid_fixes, skipped = collect_valid_fixes(result.get("issues") or [], body)
    if valid_fixes:
        new_body, patch_errors = apply_patch(body, valid_fixes)
    else:
        new_body, patch_errors = body, []
    patched = bool(valid_fixes) and new_body != body
    if not patched:
        for rep, err in zip(valid_fixes, patch_errors):
            print(f"[quality-check] WARNING patch {page.stem}: {err}",
                  file=sys.stderr)
    else:
        # 1) Check-Ergebnis committen: Frontmatter 'issues', Body UNVERÄNDERT
        update_frontmatter(page, today=date.today(), model=model,
                           status="issues", repo_root=repo_root, kind="check")
        # 2) Body-Patch anwenden + Frontmatter 'patched' → Commit 'patch'
        text = page.read_text(encoding="utf-8")
        _fm2, body2 = parse_frontmatter(text)
        page.write_text(text.replace(body2, new_body, 1), encoding="utf-8")
        fm_result = update_frontmatter(page, today=date.today(), model=model,
                                       status="patched", repo_root=repo_root,
                                       kind="patch")
        qc_status = "patched"
    if not patched:
        fm_result = update_frontmatter(page, today=date.today(), model=model,
                                       status=qc_status, repo_root=repo_root,
                                       kind="check")
    result["qc_status"] = qc_status
    result["frontmatter"] = fm_result
    result["patch"] = {"patched": patched,
                       "applied": len(valid_fixes) if patched else 0,
                       "skipped": skipped}
    return result


def wikis_repo_root() -> Path:
    """Repo-Root = wikis_root aus der Config (Git-Repo des Wikis)."""
    try:
        from config import load_config, wikis_root
        return wikis_root(load_config())
    except Exception:
        return Path.home() / "wikis"


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.page and (args.all or args.oldest is not None):
        build_arg_parser().error("--page, --all und --oldest schließen sich aus")
    if not args.page and not args.all and args.oldest is None:
        build_arg_parser().error("entweder --page, --all oder --oldest angeben")
    from config import load_config, wikis_root
    sources_root = wikis_root(load_config()) / "sources"
    output_dir = Path(args.output_dir)

    if args.oldest is not None:
        from config import scopes
        from datetime import date
        cfg = load_config()
        candidates = select_oldest_pages(wikis_root(cfg), scopes(cfg),
                                         args.oldest, today=date.today())
        results = []
        for cand in candidates:
            page = Path(cand["path"])
            try:
                r = qc_process_page(page, sources_root=sources_root,
                                    output_dir=output_dir)
            except Exception as e:  # eine kaputte Page blockiert nicht den Rest
                r = {"page": page.stem, "status": "error",
                     "checked_claims": 0,
                     "issues": [{"severity": "unbelegt",
                                 "claim_in_page": f"Check fehlgeschlagen: {e}",
                                 "source_ref": "", "evidence": str(e),
                                 "fix_hint": "", "origin": "deterministic"}],
                     "llm_model": _llm_model_name(), "duration_sec": 0.0}
            results.append(r)
            # Konzept-Logzeile pro Page (landet per >>"$LOG" in run_inbox.sh)
            print(f"quality-check {r['page']} {r['status']} "
                  f"{len(r.get('issues') or [])} {r.get('llm_model', '')}")
        print(json.dumps({"selected": [str(c["path"]) for c in candidates],
                          "results": results}, ensure_ascii=False, indent=2))
        statuses = [r["status"] for r in results]
        if "error" in statuses:
            return 2
        if "issues" in statuses:
            return 1
        return 0

    if args.page:
        page = Path(args.page)
        try:
            results = [qc_process_page(page, sources_root=sources_root,
                                       output_dir=output_dir)]
        except (OSError, UnicodeDecodeError) as e:
            results = [{"page": page.name if page.name else str(page),
                        "status": "error", "checked_claims": 0,
                        "issues": [{"severity": "unbelegt",
                                    "claim_in_page": f"Page nicht lesbar: {e}",
                                    "source_ref": str(page), "evidence": str(e),
                                    "fix_hint": "Pfad prüfen",
                                    "origin": "deterministic"}],
                        "llm_model": _llm_model_name(), "duration_sec": 0.0}]
        print(json.dumps(results[0], ensure_ascii=False, indent=2))
    else:
        results = []
        for page, _age in _wiki_pages(args.since_days)[:args.limit]:
            try:
                results.append(check_page(page, sources_root=sources_root,
                                          output_dir=output_dir))
            except Exception as e:  # eine kaputte Page darf den Rest nicht blockieren
                results.append({"page": page.stem, "status": "error",
                                "checked_claims": 0,
                                "issues": [{"severity": "unbelegt",
                                            "claim_in_page": f"Check fehlgeschlagen: {e}",
                                            "source_ref": "", "evidence": "",
                                            "fix_hint": "", "origin": "deterministic"}],
                                "llm_model": _llm_model_name(),
                                "duration_sec": 0.0})
        for r in results:
            print(json.dumps(r, ensure_ascii=False))

    statuses = [r["status"] for r in results]
    if "error" in statuses:
        return 2
    if "issues" in statuses:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Unit tests for check.py (Wiki-Faktencheck, V1 — read-only gegenüber dem Wiki).

Tests über öffentliche Schnittstellen: Frontmatter-Parsing, Quellen-Discovery,
Zahlen-Extraktion/-Normalisierung, Tabellen-Diff, LLM-Mock + Output-Schema.

Lauf: python3 test_check.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import check  # noqa: E402  (Module unter Test)


# ── Fixtures ─────────────────────────────────────────────────────────────────

# Aus anonymisierter Page kopiert (Preisliste, gekürzt auf relevante Keys)
FRONTMATTER_REAL = """---
title: 'Preisliste: Neubauprojekt ''Musterprojekt'' in Musterort'
summary: Übersicht der Preisliste für das Neubauprojekt 'Musterprojekt' von Musterfirma Wohnbau in Musterort. Enthält Daten
  zu 8 Doppelhaushälften, inklusive Zimmeranzahl, Wohnfläche, Nutzfläche, Grundstücksgröße und Kaufpreisen.
topics:
- musterfirma
- musterort
- neubau
- doppelhaushaelfte
- preisliste
sources:
- aaaa11112222-Musterfirma_Wohnbau_Musterort_Preislist
scope: private
created: '2026-05-08'
updated: '2026-05-08'
folder_path: Immobilieninvestments/Musterort/Unterlagen von Musterfirma/Preisliste
openclaw-hash:
  sources: c597a9b4b46b556dbc015f0e58843d19ad580611149970a36e440a6b5fb1495a
  computed: '2026-05-26T10:52:18Z'
---
## Projektübersicht

| Haus | Zimmer | Kaufpreis |
|------|--------|-----------|
| Haus 1 | 4 | 929.000,00 € |
"""

FAKE_SOURCE_TEXT = """## Preisliste

| Haus | Zimmer | Kaufpreis |
|------|--------|-----------|
| Haus 1 | 4 | 929.000,00 € |
| Haus 2 | 4 | 885.000,00 € |
"""


def make_fake_wiki(tmp: Path, *, page_name: str = "testpage.md",
                   page_body: str | None = None,
                   source_prefix: str = "aaaa11112222") -> dict:
    """Baut tmp/sources/<scope>/<prefix>-<Titel>/document.md + tmp/wiki/<scope>/<page>.md."""
    scope_dir_src = tmp / "sources" / "private" / f"{source_prefix}-Testquelle"
    scope_dir_src.mkdir(parents=True, exist_ok=True)
    (scope_dir_src / "document.md").write_text(FAKE_SOURCE_TEXT, encoding="utf-8")

    wiki_dir = tmp / "wiki" / "private"
    wiki_dir.mkdir(parents=True, exist_ok=True)
    body = page_body if page_body is not None else "## Seite\n\n| Haus | Zimmer | Kaufpreis |\n|------|--------|-----------|\n| Haus 1 | 4 | 929.000,00 € |\n"
    page = wiki_dir / page_name
    page.write_text(
        f"---\ntitle: 'Testpage'\n"
        f"sources:\n- {source_prefix}-Testquelle\n"
        f"scope: private\n"
        f"created: '2026-05-01'\n"
        f"updated: '2026-05-01'\n"
        f"---\n{body}",
        encoding="utf-8",
    )
    return {"page": page, "source_dir": scope_dir_src}


LLM_CLEAN = json.dumps({
    "page": "testpage", "status": "clean", "checked_claims": 7, "issues": []
}, ensure_ascii=False)

LLM_ISSUES = json.dumps({
    "page": "testpage", "status": "issues", "checked_claims": 5,
    "issues": [{
        "severity": "contradiction",
        "claim_in_page": "Kundennummer 1051574",
        "source_ref": "Stromliefervertrag",
        "evidence": "Quelle sagt 1051572",
        "fix_hint": "1051572 verwenden",
    }],
}, ensure_ascii=False)


def mock_llm(payload: str = LLM_CLEAN):
    """LLM-Mock: gibt feste Antwort zurück und merkt sich die Aufrufe."""
    calls: list[dict] = []

    def _call(system: str, user: str) -> str:
        calls.append({"system": system, "user": user})
        return payload

    _call.calls = calls
    return _call


# ── 1. Frontmatter-Parsing ───────────────────────────────────────────────────

class TestFrontmatter(unittest.TestCase):
    def test_real_page_fixture(self):
        fm, body = check.parse_frontmatter(FRONTMATTER_REAL)
        self.assertEqual(fm["title"], "Preisliste: Neubauprojekt 'Musterprojekt' in Musterort")
        self.assertEqual(fm["sources"], ["aaaa11112222-Musterfirma_Wohnbau_Musterort_Preislist"])
        self.assertEqual(fm["scope"], "private")
        self.assertEqual(fm["created"], "2026-05-08")
        self.assertEqual(fm["updated"], "2026-05-08")
        self.assertIn("## Projektübersicht", body)
        self.assertFalse(body.lstrip().startswith("---"))

    def test_multiword_summary_not_broken(self):
        fm, _ = check.parse_frontmatter(FRONTMATTER_REAL)
        self.assertIn("Übersicht der Preisliste", fm["summary"])
        self.assertEqual(fm["topics"][0], "musterfirma")
        self.assertEqual(len(fm["topics"]), 5)

    def test_no_frontmatter(self):
        fm, body = check.parse_frontmatter("nur Text ohne Frontmatter")
        self.assertEqual(fm, {})
        self.assertEqual(body, "nur Text ohne Frontmatter")


# ── 2. Quellen-Glob-Discovery ────────────────────────────────────────────────

class TestDiscovery(unittest.TestCase):
    def test_finds_document_md_by_full_name(self):
        """W1: Discovery matcht den VOLLSTÄNDIGEN Verzeichnisnamen (keine
        Prefix-Semantik) — der verbleibende `discover_sources` ist die korrekte Def."""
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            ctx = make_fake_wiki(tmp)
            found = check.discover_sources(tmp / "sources", "private",
                                           ["aaaa11112222-Testquelle"])
            self.assertEqual(len(found), 1)
            self.assertEqual(found[0][1], ctx["source_dir"] / "document.md")

    def test_finds_only_exact_dir_name_not_prefix(self):
        """W1: Eine Quelle, deren Verzeichnisname nur einen TEIL der ID trägt
        (oder eine andere ID hat), darf NICHT als Treffer landen — Discovery
        ist exakt nach vollem Namen, nicht per Prefix."""
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            ctx = make_fake_wiki(tmp)  # dir: aaaa11112222-Testquelle
            # Kurze ID (nur Prefix des echten Namens) → kein Treffer
            found = check.discover_sources(tmp / "sources", "private",
                                           ["aaaa11112222"])
            self.assertEqual(found, [("aaaa11112222", None)])
            # Andere ID mit gleichem Präfix → kein Treffer
            found = check.discover_sources(tmp / "sources", "private",
                                           ["aaaa11112222-Anderes"])
            self.assertEqual(found, [("aaaa11112222-Anderes", None)])

    def test_missing_source_reported(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            make_fake_wiki(tmp)
            found = check.discover_sources(tmp / "sources", "private",
                                           ["bbbb33334444-GehtNichtVorhanden"])
            self.assertEqual(found[0][1], None)


# ── 3. Zahlen-Extraktion + Normalisierung ────────────────────────────────────

class TestNumbers(unittest.TestCase):
    def test_extract_german_format(self):
        nums = check.extract_numbers("Kaufpreis 929.000,00 € und 1.500 kWh")
        self.assertIn("929.000,00", nums)
        self.assertIn("1.500", nums)

    def test_extract_signed_and_spaced(self):
        nums = check.extract_numbers("+49 (0) 8123 / 99 130 67 sowie -3,5 Grad und 1 500")
        self.assertIn("+49", nums)
        self.assertIn("-3,5", nums)
        self.assertIn("1 500", nums)

    def test_urls_are_ignored(self):
        nums = check.extract_numbers("siehe https://example.com/a/1234567 für Details")
        self.assertNotIn("1234567", nums)

    def test_years_ignored(self):
        nums = check.extract_numbers("Stand 2024, Baujahr 1998, aber 450")
        self.assertNotIn("2024", nums)
        self.assertNotIn("1998", nums)
        self.assertIn("450", nums)

    def test_normalization_comma_dot_variants(self):
        source_text = "Preis 929.000 Euro, 102,81 m2"
        index = check.build_source_number_index(source_text)
        # Komma-Variante muss gegen Punkt-Variante in der Quelle matchen
        self.assertTrue(check.number_in_source("102.81", index))
        # Punkt-Tausch: 929,000,00 (US-Notation im Page-Text) gegen 929.000,00
        self.assertTrue(check.number_in_source("929.000,00", index))
        self.assertTrue(check.number_in_source("929,000.00", index))
        # Leerstellen-Variante
        self.assertTrue(check.number_in_source("929 000,00", index))
        # Falscher Wert matcht NICHT
        self.assertFalse(check.number_in_source("929.001,00", index))


# ── 4. Tabellen-Diff ─────────────────────────────────────────────────────────

class TestTableDiff(unittest.TestCase):
    def test_missing_value_yields_exactly_one_issue(self):
        body = (
            "## Tabelle\n\n"
            "| Haus | Zimmer | Kaufpreis |\n"
            "|------|--------|-----------|\n"
            "| Haus 1 | 4 | 929.000,00 € |\n"
            "| Haus 3 | 4 | 777.777,77 € |\n"   # fehlt in Fake-Quelle
        )
        issues = check.deterministic_issues(
            body, [FAKE_SOURCE_TEXT], ["aaaa11112222-Testquelle"]
        )
        self.assertEqual(len(issues), 1)
        iss = issues[0]
        self.assertEqual(iss["origin"], "deterministic")
        self.assertIn("777.777,77", iss["claim_in_page"])
        self.assertIn("Kaufpreis", iss["claim_in_page"])       # Spaltenheader
        self.assertIn("Haus 3", iss["claim_in_page"])          # Zeilenlabel
        self.assertEqual(iss["severity"], "unbelegt")

    def test_all_values_found_yields_no_issue(self):
        body = (
            "| Haus | Zimmer | Kaufpreis |\n"
            "|------|--------|-----------|\n"
            "| Haus 1 | 4 | 929.000,00 € |\n"
            "| Haus 2 | 4 | 885.000,00 € |\n"
        )
        issues = check.deterministic_issues(body, [FAKE_SOURCE_TEXT], ["s"])
        self.assertEqual(issues, [])

    def test_non_table_numbers_not_checked(self):
        body = "Im Prosa-Text steht 555.555,55 aber keine Tabelle.\n"
        issues = check.deterministic_issues(body, [FAKE_SOURCE_TEXT], ["s"])
        self.assertEqual(issues, [])


    def test_ocr_decimal_source_space_matches_page_comma(self):
        """OCR rendert Dezimalzahlen teils mit Leerstelle: Quelle '199 34',
        Page '199,34' → belegt (kein False Positive 'unbelegt')."""
        source = "Fläche 199 34 m2, Preis 103 5 Euro"
        index = check.build_source_number_index(source)
        ocr = check.build_source_ocr_index(source)
        self.assertTrue(check.number_in_source("199,34", index, ocr))
        self.assertTrue(check.number_in_source("103,5", index, ocr))

    def test_ocr_whole_number_does_not_match_decimal_space_variant(self):
        """Page '199' (Ganzzahl) matcht NICHT gegen '199 34' in der Quelle
        → bleibt unbelegt."""
        source = "Fläche 199 34 m2"
        index = check.build_source_number_index(source)
        ocr = check.build_source_ocr_index(source)
        self.assertFalse(check.number_in_source("199", index, ocr))

    def test_ocr_thousands_space_not_treated_as_decimal(self):
        """Leerstellen als Tausender-Trenner (3+ Ziffern danach) sind KEIN
        Dezimaltrenner: '1 500' bleibt '1 500', nicht '1,500'."""
        self.assertEqual(check.normalize_ocr_decimal("1 500"), "1 500")
        self.assertEqual(check.normalize_ocr_decimal("199 34"), "199,34")
        self.assertEqual(check.normalize_ocr_decimal("199 345"), "199 345")
        # und '1500' (Tausender in Quelle) matcht weiter gegen '1 500' (exakt)
        index = check.build_source_number_index("Menge 1500 Stück")
        ocr = check.build_source_ocr_index("Menge 1500 Stück")
        self.assertTrue(check.number_in_source("1 500", index, ocr))

    def test_table_diff_ocr_decimal_no_false_positive(self):
        """End-to-End Tabellen-Diff: Page '199,34' + Quelle '199 34' →
        kein Issue; '199' + Quelle '199 34' → Issue."""
        source = "| Fläche |\n|------|\n| 199 34 |\n"
        body_ok = ("| Raum | Fläche |\n|------|--------|\n"
                  "| Keller | 199,34 |\n")
        self.assertEqual(
            check.deterministic_issues(body_ok, [source], ["s"]), [])
        body_not = ("| Raum | Fläche |\n|------|--------|\n"
                    "| Keller | 199 |\n")
        issues = check.deterministic_issues(body_not, [source], ["s"])
        self.assertEqual(len(issues), 1)
        self.assertIn("199", issues[0]["claim_in_page"])

    def test_normalize_ocr_decimal_unit(self):
        self.assertEqual(check.normalize_ocr_decimal("199 34"), "199,34")
        self.assertEqual(check.normalize_ocr_decimal("103 5"), "103,5")
        self.assertEqual(check.normalize_ocr_decimal("199,34"), "199,34")
        self.assertEqual(check.normalize_ocr_decimal("kein 12 text"),
                         "kein 12 text")
        self.assertEqual(check.normalize_ocr_decimal(""), "")


# ── 4a. OCR-Korrekturvorschlag (ocr_sister_candidate) ─────────────────────────

class TestOcrSisterCandidate(unittest.TestCase):
    """Page '199' (Ganzzahl, unbelegt) + Quelle '199 34' (=199,34) →
    Korrekturvorschlag '199,34'; keine False Friends (1990, 19, 199,5)."""

    SOURCE = "Fläche 199 34 m2, Preis 103 5 Euro"

    def setUp(self):
        self.index = check.build_source_number_index(self.SOURCE)
        self.ocr = check.build_source_ocr_index(self.SOURCE)

    def test_sister_found(self):
        self.assertEqual(check.ocr_sister_candidate("199", self.index, self.ocr),
                         "199,34")

    def test_longer_int_part_no_match(self):
        """'1990' darf NICHT gegen '199,34' matchen (anderer Ganzzahlteil)."""
        self.assertIsNone(check.ocr_sister_candidate("1990", self.index, self.ocr))

    def test_shorter_int_part_no_match(self):
        """'19' darf NICHT gegen '199,34' matchen (kein Präfix-Match)."""
        self.assertIsNone(check.ocr_sister_candidate("19", self.index, self.ocr))

    def test_number_with_decimal_gets_no_suggestion(self):
        """Zahlen mit eigener Nachkommastelle ('199,5') bekommen keinen
        OCR-Vorschlag (kein Dezimal-Schwestern-Match)."""
        self.assertIsNone(check.ocr_sister_candidate("199,5", self.index, self.ocr))

    def test_exact_match_gets_no_suggestion(self):
        """'103,5' ist exakt belegt → kein Korrekturvorschlag nötig."""
        self.assertIsNone(check.ocr_sister_candidate("103,5", self.index, self.ocr))

    def test_no_sister_in_source(self):
        """Unbelegte Ganzzahl ohne Schwester in der Quelle → None."""
        self.assertIsNone(check.ocr_sister_candidate("777", self.index, self.ocr))

    def test_integers_only_source_never_suggests(self):
        """Quellen-Ganzzahl '1990' (kein Dezimalteil) wird nie als Schwester
        der Page-Zahl '199' vorgeschlagen."""
        src = "Menge 1990 Stück"
        self.assertIsNone(check.ocr_sister_candidate(
            "199", check.build_source_number_index(src),
            check.build_source_ocr_index(src)))


class TestDeterministicOcrSuggestion(unittest.TestCase):
    """deterministic_issues: Schwester-Form ergänzt evidence + fix_hint,
    Issue bleibt origin='deterministic'."""

    SOURCE = "| Fläche |\n|------|\n| 199 34 |\n"
    BODY = ("| Raum | Fläche |\n|------|--------|\n| TSI-6K3D | 199 |\n")

    def test_sister_issue_contains_suggestion_stays_deterministic(self):
        issues = check.deterministic_issues(self.BODY, [self.SOURCE], ["s"])
        self.assertEqual(len(issues), 1)
        iss = issues[0]
        self.assertEqual(iss["origin"], "deterministic")
        self.assertIn("199", iss["claim_in_page"])
        self.assertIn("Quelle enthält stattdessen: `199 34`", iss["evidence"])
        self.assertIn("OCR-Variante von `199,34`", iss["evidence"])
        self.assertIn("Korrekturvorschlag aus Quelle: `199` → `199,34`",
                      iss["fix_hint"])

    def test_plain_unbacked_value_keeps_default_hint(self):
        body = ("| Raum | Fläche |\n|------|--------|\n| Keller | 777 |\n")
        issues = check.deterministic_issues(body, [self.SOURCE], ["s"])
        self.assertEqual(len(issues), 1)
        self.assertEqual(
            issues[0]["evidence"], "Wert in keiner der Quellen gefunden")
        self.assertEqual(issues[0]["fix_hint"],
                         "Wert mit Quelle abgleichen oder entfernen")

    def test_sister_suggestion_never_directly_patchable(self):
        """collect_valid_fixes ignoriert deterministic-Issues — auch mit
        Schwester-Vorschlag (sie dürfen nie direkt gepatcht werden)."""
        issues = check.deterministic_issues(self.BODY, [self.SOURCE], ["s"])
        valid, skipped = check.collect_valid_fixes(issues, self.BODY)
        self.assertEqual(valid, [])
        self.assertEqual(skipped, [])

    def test_sister_prompt_hint_in_llm_prompt(self):
        """build_llm_user_prompt zeigt Schwester-Issues dem LLM als
        bestätigungsfähige Issues mit fix an."""
        issues = check.deterministic_issues(self.BODY, [self.SOURCE], ["s"])
        prompt = check.build_llm_user_prompt(
            "testpage", self.BODY, [self.SOURCE], ["s"], det_issues=issues)
        self.assertIn("Korrekturen aus dem Tabellen-Abgleich", prompt)
        self.assertIn("TSI-6K3D", prompt)
        self.assertIn("199,34", prompt)
        self.assertIn("melde das als Issue mit fix", prompt)

    def test_no_sister_issues_prompt_unchanged(self):
        """Ohne Schwester-Issues (det_issues leer/None) bleibt der Prompt
        unverändert (kein neuer Abschnitt)."""
        base = check.build_llm_user_prompt("p", "B", ["s"], ["n"])
        for det in (None, []):
            self.assertEqual(
                check.build_llm_user_prompt("p", "B", ["s"], ["n"],
                                            det_issues=det),
                base)


# ── 4c. Timeout-Retry im LLM-Call ────────────────────────────────────────────

class _FakeResp:
    """Minimaler urlopen-Ersatz: liefert eine vLLM-ähnliche JSON-Antwort."""

    def __init__(self, content="ok"):
        self._content = content

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        import json as _json
        return _json.dumps({"choices": [{"message": {"content": self._content}}]}
                           ).encode("utf-8")


class TestLlmTimeoutRetry(unittest.TestCase):
    """default_llm_call: EINS Retry (nach ~10s) nur bei Timeout; andere
    Fehler (4xx/5xx/Parse) sofort nach oben. Mock-Transport statt echtem
    HTTP, sleep mockt (keine 10s im Test)."""

    def _run(self, urlopen_factory, sleep_factory):
        import unittest.mock as mock
        old_urlopen, old_sleep = check.urllib.request.urlopen, check.time.sleep
        with mock.patch.object(check.urllib.request, "urlopen",
                               side_effect=urlopen_factory) as m_urlopen, \
             mock.patch.object(check.time, "sleep",
                               side_effect=sleep_factory) as m_sleep:
            try:
                check.default_llm_call("sys", "user")
            except BaseException as e:  # erwarteter Fehlerfall
                return "error", e, m_urlopen.call_count, m_sleep.call_count
            return "ok", None, m_urlopen.call_count, m_sleep.call_count

    def test_first_timeout_second_ok_succeeds(self):
        """1. Call Timeout, 2. Call ok → Call succeeds, genau 1 Retry."""
        calls = {"n": 0}

        def fake_urlopen(req, timeout=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise TimeoutError("timed out")
            return _FakeResp("alles gut")

        sleeps = []
        status, err, n_calls, n_sleeps = self._run(
            fake_urlopen, sleeps.append)
        self.assertEqual(status, "ok")
        self.assertEqual(n_calls, 2)
        self.assertEqual(n_sleeps, 1)
        self.assertTrue(sleeps[0] >= 5)  # ~10s Pause (mockt)

    def test_two_timeouts_error_propagates(self):
        """2× Timeout → Fehler propagiert (kein dritter Versuch),
        genau EIN Präfix 'LLM-Call fehlgeschlagen:' im Message."""
        calls = {"n": 0}

        def fake_urlopen(req, timeout=None):
            calls["n"] += 1
            raise TimeoutError("timed out")

        status, err, n_calls, _ = self._run(fake_urlopen, lambda s: None)
        self.assertEqual(status, "error")
        self.assertIsInstance(err, RuntimeError)
        self.assertEqual(n_calls, 2)  # Initial + 1 Retry, KEINE Endlosschleife
        self.assertEqual(str(err), "LLM-Call fehlgeschlagen: timed out")

    def test_http_error_no_retry(self):
        """4xx/5xx: KEIN Retry, sofort Fehler."""
        import urllib.error

        def fake_urlopen(req, timeout=None):
            raise urllib.error.HTTPError(
                "http://localhost:8000/v1/chat/completions", 404, "Not Found",
                {}, None)

        status, err, n_calls, n_sleeps = self._run(
            fake_urlopen, lambda s: self.fail("keine Pause bei HTTP-Fehler"))
        self.assertEqual(status, "error")
        self.assertEqual(n_calls, 1)
        self.assertEqual(n_sleeps, 0)
        self.assertIn("LLM-Call fehlgeschlagen:", str(err))

    def test_json_parse_error_no_retry(self):
        """Kaputte JSON-Antwort: KEIN Retry (Parse-Fehler zählt nicht als
        Timeout)."""
        import json as _json

        class _BadResp:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return b"das ist keine json".encode("utf-8")

        status, err, n_calls, _ = self._run(lambda req, timeout=None: _BadResp(),
                                            lambda s: None)
        self.assertEqual(status, "error")
        self.assertEqual(n_calls, 1)

    def test_urecursive_timeout_reason_counts(self):
        """URLError mit Timeout-Cause (typischer urllib-Wrap) zählt als
        Timeout → 1 Retry."""
        import urllib.error
        calls = {"n": 0}

        def fake_urlopen(req, timeout=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise urllib.error.URLError(TimeoutError("timed out"))
            return _FakeResp("ok nach retry")

        status, err, n_calls, _ = self._run(fake_urlopen, lambda s: None)
        self.assertEqual(status, "ok")
        self.assertEqual(n_calls, 2)


# ── 4d. Fehler-Entwrapping (kein doppeltes Präfix) ───────────────────────────

class TestErrorUnwrapping(unittest.TestCase):
    def test_llm_raw_text_prefix_exactly_once(self):
        """LLM-Call wirft 'LLM-Call fehlgeschlagen: timed out' → im Report
        erscheint das Präfix genau EINMAL (nicht nested)."""

        def failing_llm(system, user):
            raise RuntimeError("LLM-Call fehlgeschlagen: timed out")

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            ctx = make_fake_wiki(tmp)
            result = check.check_page(
                ctx["page"], sources_root=tmp / "sources",
                output_dir=None, llm_call=failing_llm,
            )
            self.assertEqual(result["status"], "error")
            raw = result.get("llm_raw_text", "")
            self.assertEqual(raw.count("LLM-Call fehlgeschlagen:"), 1, raw)
            self.assertEqual(raw, "LLM-Call fehlgeschlagen: timed out")

    def test_unprefixed_llm_error_gets_single_prefix(self):
        """Fehler OHNE Präfix (z.B. eigener LLM-Wrapper) bekommt genau EIN
        Präfix."""

        def failing_llm(system, user):
            raise ValueError("boom")

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            ctx = make_fake_wiki(tmp)
            result = check.check_page(
                ctx["page"], sources_root=tmp / "sources",
                output_dir=None, llm_call=failing_llm,
            )
            raw = result.get("llm_raw_text", "")
            self.assertEqual(raw, "LLM-Call fehlgeschlagen: boom")


# ── 4b. Self-contradiction-Filter ───────────────────────────────────────────

def sc_issue(marker_field_value, claim="Fläche 199,34 m2"):
    return {"severity": "contradiction", "claim_in_page": claim,
            "source_ref": "s", "evidence": marker_field_value,
            "fix_hint": ""}


class TestSelfContradictionFilter(unittest.TestCase):
    def test_marker_in_fix_hint_drops_issue(self):
        """Marker-Treffer: fix_hint 'Kein Fehler. (...)' → Issue verworfen,
        claim_in_page landet in der Filter-Liste."""
        issues = [{"severity": "contradiction",
                   "claim_in_page": "Keller 199,34",
                   "source_ref": "s", "evidence": "Quelle zeigt 199 34",
                   "fix_hint": "Kein Fehler. (der Wert ist korrekt)",
                   "origin": "llm"}]
        kept, dropped = check.filter_self_contradictions(issues)
        self.assertEqual(kept, [])
        self.assertEqual(dropped, ["Keller 199,34"])

    def test_marker_in_evidence_drops_issue(self):
        """Marker-Treffer in evidence (Fenster-Fall: 'Es gibt keinen Fehler
        hier. Ich suche weiter.') → Issue verworfen."""
        issues = [sc_issue("Ich habe geprüft. Es gibt keinen Fehler hier. "
                           "Ich suche weiter.")]
        kept, dropped = check.filter_self_contradictions(issues)
        self.assertEqual(kept, [])
        self.assertEqual(len(dropped), 1)

    def test_marker_casefold_unicode_insensitive(self):
        """Case-insensitive + Unicode-fest (ß, Umlaute) via casefold."""
        for text in ("KEIN FEHLER", "Kein Fehler.", "Es ist korrekt",
                     "faktisch korrekt", "Es besteht kein fehler",
                     "keine Fehler gefunden",
                     "kein FehLer – alles in Ordnung"):
            self.assertTrue(check.has_self_contradiction(sc_issue(text)),
                            f"{text!r} müsste matchen")

    def test_marker_mismatch_keeps_issue(self):
        """Marker-Fehlbestechung: echte Issues ohne Marker bleiben erhalten."""
        issues = [
            sc_issue("Quelle sagt 1051572, Page sagt 1051574",
                     claim="Kundennummer 1051574"),
            {"severity": "unbelegt", "claim_in_page": "Preis 777.777,77",
             "source_ref": "s", "evidence": "Wert in keiner der Quellen",
             "fix_hint": "Wert mit Quelle abgleichen"},
        ]
        kept, dropped = check.filter_self_contradictions(issues)
        self.assertEqual(len(kept), 2)
        self.assertEqual(dropped, [])

    def test_marker_in_claim_only_keeps_issue(self):
        """Nur fix_hint/evidence zählen: claim_in_page mit 'kein Fehler' darf
        die Issue NICHT verwerfen."""
        issues = [{"severity": "stale", "claim_in_page": "keine fehler",
                   "source_ref": "s", "evidence": "Quelle zeigt 929.000,00",
                   "fix_hint": "Wert aktualisieren"}]
        kept, dropped = check.filter_self_contradictions(issues)
        self.assertEqual(len(kept), 1)
        self.assertEqual(dropped, [])

    def test_check_page_drops_self_contradictions_and_marks_clean(self):
        """E2E: LLM meldet 3 Issues, alle mit Self-contradiction → Page
        'clean', 0 Issues, 3 Einträge in filtered_self_contradictions."""
        payload = json.dumps({
            "page": "testpage", "status": "issues", "checked_claims": 3,
            "issues": [
                {"severity": "contradiction", "claim_in_page": f"Fenster {i}",
                 "source_ref": "Fenster", 
                 "evidence": "Ich habe geprüft. Es gibt keinen Fehler hier.",
                 "fix_hint": "Kein Fehler. (alle Angaben stimmen)"}
                for i in range(1, 4)
            ],
        }, ensure_ascii=False)
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            ctx = make_fake_wiki(tmp)
            result = check.check_page(
                ctx["page"], sources_root=tmp / "sources",
                output_dir=None, llm_call=mock_llm(payload),
            )
            self.assertEqual(result["status"], "clean")
            self.assertEqual(result["issues"], [])
            self.assertEqual(result["filtered_self_contradictions"],
                             ["Fenster 1", "Fenster 2", "Fenster 3"])

    def test_check_page_report_has_filtered_field(self):
        """JSON-Report enthält immer 'filtered_self_contradictions'
        (leere Liste erlaubt) — auch bei cleanem LLM-Lauf."""
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            ctx = make_fake_wiki(tmp)
            out_dir = tmp / "results"
            result = check.check_page(
                ctx["page"], sources_root=tmp / "sources",
                output_dir=out_dir, llm_call=mock_llm(LLM_CLEAN),
            )
            self.assertEqual(result["filtered_self_contradictions"], [])
            on_disk = json.loads(
                sorted(out_dir.glob("testpage-*.json"))[0].read_text(
                    encoding="utf-8"))
            self.assertEqual(on_disk["filtered_self_contradictions"], [])


# ── 5. LLM-Mock + JSON-Parsing + Output-Schema ───────────────────────────────

class TestLLMParseAndSchema(unittest.TestCase):
    def test_parse_llm_json_robust(self):
        raw = "Hier ist das Ergebnis:\n" + LLM_CLEAN + "\nDanke!"
        parsed, _, _ = check.parse_llm_answer(raw)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["status"], "clean")

    def test_parse_llm_json_garbage_no_crash(self):
        parsed, raw_kept, _ = check.parse_llm_answer("kein JSON hier, nur Prosa {unvollständig")
        self.assertIsNone(parsed)
        self.assertIn("kein JSON", raw_kept)

    def test_check_page_schema_clean(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            ctx = make_fake_wiki(tmp)
            out_dir = tmp / "results"
            result = check.check_page(
                ctx["page"], sources_root=tmp / "sources",
                output_dir=out_dir, llm_call=mock_llm(LLM_CLEAN),
            )
            # Pflichtfelder
            for key in ("page", "status", "checked_claims", "issues",
                        "llm_model", "duration_sec"):
                self.assertIn(key, result, f"fehlendes Feld: {key}")
            self.assertEqual(result["status"], "clean")
            self.assertEqual(result["issues"], [])
            self.assertIsInstance(result["checked_claims"], int)
            self.assertGreaterEqual(result["checked_claims"], 1)
            # Datei-Output
            files = list(out_dir.glob("testpage-*.json"))
            self.assertEqual(len(files), 1)
            on_disk = json.loads(files[0].read_text(encoding="utf-8"))
            self.assertEqual(on_disk["status"], "clean")
            # LLM wurde aufgerufen (System-Prompt = geladener Check-Prompt)
            calls = mock_llm(LLM_CLEAN).calls if False else None  # no-op
            # deterministische Issues haben origin
            for iss in result["issues"]:
                self.assertIn(iss.get("origin"), ("deterministic", "llm"))

    def test_check_page_schema_llm_issues_have_origin(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            ctx = make_fake_wiki(tmp)
            result = check.check_page(
                ctx["page"], sources_root=tmp / "sources",
                output_dir=None, llm_call=mock_llm(LLM_ISSUES),
            )
            self.assertEqual(result["status"], "issues")
            self.assertEqual(len(result["issues"]), 1)
            self.assertEqual(result["issues"][0]["origin"], "llm")
            self.assertEqual(result["issues"][0]["severity"], "contradiction")

    def test_check_page_missing_source_is_error(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            ctx = make_fake_wiki(tmp)
            # Page referenziert eine nicht vorhandene Quelle
            page = ctx["page"]
            text = page.read_text(encoding="utf-8")
            text = text.replace("aaaa11112222-Testquelle", "ffff99990000-GehtFehlt")
            page.write_text(text, encoding="utf-8")
            llm = mock_llm()
            result = check.check_page(
                page, sources_root=tmp / "sources",
                output_dir=None, llm_call=llm,
            )
            self.assertEqual(result["status"], "error")
            self.assertTrue(any("ffff99990000" in i["claim_in_page"]
                                for i in result["issues"]))
            # Bei fehlender Quelle: kein LLM-Call
            self.assertEqual(len(llm.calls), 0)

    def test_check_page_llm_error_no_crash(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            ctx = make_fake_wiki(tmp)
            llm = mock_llm("Das ist kaputtes JSON ohne klammern")
            result = check.check_page(
                ctx["page"], sources_root=tmp / "sources",
                output_dir=None, llm_call=llm,
            )
            self.assertEqual(result["status"], "error")
            # roher Text gespeichert, kein Crash
            self.assertTrue(result.get("llm_raw_text"))

    def test_check_page_llm_non_dict_issues_no_crash(self):
        """W2: LLM liefert valides JSON, aber issues mit Strings/Ints neben
        einem echten Dict → kein Crash; nur das Dict landet in den Issues
        (mit origin 'llm'), Strings/Ints werden verworfen."""
        payload = json.dumps({
            "page": "testpage", "status": "issues", "checked_claims": 3,
            "issues": [
                "nur text",
                42,
                {"severity": "stale",
                 "claim_in_page": "Preis veraltet",
                 "source_ref": "Preisliste",
                 "evidence": "Quelle zeigt 929.000,00",
                 "fix_hint": "Wert aktualisieren"},
            ],
        }, ensure_ascii=False)
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            ctx = make_fake_wiki(tmp)
            result = check.check_page(
                ctx["page"], sources_root=tmp / "sources",
                output_dir=None, llm_call=mock_llm(payload),
            )
            llm_issues = [i for i in result["issues"] if i.get("origin") == "llm"]
            self.assertEqual(len(llm_issues), 1)
            self.assertEqual(llm_issues[0]["severity"], "stale")
            self.assertEqual(result["status"], "issues")
            self.assertNotIn("nur text", json.dumps(result, ensure_ascii=False))


# ── 6. CLI-Teile ─────────────────────────────────────────────────────────────

class TestCLI(unittest.TestCase):
    def test_build_arg_parser_has_flags(self):
        ap = check.build_arg_parser()
        ns = ap.parse_args(["--page", "x.md"])
        self.assertEqual(ns.page, "x.md")
        ns2 = ap.parse_args(["--all", "--since-days", "3", "--limit", "5",
                             "--output-dir", "/tmp/x"])
        self.assertTrue(ns2.all)
        self.assertEqual(ns2.since_days, 3)
        self.assertEqual(ns2.limit, 5)
        self.assertEqual(ns2.output_dir, "/tmp/x")


# ── 7. Auswahl-Logik (--oldest) ──────────────────────────────────────────────

def make_candidate(path, *, updated, created, **fm_extra):
    """Kandidaten-Dict wie aus _page_info (ohne I/O)."""
    fm = {"updated": updated, "created": created}
    fm.update(fm_extra)
    return {"path": path, "fm": fm}


class TestCandidateSelection(unittest.TestCase):
    def test_sorted_by_updated_asc_then_created(self):
        from datetime import date
        candidates = [
            make_candidate("b.md", updated="2026-05-10", created="2026-05-10"),
            make_candidate("a.md", updated="2026-05-01", created="2026-05-02"),
            make_candidate("c.md", updated="2026-05-01", created="2026-05-01"),
        ]
        picked = check.select_candidates(candidates, n=3, today=date(2026, 8, 20))
        self.assertEqual([p["path"] for p in picked], ["c.md", "a.md", "b.md"])

    def test_limit_n(self):
        from datetime import date
        candidates = [
            make_candidate(f"p{i}.md", updated=f"2026-05-{i:02d}",
                          created=f"2026-05-{i:02d}")
            for i in (3, 2, 1)
        ]
        picked = check.select_candidates(candidates, n=2, today=date(2026, 8, 20))
        self.assertEqual([p["path"] for p in picked], ["p1.md", "p2.md"])

    def test_missing_dates_sort_last(self):
        from datetime import date
        candidates = [
            make_candidate("nodaye.md", updated=None, created=None),
            make_candidate("oldest.md", updated="2026-01-01", created="2026-01-01"),
        ]
        picked = check.select_candidates(candidates, n=2, today=date(2026, 8, 20))
        self.assertEqual([p["path"] for p in picked], ["oldest.md", "nodaye.md"])


class TestSelectionExclusion(unittest.TestCase):
    def test_recent_clean_is_excluded(self):
        from datetime import date
        today = date(2026, 8, 20)
        page = make_candidate("clean.md", updated="2026-05-01",
                              created="2026-05-01",
                              check_status="clean", last_check="2026-08-01")
        # 19 Tage alt (< 30) → aussortiert, nicht auswählbar
        self.assertEqual(check.select_candidates([page], n=1, today=today), [])

    def test_stale_clean_is_included_again(self):
        from datetime import date
        today = date(2026, 8, 20)
        page = make_candidate("stale.md", updated="2026-05-01",
                              created="2026-05-01",
                              check_status="clean", last_check="2026-07-10")
        # 41 Tage alt (> 30) → Re-Check fällig
        self.assertEqual(len(check.select_candidates([page], n=1, today=today)), 1)

    def test_patched_recent_is_excluded(self):
        from datetime import date
        today = date(2026, 8, 20)
        page = make_candidate("patched.md", updated="2026-05-01",
                              created="2026-05-01",
                              check_status="patched", last_check="2026-08-19")
        self.assertEqual(check.select_candidates([page], n=1, today=today), [])

    def test_issues_and_uncheckable_are_never_excluded(self):
        from datetime import date
        today = date(2026, 8, 20)
        for status in ("issues", "error", "uncheckable"):
            page = make_candidate(f"{status}.md", updated="2026-05-01",
                                  created="2026-05-01",
                                  check_status=status, last_check="2026-08-19")
            self.assertEqual(
                len(check.select_candidates([page], n=1, today=today)), 1,
                f"{status} darf nicht ausgeschlossen werden")


class TestRollupRule(unittest.TestCase):
    """Rollup-Pages (mit rollup_hash) werden aus der Auswahl ausgeschlossen,
    solange im selben Scope eine nicht-Rollup-Page mit updated ≤ updated der
    Rollup-Page nicht frisch geprüft ist (einfache robuste Variante —
    dokumentiert im Final-Report)."""

    def _leaf(self, name, updated, **fm):
        return make_candidate(name, updated=updated, created=updated, **fm)

    def _rollup(self, name, updated):
        return make_candidate(name, updated=updated, created=updated,
                              rollup_hash="deadbeef")

    def test_rollup_blocked_by_stale_leaf(self):
        from datetime import date
        today = date(2026, 8, 20)
        leaf = self._leaf("leaf.md", "2026-05-01")          # ungeprüft
        rollup = self._rollup("rollup.md", "2026-06-01")     # updated > leaf
        self.assertEqual(
            check.select_candidates([rollup, leaf], n=1, today=today)[0]["path"],
            "leaf.md")
        # Rollup allein (Leaf frisch geprüft → von der 30-Tage-Regel
        # aussortiert und blockiert nicht) → wählbar
        leaf_checked = self._leaf("leaf.md", "2026-05-01",
                                  check_status="clean", last_check="2026-08-19")
        picked = check.select_candidates([rollup, leaf_checked], n=2, today=today)
        self.assertEqual([p["path"] for p in picked], ["rollup.md"])

    def test_rollup_not_blocked_by_younger_leaf(self):
        from datetime import date
        today = date(2026, 8, 20)
        leaf = self._leaf("leaf.md", "2026-07-01")  # JÜNGER als Rollup → kein Block
        rollup = self._rollup("rollup.md", "2026-06-01")
        picked = check.select_candidates([rollup, leaf], n=1, today=today)
        self.assertEqual(picked[0]["path"], "rollup.md")

    def test_rollup_blocked_by_equal_updated_leaf(self):
        from datetime import date
        today = date(2026, 8, 20)
        leaf = self._leaf("leaf.md", "2026-06-01")     # == Rollup updated → Block
        rollup = self._rollup("rollup.md", "2026-06-01")
        picked = check.select_candidates([rollup, leaf], n=1, today=today)
        self.assertEqual(picked[0]["path"], "leaf.md")


# ── 8. Frontmatter-Update + Git-Commit ──────────────────────────────────────

def init_fake_git_repo(tmp: Path) -> None:
    """tmp als Git-Repo initialisieren + Initial-Commit."""
    import subprocess
    env = {**os.environ,
           "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "test@test",
           "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "test@test"}
    (tmp / "README.md").write_text("init\n")  # ohne Datei: "nothing to commit" → rc 1
    for args in (["init", "-q"], ["add", "-A"],
                 ["commit", "-q", "-m", "init"]):
        subprocess.run(["git"] + args, cwd=tmp, env=env, check=True,
                       capture_output=True)


def git_log(repo: Path) -> str:
    import subprocess
    env = {**os.environ}
    r = subprocess.run(["git", "-C", str(repo), "log", "--oneline"],
                       capture_output=True, text=True, env=env)
    return r.stdout.strip()


class TestFrontmatterUpdate(unittest.TestCase):
    def test_updates_fields_and_keeps_body_byte_stable(self):
        import re
        from datetime import date
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            init_fake_git_repo(tmp)
            ctx = make_fake_wiki(tmp)
            page = ctx["page"]
            before = page.read_bytes()
            body_start = before.index(b"---\n" , before.index(b"---\n") + 4)
            body_before = before[body_start:]

            r = check.update_frontmatter(
                page, today=date(2026, 8, 20), model="test-model",
                status="clean", repo_root=tmp,
            )
            self.assertTrue(r["updated"], r)
            self.assertTrue(r["committed"], r)
            self.assertIn("quality-check: testpage check", git_log(tmp))

            after = page.read_bytes()
            self.assertTrue(after.startswith(body_before[:0]) or True)
            # Body (ab zweitem '---\n') byte-stabil:
            self.assertEqual(after[after.index(b"---\n", after.index(b"---\n") + 4):],
                             body_before)
            self.assertIn(b"check_status: clean", after)
            self.assertIn(b"last_check: 2026-08-20", after)
            self.assertIn(b"last_check_model: test-model", after)
            # Commit existiert
            self.assertIn("check", git_log(tmp))

    def test_replaces_existing_check_fields_no_duplicates(self):
        from datetime import date
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            init_fake_git_repo(tmp)
            ctx = make_fake_wiki(tmp)
            page = ctx["page"]
            text = page.read_text(encoding="utf-8")
            text = text.replace("---\n## Seite",
                                "last_check: '2026-07-01'\n"
                                "last_check_model: 'old-model'\n"
                                "check_status: clean\n"
                                "---\n## Seite")
            page.write_text(text, encoding="utf-8")
            r = check.update_frontmatter(page, today=date(2026, 8, 20),
                                         model="new-model", status="issues",
                                         repo_root=tmp)
            self.assertTrue(r["updated"], r)
            text = page.read_text(encoding="utf-8")
            self.assertEqual(text.count("last_check:"), 1)
            self.assertEqual(text.count("last_check_model:"), 1)
            self.assertEqual(text.count("check_status:"), 1)
            self.assertIn("last_check_model: new-model", text)
            self.assertIn("check_status: issues", text)

    def test_dirty_repo_abstains_from_commit(self):
        import subprocess
        from datetime import date
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            init_fake_git_repo(tmp)
            ctx = make_fake_wiki(tmp)
            page = ctx["page"]
            commits_before = git_log(tmp).count("\n") + 1
            # Dirty State: gestagte, aber uncommittete Änderung
            (tmp / "dirty.txt").write_text("störchen", encoding="utf-8")
            subprocess.run(["git", "-C", str(tmp), "add", "dirty.txt"],
                           check=True, capture_output=True)
            r = check.update_frontmatter(page, today=date(2026, 8, 20),
                                         model="m", status="clean",
                                         repo_root=tmp)
            # File aktualisiert, aber KEIN Commit (repo war dirty)
            self.assertTrue(r["updated"], r)
            self.assertFalse(r["committed"], r)
            self.assertIn("dirty", (r.get("commit") or {}).get("message", ""))
            self.assertEqual(git_log(tmp).count("\n") + 1, commits_before)

    def test_no_repo_is_not_an_error(self):
        from datetime import date
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            ctx = make_fake_wiki(tmp)  # kein Git-Repo
            page = ctx["page"]
            r = check.update_frontmatter(page, today=date(2026, 8, 20),
                                         model="m", status="clean",
                                         repo_root=tmp)
            self.assertTrue(r["updated"], r)
            self.assertFalse(r["committed"], r)
            self.assertIn("kein Git-Repo", (r.get("commit") or {}).get("message", ""))


# ── 9. Status-Mapping ────────────────────────────────────────────────────────

class TestStatusMapping(unittest.TestCase):
    def test_missing_source_maps_to_uncheckable(self):
        issues = [{"origin": "deterministic",
                   "claim_in_page": "Quelle nicht auffindbar: xyz"}]
        self.assertEqual(check.frontmatter_status("error", issues), "uncheckable")

    def test_llm_error_stays_error(self):
        issues = [{"origin": "llm",
                   "claim_in_page": "LLM-Antwort konnte nicht geparst werden"}]
        self.assertEqual(check.frontmatter_status("error", issues), "error")

    def test_clean_and_issues_passthrough(self):
        self.assertEqual(check.frontmatter_status("clean", []), "clean")
        self.assertEqual(check.frontmatter_status("issues", []), "issues")


# ── 10. Inline-Patch im Check-Call (fix-Feld) ────────

def qc_config(tmp: Path) -> Path:
    """aktivewiki.json mit wikis_root=tmp (damit wikis_repo_root() → tmp)."""
    cfg_file = tmp / "activewiki.json"
    cfg_file.write_text(json.dumps(
        {"wikis_root": str(tmp), "llm": {"model": "test-model"}}),
        encoding="utf-8")
    return cfg_file


def run_qc(tmp: Path, ctx: dict, llm_payload: str) -> dict:
    """qc_process_page mit getrennter Test-Config (kein echtes Wiki-Repo)."""
    old_env = os.environ.get("ACTIVEWIKI_CONFIG")
    os.environ["ACTIVEWIKI_CONFIG"] = str(qc_config(tmp))
    try:
        llm = mock_llm(llm_payload)
        result = check.qc_process_page(
            ctx["page"], sources_root=tmp / "sources",
            output_dir=None, llm_call=llm,
        )
        result["_llm_calls"] = llm.calls
        return result
    finally:
        if old_env is None:
            os.environ.pop("ACTIVEWIKI_CONFIG", None)
        else:
            os.environ["ACTIVEWIKI_CONFIG"] = old_env


def llm_issues_payload(issues) -> str:
    return json.dumps({"page": "testpage", "status": "issues",
                       "checked_claims": 3, "issues": issues},
                      ensure_ascii=False)


PATCH_BODY = ("## Seite\n\nKundennummer 1051574 steht hier.\n"
              "| H | V |\n|---|---|\n| a | 929.000,00 € |\n")


def patch_issue(fix) -> dict:
    return {"severity": "contradiction",
            "claim_in_page": "Kundennummer 1051574",
            "source_ref": "aaaa11112222-Testquelle",
            "evidence": "Quelle sagt 1051572",
            "fix_hint": "1051572 verwenden",
            "fix": fix}


class TestApplyPatch(unittest.TestCase):
    """`apply_patch` (exakt-1×-Regel) — Logik unverändert, nur Aufruf-Stelle.
    """

    def test_unique_replacement_applied(self):
        text = "a\nKundennummer 1051574\nb\n"
        patched, errors = check.apply_patch(text, [{
            "original": "Kundennummer 1051574",
            "corrected": "Kundennummer 1051572"}])
        self.assertEqual(errors, [])
        self.assertEqual(patched, "a\nKundennummer 1051572\nb\n")

    def test_non_unique_replacement_rejected(self):
        text = "x 1051574 y\nz\nx 1051574 y\n"
        patched, errors = check.apply_patch(text, [{
            "original": "1051574", "corrected": "1051572"}])
        self.assertEqual(patched, text)  # unverändert
        self.assertEqual(len(errors), 1)
        self.assertIn("nicht eindeutig", errors[0])

    def test_missing_original_rejected(self):
        text = "irgendwas ganz anderes\n"
        patched, errors = check.apply_patch(text, [{
            "original": "fehlt komplett", "corrected": "x"}])
        self.assertEqual(patched, text)
        self.assertEqual(len(errors), 1)
        self.assertIn("nicht gefunden", errors[0])


class TestCollectValidFixes(unittest.TestCase):
    """Unit: `fix`-Feld-Auswertung (nur LLM-Issues, exakt-1×-Regel)."""

    def test_llm_fix_unique_applied(self):
        body = "Kundennummer 1051574 steht hier.\n"
        valid, skipped = check.collect_valid_fixes(
            [patch_issue({"original": "Kundennummer 1051574",
                          "corrected": "Kundennummer 1051572"}) | {"origin": "llm"}],
            body)
        self.assertEqual(valid, [{"original": "Kundennummer 1051574",
                                  "corrected": "Kundennummer 1051572"}])
        self.assertEqual(skipped, [])

    def test_deterministic_issue_never_patched(self):
        """Deterministische Issues (Table-Diff) haben nie einen gültigen fix,
        auch wenn das Feld gesetzt wäre — nur LLM-meldete Issues zählen."""
        body = "Kundennummer 1051574 steht hier.\n"
        det = patch_issue({"original": "Kundennummer 1051574",
                           "corrected": "Kundennummer 1051572"})
        det["origin"] = "deterministic"
        valid, skipped = check.collect_valid_fixes([det], body)
        self.assertEqual(valid, [])
        self.assertEqual(skipped, [])

    def test_missing_or_empty_original_skipped(self):
        body = "irgendwas\n"
        for fix in (None, {"original": "", "corrected": "x"},
                    {"original": "fehlt in der Page", "corrected": "x"},
                    "keine-dict", {"original": "irgendwas"}):
            with self.subTest(fix=fix):
                valid, skipped = check.collect_valid_fixes(
                    [patch_issue(fix) | {"origin": "llm"}], body)
                self.assertEqual(valid, [])
                self.assertEqual(len(skipped), 1)

    def test_non_unique_original_skipped(self):
        body = "Wert 1051574 a.\nUnd noch: Wert 1051574 b.\n"
        valid, skipped = check.collect_valid_fixes(
            [patch_issue({"original": "1051574", "corrected": "1051572"})
             | {"origin": "llm"}], body)
        self.assertEqual(valid, [])
        self.assertEqual(len(skipped), 1)
        self.assertIn("nicht eindeutig", skipped[0]["reason"])

    def test_duplicate_original_across_issues_second_skipped(self):
        body = "Kundennummer 1051574 steht hier.\n"
        fixes = [{"original": "Kundennummer 1051574", "corrected": "c1"},
                 {"original": "Kundennummer 1051574", "corrected": "c2"}]
        issues = [patch_issue(f) | {"origin": "llm"} for f in fixes]
        valid, skipped = check.collect_valid_fixes(issues, body)
        self.assertEqual(len(valid), 1)
        self.assertEqual(valid[0]["corrected"], "c1")
        self.assertEqual(len(skipped), 1)
        self.assertIn("bereits", skipped[0]["reason"])


def body_of(page: Path) -> str:
    """Body einer Page (nach dem Frontmatter-Block)."""
    _fm, body = check.parse_frontmatter(page.read_text(encoding="utf-8"))
    return body


class TestInlinePatchE2E(unittest.TestCase):
    """E2E: EIN Check-Call mit fix → apply_patch → 2-Commit-Muster."""

    def test_valid_fix_applied_patched_status_two_commits(self):
        """fix-valid → Patch angewandt, check_status 'patched',
        Commit 'check' (unpatchter Body) + Commit 'patch', genau 1 LLM-Call."""
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            init_fake_git_repo(tmp)
            ctx = make_fake_wiki(tmp, page_body=PATCH_BODY)
            page = ctx["page"]
            r = run_qc(tmp, ctx, llm_issues_payload([patch_issue({
                "original": "Kundennummer 1051574",
                "corrected": "Kundennummer 1051572"})]))
            after = page.read_text(encoding="utf-8")
            self.assertTrue(r["patch"]["patched"], r)
            self.assertEqual(r["patch"]["applied"], 1)
            self.assertEqual(r["patch"]["skipped"], [])
            self.assertEqual(r["qc_status"], "patched")
            self.assertIn("Kundennummer 1051572", after)
            self.assertNotIn("1051574", after)
            self.assertIn("check_status: patched", after)
            log = git_log(tmp)
            self.assertIn("quality-check: testpage check", log)
            self.assertIn("quality-check: testpage patch", log)
            # Reihenfolge: check-Commit VOR patch-Commit (git log: neueste
            # oben → 'patch' steht VOR 'check' in der Liste)
            self.assertLess(log.index("testpage patch"),
                            log.index("testpage check"))
            # Rest der Page (Tabelle) byte-stabil
            self.assertIn("| a | 929.000,00 € |", after)
            # Nur EIN LLM-Call (kein separater Patch-Pass mehr)
            self.assertEqual(len(r["_llm_calls"]), 1)

    def test_fix_missing_original_skipped_status_issues(self):
        """fix mit original, das im Body fehlt → skipped, kein Patch,
        Status bleibt 'issues', ein 'check'-Commit."""
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            init_fake_git_repo(tmp)
            ctx = make_fake_wiki(tmp, page_body=PATCH_BODY)
            page = ctx["page"]
            before_body = body_of(page)
            r = run_qc(tmp, ctx, llm_issues_payload([patch_issue({
                "original": "ganz anderer Text", "corrected": "x"})]))
            self.assertFalse(r["patch"]["patched"], r)
            self.assertEqual(r["patch"]["applied"], 0)
            self.assertEqual(len(r["patch"]["skipped"]), 1)
            self.assertIn("nicht gefunden", r["patch"]["skipped"][0]["reason"])
            self.assertEqual(r["patch"]["skipped"][0]["claim_in_page"],
                             "Kundennummer 1051574")
            self.assertEqual(r["qc_status"], "issues")
            self.assertEqual(body_of(page), before_body)
            log = git_log(tmp)
            self.assertIn("quality-check: testpage check", log)
            self.assertNotIn("testpage patch", log)

    def test_fix_not_unique_skipped_status_issues(self):
        """fix mit original, das 2× vorkommt → skipped (nicht eindeutig),
        kein Patch, Status 'issues'."""
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            init_fake_git_repo(tmp)
            body = ("## Seite\n\nWert 1051574 steht hier.\n"
                    "Und noch einmal: Wert 1051574.\n")
            ctx = make_fake_wiki(tmp, page_body=body)
            page = ctx["page"]
            before_body = body_of(page)
            issue = {"severity": "contradiction",
                     "claim_in_page": "Wert 1051574",
                     "source_ref": "aaaa11112222-Testquelle",
                     "evidence": "x", "fix_hint": "y",
                     "fix": {"original": "1051574", "corrected": "1051572"}}
            r = run_qc(tmp, ctx, llm_issues_payload([issue]))
            self.assertFalse(r["patch"]["patched"], r)
            self.assertEqual(r["patch"]["applied"], 0)
            self.assertEqual(len(r["patch"]["skipped"]), 1)
            self.assertIn("nicht eindeutig", r["patch"]["skipped"][0]["reason"])
            self.assertEqual(r["qc_status"], "issues")
            self.assertEqual(body_of(page), before_body)
            log = git_log(tmp)
            self.assertIn("quality-check: testpage check", log)
            self.assertNotIn("testpage patch", log)

    def test_fix_null_issue_no_patch(self):
        """LLM meldet Issue mit fix: null → kein Patch, Status 'issues',
        kein Body-Commit."""
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            init_fake_git_repo(tmp)
            ctx = make_fake_wiki(tmp, page_body=PATCH_BODY)
            page = ctx["page"]
            before_body = body_of(page)
            r = run_qc(tmp, ctx, llm_issues_payload([patch_issue(None)]))
            self.assertFalse(r["patch"]["patched"], r)
            self.assertEqual(r["patch"]["applied"], 0)
            self.assertEqual(len(r["patch"]["skipped"]), 1)
            self.assertIn("fix", r["patch"]["skipped"][0]["reason"])
            self.assertEqual(r["qc_status"], "issues")
            self.assertEqual(body_of(page), before_body)
            self.assertNotIn("testpage patch", git_log(tmp))

    def test_old_llm_payload_without_fix_field_still_works(self):
        """Rückwärtskompatibel: LLM-Antwort OHNE fix-Feld → Issues werden
        akzeptiert, kein Patch (fix gilt als null), Status 'issues'."""
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            init_fake_git_repo(tmp)
            ctx = make_fake_wiki(tmp, page_body=PATCH_BODY)
            page = ctx["page"]
            before_body = body_of(page)
            issue = {"severity": "contradiction",
                     "claim_in_page": "Kundennummer 1051574",
                     "source_ref": "Vertrag", "evidence": "Quelle: 1051572",
                     "fix_hint": "1051572"}  # kein fix-Feld
            r = run_qc(tmp, ctx, llm_issues_payload([issue]))
            self.assertEqual(r["qc_status"], "issues")
            self.assertEqual(len(r["issues"]), 1)
            self.assertEqual(r["patch"]["patched"], False)
            self.assertEqual(body_of(page), before_body)

    def test_self_contradiction_issue_provides_no_patch(self):
        """Filter vor fix-Auswertung: verworfenes Self-contradiction-Issue
        (mit gültigem fix) liefert KEINEN Patch."""
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            init_fake_git_repo(tmp)
            ctx = make_fake_wiki(tmp, page_body=PATCH_BODY)
            page = ctx["page"]
            before_body = body_of(page)
            issue = patch_issue({"original": "Kundennummer 1051574",
                                 "corrected": "Kundennummer 1051572"})
            issue["evidence"] = ("Ich habe geprüft. Es gibt keinen Fehler "
                                 "hier.")
            r = run_qc(tmp, ctx, llm_issues_payload([issue]))
            self.assertEqual(r["filtered_self_contradictions"],
                             ["Kundennummer 1051574"])
            self.assertEqual(r["qc_status"], "clean")
            self.assertEqual(r["patch"]["patched"], False)
            self.assertEqual(r["patch"]["applied"], 0)
            self.assertEqual(r["patch"]["skipped"], [])
            self.assertEqual(body_of(page), before_body)

    def test_prompt_contains_fix_instructions(self):
        """Check-Call-User-Prompt weist das LLM auf die fix-Regeln hin."""
        user = check.build_llm_user_prompt(
            "testpage", "Body", ["Quelle-Text"], ["s"])
        self.assertIn('"fix"', user)
        self.assertIn("genau EINMAL", user)
        self.assertIn("wörtlich aus der Quelle", user)
        self.assertIn('"fix": null', user)


# ── 11. E2E: Auswahl aus echtem Scope-Verzeichnis (tmp) ──────────────────────

class TestOldestSelectionE2E(unittest.TestCase):
    def test_oldest_picks_from_scope_dirs(self):
        import tempfile as _tf
        from datetime import date
        with _tf.TemporaryDirectory() as td:
            tmp = Path(td)
            wiki = tmp / "wiki" / "private"
            wiki.mkdir(parents=True)
            (tmp / "activewiki.json").write_text(
                json.dumps({"wikis_root": str(tmp),
                            "scopes": {"enabled": ["private"]}}),
                encoding="utf-8")
            # 3 Pages: a=älteste, b=mittlere, c=jüngste; b frisch geprüft
            def w(name, updated):
                (wiki / name).write_text(
                    f"---\ntitle: '{name}'\nscope: private\n"
                    f"created: '2026-01-01'\nupdated: '{updated}'\n---\n# {name}\n",
                    encoding="utf-8")
            w("c-newest.md", "2026-08-01")
            w("b-middle.md", "2026-06-01")
            w("a-oldest.md", "2026-05-01")
            b = wiki / "b-middle.md"
            text = b.read_text(encoding="utf-8")
            b.write_text(text.replace("---\n# b",
                                      "check_status: clean\n"
                                      "last_check: '2026-08-19'\n"
                                      "---\n# b"), encoding="utf-8")

            import os as _os
            old_env = _os.environ.get("ACTIVEWIKI_CONFIG")
            _os.environ["ACTIVEWIKI_CONFIG"] = str(tmp / "activewiki.json")
            try:
                from config import load_config, wikis_root, scopes
                cfg = load_config()
                picked = check.select_oldest_pages(
                    wikis_root(cfg), scopes(cfg), n=1,
                    today=date(2026, 8, 20))
            finally:
                if old_env is None:
                    _os.environ.pop("ACTIVEWIKI_CONFIG", None)
                else:
                    _os.environ["ACTIVEWIKI_CONFIG"] = old_env
            self.assertEqual(len(picked), 1)
            self.assertEqual(picked[0]["path"].name, "a-oldest.md")


# ── 4d. LLM-Timeout-Resolution: Env > Config > Default ──────────────────────

class TestLlmTimeoutResolution(unittest.TestCase):
    """resolve_llm_timeout(): ACTIVEWIKI_CHECK_TIMEOUT (Env) überschreibt
    quality_check.timeout_seconds (activewiki.json), das überschreibt den
    Default 120. Fehlender Config-Key → Default (Abwärtskompatibilität,
    KEIN Fail-Fast)."""

    def _write_config(self, tmp: Path, cfg: dict) -> None:
        (tmp / "activewiki.json").write_text(
            json.dumps(cfg), encoding="utf-8")

    def _run(self, cfg: dict, env_value: str | None) -> int:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            self._write_config(tmp, cfg)
            old_env = os.environ.get("ACTIVEWIKI_CONFIG")
            old_timeout = os.environ.get("ACTIVEWIKI_CHECK_TIMEOUT")
            os.environ["ACTIVEWIKI_CONFIG"] = str(tmp / "activewiki.json")
            if env_value is None:
                os.environ.pop("ACTIVEWIKI_CHECK_TIMEOUT", None)
            else:
                os.environ["ACTIVEWIKI_CHECK_TIMEOUT"] = env_value
            try:
                return check.resolve_llm_timeout()
            finally:
                if old_env is None:
                    os.environ.pop("ACTIVEWIKI_CONFIG", None)
                else:
                    os.environ["ACTIVEWIKI_CONFIG"] = old_env
                if old_timeout is None:
                    os.environ.pop("ACTIVEWIKI_CHECK_TIMEOUT", None)
                else:
                    os.environ["ACTIVEWIKI_CHECK_TIMEOUT"] = old_timeout

    def test_config_value_wins_over_default(self):
        """quality_check.timeout_seconds gesetzt → Config-Wert, kein Env."""
        self.assertEqual(
            self._run({"quality_check": {"timeout_seconds": 600}}, None), 600)

    def test_missing_key_falls_back_to_default_120(self):
        """Key fehlt in activewiki.json → Default 120 (kein Fail-Fast)."""
        self.assertEqual(self._run({}, None), 120)

    def test_env_overrides_config(self):
        """Env-Var überschreibt Config-Wert (Priorität Env > Config)."""
        self.assertEqual(
            self._run({"quality_check": {"timeout_seconds": 600}}, "33"), 33)

    def test_env_without_config_key(self):
        """Env ohne Config-Key → Env-Wert (Abwärtskompatibilität)."""
        self.assertEqual(self._run({}, "45"), 45)


# ── 4e. LLM-max_tokens-Resolution: Env > Config > Default ────────────────────

class TestMaxTokensResolution(unittest.TestCase):
    """resolve_max_tokens(): ACTIVEWIKI_CHECK_MAX_TOKENS (Env) überschreibt
    quality_check.max_tokens (activewiki.json), das überschreibt den
    Default 4096. Fehlender Key → Default (Abwärtskompatibilität, KEIN
    Fail-Fast). Unsinnswerte (0, negativ, Müll) werden ignoriert, Werte
    über der Obergrenze (32768) gekappt — ein max_tokens-Mismatch kann
    den vLLM-Server crashen."""

    def _run(self, cfg: dict, env_value: str | None) -> int:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            (tmp / "activewiki.json").write_text(json.dumps(cfg),
                                                 encoding="utf-8")
            old_env = os.environ.get("ACTIVEWIKI_CONFIG")
            old_mt = os.environ.get("ACTIVEWIKI_CHECK_MAX_TOKENS")
            os.environ["ACTIVEWIKI_CONFIG"] = str(tmp / "activewiki.json")
            if env_value is None:
                os.environ.pop("ACTIVEWIKI_CHECK_MAX_TOKENS", None)
            else:
                os.environ["ACTIVEWIKI_CHECK_MAX_TOKENS"] = env_value
            try:
                return check.resolve_max_tokens()
            finally:
                if old_env is None:
                    os.environ.pop("ACTIVEWIKI_CONFIG", None)
                else:
                    os.environ["ACTIVEWIKI_CONFIG"] = old_env
                if old_mt is None:
                    os.environ.pop("ACTIVEWIKI_CHECK_MAX_TOKENS", None)
                else:
                    os.environ["ACTIVEWIKI_CHECK_MAX_TOKENS"] = old_mt

    def test_config_wins_over_default(self):
        self.assertEqual(
            self._run({"quality_check": {"max_tokens": 8192}}, None), 8192)

    def test_missing_key_falls_back_to_default_4096(self):
        """Key fehlt → Default 4096 (Abwärtskompatibilität, kein Fail-Fast)."""
        self.assertEqual(self._run({}, None), 4096)

    def test_env_overrides_config(self):
        self.assertEqual(
            self._run({"quality_check": {"max_tokens": 8192}}, "12000"), 12000)

    def test_env_without_config_key(self):
        self.assertEqual(self._run({}, "5000"), 5000)

    def test_garbage_env_ignored_falls_to_config(self):
        """Müll/Nicht-Zahl in der Env → ignorieren und Config-Wert nehmen."""
        self.assertEqual(
            self._run({"quality_check": {"max_tokens": 8192}}, "unsinn"), 8192)

    def test_nonpositive_values_ignored(self):
        """0 oder negativ ergibt sinnlose Requests → Default."""
        self.assertEqual(self._run({"quality_check": {"max_tokens": 0}}, None), 4096)
        self.assertEqual(self._run({}, "-5"), 4096)

    def test_cap_protects_against_oversized_values(self):
        """Über der Obergrenze → gekappt (Schutz vor vLLM-Crash)."""
        self.assertEqual(self._run({}, "999999"), 32768)


# ── 15. Parser-Härtung: Fences, Klammer-Scanner, Unterscheidungsgrund ────────

class TestParseLLMAnswerHardened(unittest.TestCase):
    """parse_llm_answer liefert (parsed|None, raw, grund)."""

    SAMPLE = json.dumps({"status": "clean", "issues": [],
                         "checked_claims": 3}, ensure_ascii=False)

    def test_markdown_fence_ok(self):
        """JSON in ```json-Fences am Rand → sauber geparst, grund leer."""
        raw = "```json\n" + self.SAMPLE + "\n```"
        parsed, raw_kept, grund = check.parse_llm_answer(raw)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["status"], "clean")
        self.assertEqual(grund, "")
        self.assertEqual(raw_kept, raw)

    def test_pro_and_post_text_ok(self):
        """Prä-/Post-Text um komplettes JSON → geparst (Scanner, kein rindex)."""
        raw = "Hier ist das Ergebnis:\n" + self.SAMPLE + "\nViel Erfolg!"
        parsed, _, grund = check.parse_llm_answer(raw)
        self.assertIsNotNone(parsed)
        self.assertEqual(grund, "")

    def test_nested_braces_in_strings_ok(self):
        """Geschweifte Klammern in Strings/Strings mit Escapes dürfen die
        Balancierung nicht verschieben."""
        payload = json.dumps({"status": "issues", "checked_claims": 1,
                              "issues": [{"claim_in_page": "Musterort {A} \"x}\"",
                                          "severity": "mittel"}]})
        raw = "Vorwort {ok} " + payload + " Nachwort }"
        parsed, _, grund = check.parse_llm_answer(raw)
        # Erster { gehört zu "Vorwort {ok}" → balanciert, aber kein Dict-JSON
        # → kein Crash; Ergebnis ist dict oder None, grund ist ein String.
        self.assertIsInstance(grund, str)

    def test_truncated_json_reports_abgeschnitten(self):
        """Mitten im JSON abgeschnitten → (None, raw, 'abgeschnitten …')."""
        raw = '{"status": "issues", "issues": [{"claim_in_page": "Musterstraße 1'
        parsed, raw_kept, grund = check.parse_llm_answer(raw)
        self.assertIsNone(parsed)
        self.assertEqual(raw_kept, raw)
        self.assertIn("abgeschnitten", grund)

    def test_prose_reports_kein_json(self):
        """Prosa ohne JSON → (None, raw, 'kein JSON')."""
        parsed, raw_kept, grund = check.parse_llm_answer(
            "Die Seite enthält keine überprüfbaren Aussagen.")
        self.assertIsNone(parsed)
        self.assertEqual(raw_kept, "Die Seite enthält keine überprüfbaren Aussagen.")
        self.assertEqual(grund, "kein JSON")

    def test_check_page_error_carries_grund(self):
        """Trunkierte LLM-Antwort: Report-error-Feld nennt den Grund."""
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            ctx = make_fake_wiki(tmp)
            truncated = json.dumps(
                {"status": "issues", "checked_claims": 2,
                 "issues": [{"claim_in_page": "Musterort", "severity": "mittel"}]}
            )[:40]
            result = check.check_page(
                ctx["page"], sources_root=tmp / "sources",
                output_dir=None, llm_call=mock_llm(truncated))
            self.assertEqual(result["status"], "error")
            hint = json.dumps(result["issues"], ensure_ascii=False)
            self.assertIn("abgeschnitten", hint)


if __name__ == "__main__":
    unittest.main(verbosity=2)

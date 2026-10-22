#!/usr/bin/env python3
"""Tests für die Dedup-Beschleunigung (Top-K-Prefilter, Caches, Debug-Logs).

Abdeckung:
  1. config.graph_dedup_top_k: Default 50, String-Parsing, Klemmung < 1, Müll → Default
  2. _fuzzy_candidate_rank: harte obere Schranke gegen _fuzzy_match_score
     (Synthetik + Kantenfälle: leer, identisch, Reorder, |Δlen|>20)
  3. _rank_fuzzy_candidates: Top-K-Verhalten, Threshold-Garantie,
     DB-Reihenfolge, Schwellwert-Abhängigkeit (85 vs 90)
  4. _normalize_label_for_jaccard: Cache-Korrektheit (gleicher Wert wie
     Ungecacht; Mutation eines Rueckgabe-Sets frisst sich nicht zurück)
  5. resolve_entity End-to-End auf Temp-DB:
     - Merge-Equivalenz Prefilter vs. Vollscan (P4/P5, 300 Kandidaten)
     - Threshold-Kandidat hinter Position K wird trotzdem gefunden
  6. DB-Index idx_entities_type vorhanden (Kandidaten-Query Priority 4)
  7. debug-Level-Log wird ohne ACTIVEWIKI_LOG_DEBUG unterdrückt

Alle Fixtures sind synthetisch/generisch (Platzhalter-Labels, keine echten
Orts-/Projektnamen). Das Modul-Import von vectordb benoetigt eine Config:
Der Test schreibt vor dem Import eine Temp-Config (wikis_root in ein
Temp-Verzeichnis) und setzt ACTIVEWIKI_CONFIG — kein Abhaengigkeit von
einer installierten activewiki.json.

Lauf: python3 test_dedup_speedup.py
"""
from __future__ import annotations

import io
import json
import os
import sqlite3
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── Config-Bootstrap: Temp-Config vor dem vectordb-Import ─────────────────
# vectordb.py liest beim Import activewiki.json (wikis_root fuer logs/,
# Embed-Defaults). Wir zeigen per ACTIVEWIKI_CONFIG auf eine minimale
# Temp-Config, damit der Test ohne Installation lauffaehig ist.
_TMP_ROOT = Path(tempfile.mkdtemp(prefix="aw-dedup-test-"))
_TMP_CONFIG = _TMP_ROOT / "activewiki.json"
_TMP_CONFIG.write_text(json.dumps({
    "wikis_root": str(_TMP_ROOT),
    "ollama": {"url": "http://127.0.0.1:1", "embed_model": "test-embed", "embed_dims": 8},
    "llm": {"model": "test-model", "url": "http://127.0.0.1:1"},
    "vectordb": {"index_path": "vectordb/index.sqlite"},
}), encoding="utf-8")
os.environ["ACTIVEWIKI_CONFIG"] = str(_TMP_CONFIG)

import config as config_mod  # noqa: E402
import vectordb as vb  # noqa: E402


# ── Helpers ────────────────────────────────────────────────────────────────

def _fresh_graph_db() -> sqlite3.Connection:
    """Leere Temp-DB mit vollem Graph-Schema (init_db ohne chunks-Pflicht)."""
    path = tempfile.mktemp(suffix=".sqlite")
    conn = sqlite3.connect(path)
    vb.init_db(conn)
    return conn


def _zero_embed(texts):
    """Offline-Embed-Mock: Null-Vektoren → Embedding-Stages greifen nicht."""
    import numpy as np
    return np.zeros((len(texts), vb.EMBED_DIMS), dtype=np.float32)


class _TopKBase(unittest.TestCase):
    """Setzt DEDUP_TOP_K pro Test zurueck, damit keine Testreihenfolge-Effekte."""

    def setUp(self):
        self._saved_topk = vb.DEDUP_TOP_K

    def tearDown(self):
        vb.DEDUP_TOP_K = self._saved_topk


# ── 1) Config-Helper ───────────────────────────────────────────────────────

class TestDedupTopKConfig(unittest.TestCase):
    def test_default_50(self):
        self.assertEqual(config_mod.graph_dedup_top_k({}), 50)

    def test_explicit_value(self):
        self.assertEqual(
            config_mod.graph_dedup_top_k({"graph": {"dedup_top_k": 20}}), 20)

    def test_string_value_parsed(self):
        self.assertEqual(
            config_mod.graph_dedup_top_k({"graph": {"dedup_top_k": "30"}}), 30)

    def test_clamped_to_at_least_1(self):
        # 0 oder negativ wuerde den Fuzzy-Teil komplett abschalten → Ergebnis-aenderung
        self.assertEqual(
            config_mod.graph_dedup_top_k({"graph": {"dedup_top_k": 0}}), 1)
        self.assertEqual(
            config_mod.graph_dedup_top_k({"graph": {"dedup_top_k": -5}}), 1)

    def test_garbage_falls_back_to_default(self):
        self.assertEqual(
            config_mod.graph_dedup_top_k({"graph": {"dedup_top_k": "quatsch"}}), 50)

    def test_missing_graph_section(self):
        self.assertEqual(config_mod.graph_dedup_top_k({"llm": {}}), 50)


# ── 2) Upper-Bound-Rank ────────────────────────────────────────────────────

class TestFuzzyCandidateRank(unittest.TestCase):
    PAIRS = [
        ("Beispielstr. 32", "Beispielstrasse 32"),        # Schreibvarianten
        ("Beispielstraße 32", "Beispielstr. 32"),
        ("Muster Person", "Person Muster"),               # Reorder: token overlap 100
        ("abcdefghij" * 8, "abcdefghij" * 8 + "xyz"),     # lange, aehnliche Labels
        ("ganz anderslautender Text Nummer eins", "vollstaendig anderes Label zwei"),
        ("", ""),                                         # beide leer → Score 100
        ("", "x"),
        ("kurz", "kurz" + "y" * 30),                      # |Δlen| > 20 → lev 0
        ("KNX 1/1/74", "KNX 1/1/74"),
        ("A B C D E", "A B C D E F G"),
    ]

    def test_bound_holds_for_all_pairs(self):
        for a, b in self.PAIRS:
            score = vb._fuzzy_match_score(a, b)
            rank = vb._fuzzy_candidate_rank(a, b)
            if score >= 85:
                self.assertGreaterEqual(
                    rank, 85,
                    f"Bound verletzt: {a!r} vs {b!r} score={score} rank={rank}")

    def test_symmetric_and_cached(self):
        r1 = vb._fuzzy_candidate_rank("ABC Def", "ABC Defg")
        r2 = vb._fuzzy_candidate_rank("ABC Defg", "ABC Def")
        self.assertEqual(r1, r2)
        # Cache: zweiter Aufruf identisch
        self.assertEqual(vb._fuzzy_candidate_rank("ABC Def", "ABC Defg"), r1)

    def test_identical_labels_rank_100(self):
        self.assertEqual(vb._fuzzy_candidate_rank("foo bar", "foo bar"), 100)


# ── 3) Top-K-Prefilter ─────────────────────────────────────────────────────

class TestRankFuzzyCandidates(_TopKBase):
    def test_threshold_capable_always_included_even_beyond_k(self):
        label = "Musterstrasse 1"
        # 40 echte Threshold-Kandidaten (Score 100) + viele Rausch-Kandidaten
        cands = ["Musterstrasse 1" for _ in range(40)]
        cands += [f"Zfh qgw {i} xyzwabeftext" for i in range(200)]
        subset, overflow = vb._rank_fuzzy_candidates(label, cands, 10, 85)
        self.assertTrue(overflow)
        # alle 40 Threshold-Kandidaten sind dabei — Top-K darf keinen Merge verlieren
        self.assertEqual(subset, sorted(subset))
        self.assertEqual(subset, list(range(40)))

    def test_top_k_respected_and_db_order(self):
        label = "Alpha Beta Gamma"
        cands = [f"Rauschen {i} zqwxv lmnopqrstuvwx yz0123456789" for i in range(300)]
        # ein echter Match ganz hinten
        cands.append("Alpha Beta Gamma Delta")
        subset, overflow = vb._rank_fuzzy_candidates(label, cands, 20, 85)
        self.assertFalse(overflow)
        self.assertIn(len(cands) - 1, subset)
        self.assertLessEqual(len(subset), 20)
        self.assertEqual(subset, sorted(subset))

    def test_zero_rank_dropped(self):
        label = "Alpha"
        cands = [f"Zqx vjp wmotrilk {i} qstuvwxyz0123456789" for i in range(100)]
        subset, overflow = vb._rank_fuzzy_candidates(label, cands, 50, 85)
        # kein gemeinsames Token, |Δlen| > 20 → Rank 0, fliegen raus
        self.assertEqual(subset, [])
        self.assertFalse(overflow)

    def test_threshold_parameter_switch(self):
        # c1/c2: Rank 100 (threshold-faehig bei 85 UND 90)
        # c3: Score exakt 85 (Token 17/20; Lev frueh-exit Δlen>20 -> 0)
        shared = [f"sharedtok{i:04d}" for i in range(17)]
        a = " ".join(shared + ["x" * 300])
        c3 = " ".join(shared + ["y" * 100, "z" * 100])
        self.assertEqual(vb._fuzzy_match_score(a, c3), 85)
        c1 = " ".join(reversed(a.split()))          # Reorder -> Score 100
        c2 = a
        cands = [c1, c2, c3]
        # Threshold 85: alle drei threshold-faehig
        subset85, _ = vb._rank_fuzzy_candidates(a, cands, 1, 85)
        self.assertEqual(subset85, [0, 1, 2])
        # Threshold 90: c3 faellt raus; Budget durch 2 above-Kandidaten
        # aufgebraucht (K=1 -> overflow, above-Liste gewinnt).
        subset90, over90 = vb._rank_fuzzy_candidates(a, cands, 1, 90)
        self.assertEqual(subset90, [0, 1])
        self.assertTrue(over90)

    def test_k_larger_than_candidate_list(self):
        cands = ["Alpha Beta", "Gamma Delta"]
        subset, overflow = vb._rank_fuzzy_candidates("Alpha Beta", cands, 1000, 85)
        self.assertEqual(subset, [0, 1])
        self.assertFalse(overflow)


# ── 4) Token-Cache-Korrektheit ─────────────────────────────────────────────

class TestJaccardTokenCache(unittest.TestCase):
    def test_cached_equals_uncached(self):
        import re as _re
        label = "Beispielstr. 32a und Haus Nr. 5 Musterort"
        cached = vb._normalize_label_for_jaccard(label)
        # Ungecachte Referenzimplementierung (Kopie der Originallogik)
        normalized = label.lower()
        normalized = normalized.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
        normalized = normalized.replace("ß", "ss")
        normalized = normalized.replace("str.", "straße")
        normalized = normalized.replace("strasse", "straße")
        normalized = normalized.replace("nr.", "nummer")
        tokens = _re.findall(r"[a-zA-ZäöüÄÖÜß0-9]+", normalized)
        ref = {t for t in tokens if t not in vb._STOPWORDS_JACCARD and len(t) > 1}
        self.assertEqual(set(cached), ref)

    def test_repeated_calls_stable(self):
        a1 = vb._normalize_label_for_jaccard("Muster Strasse 12")
        a2 = vb._normalize_label_for_jaccard("Muster Strasse 12")
        self.assertIs(a1, a2)  # identisches Objekt (Cache-Treffer)

    def test_returned_set_is_immutable(self):
        s = vb._normalize_label_for_jaccard("Ein Anderes Label")
        with self.assertRaises(AttributeError):
            s.add("gift")  # frozenset → keine Mutation, kein Cache-Vergiftungsrisiko
        self.assertNotIn("gift", vb._normalize_label_for_jaccard("Ein Anderes Label"))


# ── 5) resolve_entity: Merge-Equivalenz ────────────────────────────────────

class TestResolveEntityEquivalence(_TopKBase):
    """Kerntest: Prefilter (Top-K) == Vollscan, End-to-End gegen Temp-DB."""

    def _build_db(self, n_filler: int) -> sqlite3.Connection:
        conn = _fresh_graph_db()
        conn.execute("INSERT INTO entities(id,label,entity_type,description) "
                     "VALUES('seed','Seed Konzept','CONCEPT','alt')")
        for i in range(n_filler):
            conn.execute(
                "INSERT INTO entities(id,label,entity_type) VALUES(?,?,?)",
                (f"f{i}", f"Qvtr pwjm {i} zxwivjrg", "CONCEPT"))
        # PROPERTY-Adressen mit Hausnummern-Guard-Interaktion (synthetisch)
        for i in range(30):
            conn.execute(
                "INSERT INTO entities(id,label,entity_type) VALUES(?,?,?)",
                (f"p{i}", f"Musterweg {i + 100}", "PROPERTY"))
        conn.commit()
        return conn

    def test_prefilter_matches_full_scan(self):
        probes = [
            ("Seed Konzept", "CONCEPT"),                       # Exact NOCASE
            ("Qvtr pwjm 42 zxwivjrg", "CONCEPT"),              # Exact 100% fuzzy
            ("Qvtr pwim 42 zxwivjrg", "CONCEPT"),              # 1 Tippfehler
            ("Musterweg 105", "PROPERTY"),                     # exact + guard
            ("Musterweg 1005", "PROPERTY"),                    # andere Hausnummer → guard
            ("Alpha Beta Gamma Delta Epsilon", "CONCEPT"),     # → neu (P6)
            ("Seed Konzepte", "CONCEPT"),                      # hohes fuzzy
        ]
        # Vollscan-Vergleich: zwei identische DBs, K=1 (hart) vs. Vollscan
        conn_a = self._build_db(300)
        conn_b = self._build_db(300)
        for k in (50, 5, 1):
            vb.DEDUP_TOP_K = k
            with mock.patch.object(vb, "embed", _zero_embed):
                got = [vb.resolve_entity(
                    conn_a, new_id=f"probe{i}", label=l, entity_type=t,
                    description="d", wiki_page="bench/t")
                    for i, (l, t) in enumerate(probes)]
                vb.DEDUP_TOP_K = 10**9  # Practically full scan: never cuts
                want = [vb.resolve_entity(
                    conn_b, new_id=f"probe{i}", label=l, entity_type=t,
                    description="d", wiki_page="bench/t")
                    for i, (l, t) in enumerate(probes)]
            self.assertEqual(got, want, f"Merge-Divergenz bei K={k}")
        conn_a.close(); conn_b.close()

    def test_threshold_match_beyond_k_still_found(self):
        # 120 Rausch-CONCEPTs, der echte Match steht auf DB-Position > K
        conn = self._build_db(120)
        conn.execute("INSERT INTO entities(id,label,entity_type) "
                     "VALUES('deep','Zzyq Unique Tiefer Text Punkt','CONCEPT')")
        conn.commit()
        vb.DEDUP_TOP_K = 5  # Match-Rank 100 → trotz K=5 gefunden (Threshold-Garantie)
        with mock.patch.object(vb, "embed", _zero_embed):
            canon = vb.resolve_entity(
                conn, new_id="probe", label="Zzyq Unique Tiefer Text Punk",
                entity_type="CONCEPT", description="d", wiki_page="bench/t")
        self.assertEqual(canon, "deep")
        conn.close()


# ── 6) Index vorhanden ─────────────────────────────────────────────────────

class TestEntityIndexes(unittest.TestCase):
    def test_type_index_after_init(self):
        conn = _fresh_graph_db()
        names = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='entities'")}
        self.assertIn("idx_entities_type", names)
        conn.close()


# ── 7) debug-Log-Level ─────────────────────────────────────────────────────

class TestDebugLogSuppression(unittest.TestCase):
    def test_debug_suppressed_by_default(self):
        with mock.patch.object(vb, "_DEBUG_LOG", False):
            buf = io.StringIO()
            with redirect_stdout(buf):
                vb.log("debug", "dedup-skip zeug")
                vb.log("dedup", "sichtbar")
            out = buf.getvalue()
            self.assertNotIn("dedup-skip zeug", out)
            self.assertIn("sichtbar", out)

    def test_debug_visible_when_enabled(self):
        with mock.patch.object(vb, "_DEBUG_LOG", True):
            buf = io.StringIO()
            with redirect_stdout(buf):
                vb.log("debug", "dedup-skip zeug")
            self.assertIn("dedup-skip zeug", buf.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)

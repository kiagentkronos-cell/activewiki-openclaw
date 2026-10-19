#!/usr/bin/env python3
"""Tests für die rollenbezogene Modell-/Thinking-Config (seit 2026-10-18).

Abdeckung:
  1. Fallback-Ketten: build_model/qc_model → llm.model (fehlender Rollen-Key)
  2. Rollen-Keys setzen Modell + thinking pro Rolle
  3. Thinking-Defaults: fehlender Key → False (heutiges Verhalten)
  4. Typ-Robustheit: String "true"/"false" ok, Müll → Default + Warnung,
     kein Crash
  5. check.default_llm_call: Payload-Verdrahtung (Modell + enable_thinking)
     mit gemocktem Config-Load und gemocktem urllib — kein echter LLM-Call.

Lauf: python3 test_role_models.py
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
import urllib.request
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as config_mod  # noqa: E402
import check as check_mod  # noqa: E402


class TestBuildModelFallback(unittest.TestCase):
    def test_only_llm_model(self):
        cfg = {"llm": {"model": "m-base"}}
        self.assertEqual(config_mod.build_model(cfg), "m-base")

    def test_build_model_overrides(self):
        cfg = {"llm": {"model": "m-base", "build_model": "m-build"}}
        self.assertEqual(config_mod.build_model(cfg), "m-build")

    def test_empty_or_blank_build_model_falls_back(self):
        for junk in ("", "   ", None):
            cfg = {"llm": {"model": "m-base", "build_model": junk}}
            with redirect_stderr(io.StringIO()):
                self.assertEqual(config_mod.build_model(cfg), "m-base")

    def test_non_string_build_model_falls_back_with_warning(self):
        cfg = {"llm": {"model": "m-base", "build_model": 42}}
        with redirect_stderr(io.StringIO()) as err:
            self.assertEqual(config_mod.build_model(cfg), "m-base")
        self.assertIn("WARNING", err.getvalue())

    def test_no_model_at_all_is_fatal(self):
        # Config-Pflicht wie bisher: kein Modell-Key anywhere = SystemExit
        with self.assertRaises(SystemExit):
            config_mod.build_model({"llm": {}})


class TestQcModelFallback(unittest.TestCase):
    def test_only_llm_model(self):
        cfg = {"llm": {"model": "m-base"}}
        self.assertEqual(config_mod.qc_model(cfg), "m-base")

    def test_quality_check_model_overrides(self):
        cfg = {"llm": {"model": "m-base"},
               "quality_check": {"model": "m-qc"}}
        self.assertEqual(config_mod.qc_model(cfg), "m-qc")

    def test_non_string_falls_back_with_warning(self):
        cfg = {"llm": {"model": "m-base"},
               "quality_check": {"model": ["m-qc"]}}
        with redirect_stderr(io.StringIO()) as err:
            self.assertEqual(config_mod.qc_model(cfg), "m-base")
        self.assertIn("WARNING", err.getvalue())


class TestThinkingDefaults(unittest.TestCase):
    def test_missing_keys_default_false(self):
        cfg = {"llm": {"model": "m-base"}}
        self.assertFalse(config_mod.build_thinking(cfg))
        self.assertFalse(config_mod.qc_thinking(cfg))

    def test_true_values(self):
        cfg = {"llm": {"model": "m", "build_thinking": True},
               "quality_check": {"thinking": True}}
        self.assertTrue(config_mod.build_thinking(cfg))
        self.assertTrue(config_mod.qc_thinking(cfg))

    def test_string_true_false(self):
        for raw, want in (("true", True), ("TRUE", True), (" true ", True),
                          ("false", False), ("False", False),
                          ("1", True), ("0", False)):
            cfg = {"llm": {"model": "m", "build_thinking": raw}}
            with redirect_stderr(io.StringIO()) as err:
                self.assertEqual(config_mod.build_thinking(cfg), want,
                                 msg=f"raw={raw!r}")
            self.assertEqual(err.getvalue(), "",
                             msg=f"String {raw!r} darf keine Warnung auslösen")

    def test_junk_types_default_with_warning(self):
        for junk in ("vielleicht", [True], {"x": 1}, -1.5):
            cfg = {"llm": {"model": "m", "build_thinking": junk}}
            with redirect_stderr(io.StringIO()) as err:
                self.assertFalse(config_mod.build_thinking(cfg),
                                 msg=f"junk={junk!r}")
            self.assertIn("WARNING", err.getvalue(), msg=f"junk={junk!r}")

    def test_int_zero_and_one(self):
        self.assertFalse(config_mod.build_thinking(
            {"llm": {"model": "m", "build_thinking": 0}}))
        self.assertTrue(config_mod.build_thinking(
            {"llm": {"model": "m", "build_thinking": 1}}))


class TestCheckPayloadWiring(unittest.TestCase):
    """default_llm_call: Modell + enable_thinking kommen aus der Config."""

    class _FakeResp:
        def __init__(self):
            body = json.dumps({"choices": [{"message": {"content": "{}"}}]})
            self._data = body.encode("utf-8")

        def read(self):
            return self._data

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _run(self, cfg: dict) -> dict:
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["payload"] = json.loads(req.data.decode("utf-8"))
            return self._FakeResp()

        with tempfile.TemporaryDirectory() as td:
            cfg_file = Path(td) / "activewiki.json"
            cfg_file.write_text(json.dumps(cfg), encoding="utf-8")
            old = os.environ.get("ACTIVEWIKI_CONFIG")
            os.environ["ACTIVEWIKI_CONFIG"] = str(cfg_file)
            try:
                with mock.patch.object(urllib.request, "urlopen", fake_urlopen):
                    out = check_mod.default_llm_call("sys", "user")
            finally:
                if old is None:
                    os.environ.pop("ACTIVEWIKI_CONFIG", None)
                else:
                    os.environ["ACTIVEWIKI_CONFIG"] = old
            self.assertEqual(out, "{}")
        return captured["payload"]

    def test_defaults_old_config(self):
        # Alte Config ohne neue Keys = exakt heutiges Verhalten
        p = self._run({"llm": {"model": "m-base", "temperature": 0.5}})
        self.assertEqual(p["model"], "m-base")
        self.assertIs(p["chat_template_kwargs"]["enable_thinking"], False)

    def test_qc_model_and_thinking(self):
        p = self._run({"llm": {"model": "m-base"},
                       "quality_check": {"model": "m-qc", "thinking": True}})
        self.assertEqual(p["model"], "m-qc")
        self.assertIs(p["chat_template_kwargs"]["enable_thinking"], True)

    def test_content_none_raises_without_inventing(self):
        # content=None (thinking-Trunkierung) → RuntimeError, kein Retext
        class NoneResp(TestCheckPayloadWiring._FakeResp):
            def read(self):
                return json.dumps(
                    {"choices": [{"message": {"content": None,
                                             "reasoning": "denkt…"}}]}
                ).encode("utf-8")

        def fake_urlopen(req, timeout=None):
            return NoneResp()

        with tempfile.TemporaryDirectory() as td:
            cfg_file = Path(td) / "activewiki.json"
            cfg_file.write_text(json.dumps({"llm": {"model": "m"}}),
                                encoding="utf-8")
            old = os.environ.get("ACTIVEWIKI_CONFIG")
            os.environ["ACTIVEWIKI_CONFIG"] = str(cfg_file)
            try:
                with mock.patch.object(urllib.request, "urlopen", fake_urlopen):
                    with self.assertRaises(RuntimeError) as cm:
                        check_mod.default_llm_call("sys", "user")
            finally:
                if old is None:
                    os.environ.pop("ACTIVEWIKI_CONFIG", None)
                else:
                    os.environ["ACTIVEWIKI_CONFIG"] = old
        self.assertIn("content=null", str(cm.exception))

    def test_llm_model_name_uses_qc_role(self):
        with tempfile.TemporaryDirectory() as td:
            cfg_file = Path(td) / "activewiki.json"
            cfg_file.write_text(json.dumps(
                {"llm": {"model": "m-base"},
                 "quality_check": {"model": "m-qc"}}), encoding="utf-8")
            old = os.environ.get("ACTIVEWIKI_CONFIG")
            os.environ["ACTIVEWIKI_CONFIG"] = str(cfg_file)
            try:
                self.assertEqual(check_mod._llm_model_name(), "m-qc")
            finally:
                if old is None:
                    os.environ.pop("ACTIVEWIKI_CONFIG", None)
                else:
                    os.environ["ACTIVEWIKI_CONFIG"] = old


class TestDistillWiring(unittest.TestCase):
    """distill.py ist nachweislich an die Rollen-Accessoren verdrahtet
    (quellen-basierte Checks, Muster aus dem Live-Repo test_distill_gate.py —
    die Kopie führt diese Suite mit)."""

    SRC = (Path(__file__).resolve().parent / "distill.py").read_text(encoding="utf-8")

    def test_model_wired_to_build_model(self):
        self.assertIn('MODEL = os.environ.get("ACTIVEWIKI_MODEL", build_model(_CONFIG))', self.SRC)

    def test_thinking_constant_wired_to_accessor(self):
        self.assertIn("BUILD_THINKING = build_thinking(_CONFIG)", self.SRC)
        self.assertIn('"chat_template_kwargs": {"enable_thinking": BUILD_THINKING}', self.SRC)

    def test_no_hardcoded_thinking_false_left(self):
        # Kein hartes False mehr im call_llm-Payload (Kommentarzeilen egal)
        self.assertNotIn('"chat_template_kwargs": {"enable_thinking": False}', self.SRC)


class TestThinkingJsonPrefixTolerance(unittest.TestCase):
    """QC-JSON-Parsing toleriert Präfix-Text (z.B. Denkreste) vor dem
    JSON-Objekt — parse_llm_answer scannt den ersten balancierten Block."""

    def test_prefix_before_json_parses(self):
        parsed, raw, why = check_mod.parse_llm_answer(
            'Ich denke kurz nach.\n{"status": "clean", "issues": []}')
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["status"], "clean")
        self.assertEqual(why, "")

    def test_fence_after_prefix_parses(self):
        parsed, _, why = check_mod.parse_llm_answer(
            'Antwort:\n```json\n{"status": "issues", "issues": []}\n```')
        self.assertIsNotNone(parsed)
        self.assertEqual(why, "")


if __name__ == "__main__":
    unittest.main(verbosity=2)

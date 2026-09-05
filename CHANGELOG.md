# Changelog

All notable changes to the ActiveWiki project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.3] - 2026-09-05

### Added
- Configurable `max_tokens` for the nightly quality check
  (`scripts/check.py`): new `resolve_max_tokens()` resolves the per-call
  output budget as `ACTIVEWIKI_CHECK_MAX_TOKENS` env var >
  `quality_check.max_tokens` in `activewiki.json` > script default 4096.
  Values above the hard cap 32768 are clamped with a warning (protects the
  model context window); garbage values (non-numeric, 0, negative) are
  ignored so the next priority level applies. A missing config key never
  fails fast. `activewiki.example.json` documents the new key.

### Changed
- `parse_llm_answer()` hardened: now returns `(parsed, raw, grund)` and
  distinguishes failure causes — `""` on success, `abgeschnitten
  (max_tokens zu klein?)` on truncated (unbalanced) JSON, `kein JSON` on
  prose. Parsing uses a balanced-brace scanner that respects string
  escapes (instead of blind first-`{`-to-last-`}`) and strips Markdown
  code fences at the edges. On truncation, the report's error issue names
  the reason and `fix_hint` suggests raising `quality_check.max_tokens`.
- 13 new tests in `test_check.py` (103 total): max_tokens resolution
  (Env > Config > Default, cap, garbage handling) and parser hardening
  (fences, truncation, prose, nested braces in strings).

## [1.1.2] - 2026-09-03

### Added
- Configurable LLM timeout for the nightly quality check (`scripts/check.py`):
  new `resolve_llm_timeout()` resolves the per-call timeout as
  `ACTIVEWIKI_CHECK_TIMEOUT` env var > `quality_check.timeout_seconds` in
  `activewiki.json` > script default 120s. A missing config key never fails
  fast (falls back to the 120s default); invalid/unreadable config is ignored
  with the default retained. Retry logic unchanged (one retry after ~10s
  pause, timeouts only). `activewiki.example.json` documents the new key with
  600s; 4 new resolution tests in `test_check.py` (90 total).

## [1.1.1] - 2026-09-01

### Performance
- mmap search cache for `vectordb.py search` (`vectordb/search_cache/`,
  gitignored): embedding matrix as `vecs.npy` loaded via `np.load` mmap plus
  metadata as `meta.json` (id/scope/kind/ref/section/chunk_idx only — no
  content, no pickle). Staleness via DB `mtime_ns`+size signature, with a
  transparent one-shot rebuild (~20s) on the first search after an index
  change; `build` refreshes the cache after each run. Atomic publish via
  pid-suffixed temp files, `os.replace`, and fsync; cache directory 0700,
  files 0600. Result content is fetched lazily from SQLite for the final
  top-k only — the cache is never trusted for output content. Plugin search
  path: ~18s → ~1s, keeping `memory_search` safely under the 15s core limit.

## [1.1.0] - 2026-08-26

### Added
- Nightly wiki quality check (`scripts/check.py`) — fact verification of wiki
  pages against their sources with a deterministic table diff,
  OCR-decimal normalization (`199 34` ↔ `199,34`), and a self-contradiction
  filter; inline auto-patch in a single LLM call (fixes exactly evidenced by
  sources, applied via an exact-once-occurrence rule); frontmatter tracking
  (`check_status`, `last_check`, `last_check_model`); 2-commit pattern
  (`quality-check: <slug> check` + `quality-check: <slug> patch`); JSON
  reports under `quality/results/`.
- QC phase in `scripts/run_inbox.sh` — runs after distill, before vectordb
  (patches are indexed in the same night run); gated on empty inbox and a
  02:30 deadline; auto-commits uncommitted build changes before QC.
- `scripts/test_check.py` — 86 unit tests for the quality check (self-contained,
  mocked LLM).

## [1.0.18] - 2026-08-18

### Changed
- Model migration: `qwen3.6-fp8` → `qwen3.8-fp8-fast`

### Fixed
- `rel_id` schema bug in graph orphan cleanup

### Removed
- LLM model code-fallback — the model is now mandatory in the config
  (fail-fast instead of a hardcoded default)

### Security
- Full-history PII anonymization of the public repository (force-push)
- `scopes.example.json`: exact-identity security guidance (exact session-key
  patterns instead of `:discord:` wildcards)

## [1.0.17] and earlier

- (no changelog kept before v1.0.18)

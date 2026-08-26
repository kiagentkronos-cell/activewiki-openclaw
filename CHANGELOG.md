# Changelog

All notable changes to the ActiveWiki project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

# Changelog

All notable changes to the ActiveWiki project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.6] - 2026-09-13

### Added
- **3D graph viewer**: `assets/template3d.html` renders the knowledge graph
  spatially with `3d-force-graph` (fixed nodes, drag-vs-click gating,
  screen-space overlay labels with a distance gate and de-clutter pass,
  edge schema/legend identical to the 2D view, WebGL fallback panel).
- **`graph.html_mode` config key** (`"2d" | "3d" | "both"`, default `"3d"`)
  with a `--mode` CLI flag on `vectordb.py graph html` (CLI > config >
  default; unknown values warn and fall back). `config.graph_html_mode()`
  normalizes the value; `run_inbox.sh` passes the configured mode through
  to the nightly `graph-html` phase.
- **`both` mode**: one file, one shared data blob, a 2D|3D toggle in the
  header. No-CDN requirement: `d3` and the 3D library are inlined into the
  generated HTML (`assets/d3.v7.min.js`,
  `assets/vendor/3d-force-graph.min.js`), so the exported file works on a
  LAN without internet and `serve/` stays a single-file symlink.

### Fixed
- **3D layer was flat in `both` mode**: the 2D viewer boots first and
  d3-force writes `x`/`y` into the shared node objects, which d3-force-3d
  inherits while leaving `z` at 0. The 3D viewer now builds flat node
  copies per viewer in `both` mode (3d-only keeps the shared reference).
- **Toggle invisible in the 2D view**: `#header` is a fixed flex row
  without wrapping, so a toggle appended at the end slid into overflow.
  It is injected as the first header child with `flex-shrink: 0`
  (`setMode` is additionally exposed as `window.setMode`).
- **2D zoom-out limit**: `scaleExtent` `[0.15, 6]` → `[0.01, 8]`; the fit
  algorithm clamps correctly against the new limit.

### Changed
- **2D simulation convergence stop**: `alphaMin(0.02)` +
  `velocityDecay(0.55)` + `alphaDecay` start value 0.04. d3 used to stop
  only at alpha 0.001 (~300 ticks) with visible micro-jitter; now ~66
  ticks and sub-pixel residual energy. Link/label attribute updates run
  only while `alpha > 0.005` or a drag is active (node transforms stay
  per-tick to protect against drag freeze); highlight and zoom/pan are
  tick-independent and keep working after the simulation stops.

## [1.1.5] - 2026-09-10

### Added
- **Truncation-artifact filter** in the nightly quality check
  (`scripts/check.py`): sources above `SOURCE_FULL_MAX` (30 KB) are trimmed
  for the LLM prompt (head + middle excerpt + tail); the LLM used to report
  contradictions whose evidence might simply sit in the cut-out middle. The
  user prompt now carries an explicit prohibition (`VERBOT`) for exactly that
  case, and a deterministic post-filter
  (`filter_truncation_artifacts()`) discards any remaining LLM issue with
  `origin: llm` + `severity: contradiction` whose evidence/fix_hint
  references the truncation. Matching covers the literal markers plus the
  word family `gekürzt`/`gekuerzt`/`Kürzung`/`Kuerzung` via regex with
  word-boundary lookbehind (`Abkürzung`/`Verkürzung` never trigger — they
  can legitimately appear in evidence; `abgekürzt` still matches).
  Deterministic issues and other severities are never suppressed.
  Suppressed issues are counted in the new report field
  `suppressed_truncation`.
- **Unpatchable-issue backoff**: pages whose issues never yield an applied
  fix (all `fix: null`) were re-checked every night forever. After 2
  consecutive rounds with an identical issue fingerprint
  (`issue_fingerprint()` — SHA-256 over the sorted `claim_in_page` set plus
  page body, order- and duplicate-independent), the page now gets a 30-day
  grace period until `issue_until` (analog `RECHECK_DAYS`) and is excluded
  from candidate selection. Any content change (different fingerprint) or
  applied patch resets the counter. New frontmatter fields:
  `issue_fingerprint`, `issue_runden`, `issue_until`; `qc_process_page()`
  returns the state as `backoff`.

### Tests
- 18 new tests in `test_check.py` (121 total): filter triage (origin/
  severity/marker matrix), paraphrase cases, word-boundary negatives
  (`Abkürzung`/`Verkürzung` survive, `abgekürzt` matches), prompt
  prohibition assertions, fingerprint stability, `backoff_state()` round
  arithmetic, grace-period exclusion in `select_candidates()`,
  frontmatter write/reset, and E2E two-rounds-then-grace over
  `qc_process_page()` (mocks only, no real LLM).

## [1.1.4] - 2026-09-07

### Added
- **Daily graph HTML in the nightly pipeline**: new `graph-html` phase in
  `scripts/run_inbox.sh` (after vectordb) runs `vectordb.py graph html --all`
  and publishes the result atomically (temp file + `mv` inside `output/`) to
  `output/graph_latest.html`. Budget-gated via `ingest.budget_seconds`,
  hard-capped via new config key `graph-html.timeout_seconds` (default 1200s),
  runs past deadline, fail-safe: a failure of this phase never aborts the run.
- **`deploy/wikis-graph.service`**: systemd user unit serving a dedicated
  `serve/` directory (symlink to `output/graph_latest.html`) on port 8090.
  Private artifacts in `output/` stay deliberately unserved. Paths are
  placeholders — adjust to your installation.
- **Home-screen app icon**: `assets/template.html` now embeds favicon (32x32)
  and apple-touch-icon (180x180) inline as base64 plus `theme-color` and
  web-app meta tags; the PNG sources live under `assets/`.

### Fixed
- `forceManyBody` in `assets/template.html` runs without `distanceMax`/`theta`
  overrides — the force layout now matches the reference layout from 07/23.

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

# ActiveWiki — Give Your Agent Access to Your Knowledge Base

**Your OpenClaw agent remembers conversations — but not your documents.**  
ActiveWiki changes that. It plugs your entire wiki (notes, manuals, research archives) into OpenClaw's Active Memory pipeline, so your agent automatically searches and cites them in every response.

**Semantic vector search** finds relevant passages even when wording differs. A **knowledge graph** layers on relationships between entities — so the agent doesn't just find text, it understands context.

---

## The Problem

Without ActiveWiki, an agent knows what you discussed last week — but nothing from the 200 PDFs and notes sitting in your wiki. You'd have to manually feed context into every conversation.

**With ActiveWiki:** Every `memory_search` automatically queries:
- ✅ Session history (built-in)
- ✅ Your wiki — semantic vector search across all documents
- ✅ Knowledge graph — entities, relationships, communities

Zero manual prompting required.

## What You Need

- An OpenClaw instance with a wiki at `/path/to/wikis/`
- Embedding backend (Ollama with `bge-m3` recommended) + any LLM with an OpenAI-compatible API (vLLM, Ollama, OpenAI, etc.)
- Python 3.x with numpy + [Docling](https://github.com/DS4SD/docling) (for PDF/DOCX ingestion)

## Quick Start

```bash
# 1. Configure
cp activewiki.example.json activewiki.json   # set paths, models, scopes
cp scopes.example.json <wikis>/config/       # define private/family/public rules

# 2. Install plugin
cd plugin && npm install && npm run build
openclaw plugins install --link .

# 3. Ingest documents
../scripts/run_inbox.sh
```

Done. Your agent now searches your wiki in every response.

## Features

### Semantic Vector Search
- Embed all wiki documents using Ollama (bge-m3)
- Automatic chunking with Docling for PDFs and DOCX files
- Over-fetching with relevance scoring for optimal context window usage
- **Search cache** (`vectordb/search_cache/`, gitignored): mmap-able `vecs.npy` + `meta.json` (id/scope/kind/ref/section/chunk_idx only — **no content**, **no pickle**). Invalidated via DB mtime_ns+size (`_db_sig()`), rebuilt transparently on the first `search` after an index change (~20s once). Atomic publish via pid-suffixed temp files + `os.replace` + fsync; directory 0700, files 0600. Result content is always loaded lazily from SQLite for the final top-k hits — the cache is never trusted for output content. Effect: plugin search ~18s → ~1s. To drop the cache: delete the folder, it rebuilds itself.

### Knowledge Graph
- **Entity extraction** from documents via configurable LLM (OpenAI-compatible API) — automatically identifies people, places, organizations, and concepts
- **Relationship mapping** — entities are connected by typed, directed relationships (has_parent_object, part_of_project, owned_by, etc.)
- **Confidence tags** — every relationship carries a confidence level (`extracted`, `inferred`, `weak`) reflecting how reliably it was extracted. Visualized with color-coded links in the D3.js graph (green/yellow/gray)
- **Rationale entities** — the "why" behind connections is preserved as traceable rationale nodes, rendered as distinctive blue diamond shapes in the D3.js visualization
- **Community detection** — Leiden algorithm identifies clusters of related entities (e.g., "Real Estate Investment cluster", "Home Automation cluster")
- **Interactive HTML visualization** — explore the graph visually with D3.js: filter by type, search entities, adjust zoom, export as PNG/SVG
- **Community sidebar** — clickable legend highlights entities by detected community cluster without forcing layout changes
- **Live D3 sliders** — 6 real-time controls for graph layout: Repulsion, Link Distance, Node Size, Collision, Center Force, Alpha Decay
- **Semantic Entity Search** — `graph search` uses embedding-based seed discovery: query is embedded via Ollama (bge-m3), cosine similarity computed against all entity embeddings, top-10 seeds selected. **Entity-type-specific thresholds** prevent false positives: PERSON ≥ 0.80 (strict, prevents name false-positives), ORGANIZATION ≥ 0.75, CONCEPT ≥ 0.50 (broad for diffuse concepts), DOCUMENT ≥ 0.45, default 0.50. LIKE search remains as fallback when no embedding match. New entities get embeddings automatically during `graph build`; backfill existing entities with `graph embed-missing`.
- **Multi-hop BFS Expansion (Phase B)** — seeds are expanded over multiple graph hops via breadth-first search (`--max-hops 1-5`, default 1). A visited-set prevents cycles. `--max-results` (default 200, 0 = unlimited) prevents result explosion.
- **Path Scoring + Evidence Assembly (Phase C)** — relevance score combines seed similarity, hop decay, and confidence: `score = seed_similarity × 0.7^hop × confidence_weight`. The full path from seed to result is reconstructed, with relationships along the path displayed as evidence rationale.
- **God Nodes CLI** — identify the most connected entities: `vectordb.py graph god-nodes --top 10`

### Entity Resolution (Deduplication)
- **Hybrid matching** — cosine similarity (embeddings) + Jaccard string similarity (token overlap) catches duplicates that pure semantic similarity misses
- **Prevention** — new entities from LLM extraction are checked against existing ones before insertion (soft-delete with `valid_until`)
- **Batch deduplication** — retroactively clean existing graphs: `vectordb.py graph deduplicate`
- **Dry-run mode** — preview merges before applying: `--dry-run` flag shows what would be merged without changing anything
- **Transitive clustering** — Union-Find algorithm ensures A→B→C chains are resolved into single canonical entities

### Prompt Evolution Loop
- **Automatic failure detection** — `graph validate` finds dangling links, over-merged entities, orphaned entities, confidence imbalance, and missing relation coverage
- **LLM root-cause diagnosis** — `graph diagnose` and `graph evolve` analyze why extractions failed and draft concrete rules
- **Rule queuing with dedup** — similar rules are merged via embedding cosine similarity; new rules enter an approval queue
- **Human-in-the-loop (HITL)** — Every new rule requires human approval via Discord before it touches the prompt
- **Versioned prompt updates** — `graph apply-prompt` inserts approved rules with `[AUTO]` markers; old prompts are backed up
- **Degradation detection** — `graph metrics` and `graph degradation-check` compare failure rates before/after rule activation; bad rules get deprecated automatically
- **Spiral protection** — `graph spiral-protection` halts the entire loop if ≥3 rules degrade within a month

```
Failure Detect → Diagnosis → Rule Queue → Human Review → Prompt Update → Degradation Check
     ▲                                                                         │
     └──────────────────────────────────────────────────────────────────────────┘
```

New CLI commands: `graph validate`, `graph diagnose`, `graph evolve`, `graph apply-prompt`, `graph metrics`, `graph degradation-check`, `graph spiral-protection`, `graph prompt-history`, `graph prompt-backup`.

See [`plugin/README.md`](plugin/README.md) for the full Prompt Evolution Pipeline documentation.

### Source Provenance (Phase G)
- **Unified section extraction** — entities and relationships are extracted together per `##` section (~2-5K chars) via a relationship-first prompt, eliminating isolated nodes by design
- **Page summary pass** — after section extraction, an additional LLM pass checks for missing cross-section relationships using the page structure and entity list as context
- **Click-to-source** — every relationship stores the originating section header and source text snippet (max 200 chars), enabling traceability in the D3.js visualization
- **Entity-chunk mapping** — N:M junction table (`entity_chunks`) tracks which document chunks produced each entity, with source text excerpts
- **Discarded relations tracking** — relationships referencing unknown entities are stored in `discarded_relations` instead of creating stubs, preserving audit trail
- **Entity resolution** — hybrid matching (embeddings + string similarity + Union-Find deduplication) with label-based cross-page lookup for robust entity identity
- **Large section handling** — sections exceeding 15K chars are automatically split by `###` headings or paragraphs to stay within context limits

### Quality Check (Nightly Fact-Verification)
- **What it does** — `scripts/check.py` verifies the oldest wiki page against its full-text sources and **auto-repairs what is unambiguously evidenced** (inline patch in the same LLM call). Runs as a QC phase in the nightly pipeline: after distill, before vectordb — so patched pages are re-indexed in the same night run.
- **Candidate selection** — `--oldest N` picks the N leaf pages with the oldest `updated` date (rollup rule: pages with children are skipped until their children have been checked). Pages with `check_status: clean|patched` and `last_check` ≤ 30 days are excluded.
- **Two-layer verification** — one LLM call (page body + source excerpts) for prose facts, plus a deterministic table diff: every number in a wiki markdown table must appear in at least one source (number-normalization: comma↔dot decimals, thousands separators, years/URLs ignored; OCR-decimal normalization `199 34` ↔ `199,34`). OCR variants are only ever *match hints* — they never directly trigger a patch.
- **Self-contradiction filter** — LLM issues whose own evidence/fix_hint retracts the finding ("no error", "is correct", …) are discarded and only listed in the report's `filtered_self_contradictions` field; they don't count as issues.
- **Inline auto-patch** — per issue the LLM additionally returns `fix: {original, corrected}` (only when exactly supported by the loaded source evidence, else `null`). The code evaluates this deterministically and applies valid fixes via an exact-once-occurrence rule (`original` must appear exactly once in the page body — no paraphrases, no fuzzy matching). Deterministic table-diff issues are never patched.
- **Frontmatter tracking** — `check_status` (`clean|issues|patched|error|uncheckable`), `last_check`, `last_check_model` are maintained per page; clean/patched pages are not re-checked for 30 days.
- **2-commit pattern** — a successful body patch produces two git commits: `quality-check: <slug> check` (frontmatter `issues`, body unpatched) followed by `quality-check: <slug> patch` (body patched, `check_status: patched`). Without a valid patch, a single `check` commit is written. Commits are skipped on a dirty repo (safety guard).
- **Reports** — one JSON report per page under `quality/results/` with `issues`, `filtered_self_contradictions`, and `patch: {patched, applied, skipped[claim+reason]}`.
- **Configurable LLM timeout** — the per-call LLM timeout resolves in priority order: `ACTIVEWIKI_CHECK_TIMEOUT` env var > `quality_check.timeout_seconds` in `activewiki.json` > script default 120s. A missing config key never fails fast (the check falls back to 120s); the example config ships 600s for slow/large-model setups. Retry logic unchanged: one retry after ~10s pause on timeout only.

CLI:
```bash
scripts/check.py --oldest 2                    # 2 oldest leaf pages (what the nightly pipeline runs)
scripts/check.py --page wiki/<scope>/<slug>.md # single page (check + auto-patch)
scripts/check.py --all --since-days 7 --limit 20  # read-only (check only, no patches)
```

### Temporal Filtering
- **Time-bound relationship queries** — `--since YYYY-MM-DD` and `--until YYYY-MM-DD` filter relationships by validity period
- **NULL-safe semantics** — timeless relationships (no date) always pass filters
- **Active vs. historical** — distinguish current relationships from past ones in graph search output

### Active Memory Integration
- OpenClaw plugin automatically injects wiki hits into every LLM response
- Scope-gated results (private/family/public) based on user authorization
- Hybrid search: vector hits bridge to graph entities for combined context

## Pipeline

`run_inbox.sh` is the master pipeline for the nightly run. Each phase is deadline-aware (default 03:00) and logged to `logs/ingest-<date>.log`:

1. **Ingest** (`ingest.py`) — OCR new inbox documents into `sources/` (Docling; PDF, images, DOCX)
2. **Distill** (`distill.py`) — generate/update hierarchical wiki pages from sources + bottom-up rollup of the folder hierarchy
3. **Auto-commit** — uncommitted build changes are committed so the repo is clean for QC commits
4. **Quality Check** (`check.py --oldest 2`) — nightly fact-verification of the 2 oldest leaf pages vs. their sources (see Quality Check above). Only runs when the inbox is empty and before 02:30, i.e. *after* distill and *before* vectordb — patches are indexed in the same night run
5. **Vectordb** (`vectordb.py build --graph-incremental --communities --communities-incremental`) — always runs (content-hash-based incremental), even past the deadline

## Full Documentation

See [`plugin/README.md`](plugin/README.md) for complete reference documentation including all configuration options, pipeline phases, installation steps, troubleshooting, and FAQ.

## License

MIT — See [`LICENSE`](LICENSE) for details.

# ActiveWiki — OpenClaw Wiki Integration

**Status:** Stable/Release | **Last Changed:** 2026-05-30 | **Author:** Max Mustermann

---

## 1. Overview

### What

ActiveWiki is an OpenClaw `MemoryCorpusSupplement` plugin that automatically integrates a wiki vector database and a knowledge graph into OpenClaw's Active Memory retrieval pipeline.

### Why

Before the plugin, the Active Memory subagent could only search its own `memory/` files and session histories. Wiki content (private, family, public) was only manually accessible via `memory_search corpus=wiki` — not automatically injected with every response.

**Now:** Every `memory_search` query automatically searches session history + wiki vector database + knowledge graph. Results are merged, sorted by score, and injected into the LLM prompt as an `<active_memory_plugin>` block.

### Architecture

```
User Question
    ↓
Active Memory Subagent → memory_search
    ↓
┌─── memory-core (own memory/ + sessions)
├─── activewiki (plugin)
│       ├── vectordb.py search      (vector chunks)
│       └── vectordb.py graph pages (KG entities + relationships)
│       └── merged, scope-gated
└─── Framework mergeMemorySearchCorpusResults()
    ↓
<active_memory_plugin> block in prompt
```

---

## 2. Codebase

### Structure

```
activewiki/
├── activewiki.example.json   ← config template
├── scopes.json               ← scope mapping example
├── scripts/                  ← Python pipeline
│   ├── config.py             ← config loader
│   ├── ingest.py             ← import documents into inbox
│   ├── distill.py            ← generate wiki pages (LLM)
│   ├── split_pages.py        ← chunk large documents
│   ├── vectordb.py           ← vector database + knowledge graph
│   ├── graph_build.py        ← entity extraction + graph building
│   └── run_inbox.sh          ← master pipeline (all phases)
└── plugin/                   ← TypeScript OpenClaw plugin
    ├── index.ts
    ├── lib/
    │   ├── cli-wrapper.ts    ← hybrid search: vector→graph bridge
    │   ├── scope-resolver.ts ← scope gating
    │   ├── wiki-reader.ts    ← read wiki pages
    │   └── types.ts
    ├── openclaw.plugin.json
    ├── package.json
    ├── tsconfig.json
    └── knowledge-graph.md    ← operations handbook
```

### Plugin (TypeScript)

**`index.ts` — Plugin Entry:**
- Registers `MemoryCorpusSupplement` with `search()` and `get()`
- **`register(api)` is synchronous** (OpenClaw requirement)

**`lib/cli-wrapper.ts` — Hybrid Search:**
1. Vector search over-fetches (`k = clamp(maxResults×3, min 12, max 30)`)
2. Extracts `wiki_page` from top hits (max 8 pages)
3. `graph pages` fetches entities + 1-hop relationships
4. KG quota: ~⅓ of slots reserved for KG hits

**Security measures:**
- `execFile` instead of `exec` (no shell interpolation)
- Whitelist ENV (no secrets leaked to subprocess)
- Timeouts: 30s vector, 10s graph
- Buffer limits: 2MB / 1MB

**`lib/scope-resolver.ts` — Scope Gating:**
- Re-reads scopes config on each search (path from `activewiki.json` or `ACTIVEWIKI_SCOPES_CONFIG`)
- Substring matching: `sessionKey` against `sessionKeyPatterns`
- Subagent workaround: strips `:active-memory:` and `:subagent:` suffixes

**`lib/wiki-reader.ts` — Page Reader:**
- Line-based slicing (`fromLine`, `lineCount`)
- Slug validation: \^\[a-z0-9\-\]{1,100}$
- `safeResolve()`: path must stay within `wiki/<scope>/`

### Scripts (Python)

**`config.py` — Central Config Loader:**
- Reads `activewiki.json` (search order: `--config` → `ACTIVEWIKI_CONFIG` → auto-detect)
- Dot-notation access: `get(config, "embeddings.ollama_url")`
- Helper functions: `wikis_root()`, `scopes()`, `ollama_url()`, `llm_model()` etc.

**`vectordb.py` — Vector Database + Knowledge Graph:**
- Embedding via Ollama (bge-m3 or configured model)
- SQLite storage (`vectordb/index.sqlite`)
- Cosine similarity search (numpy)
- Scope-aware (SQL-level filtering)
- Knowledge Graph: entities, relationships, communities
- Incremental updates (content-hash based)

**`ingest.py` — Document Import:**
- Docling OCR (PDF, images, DOCX) → Markdown
- Scope detection from `inbox/<scope>/`
- Content hashing (avoid duplicates)

**`distill.py` — Wiki Page Generation:**
- LLM-assisted extraction (Docling output → structured wiki pages)
- Hierarchical: folder structure becomes wiki hierarchy
- Bottom-up rollup: parent pages synthesized from child pages

**`run_inbox.sh` — Master Pipeline:**
- Coordinates all phases: Ingest → Distill → Vectordb → Graph
- Deadline-respecting (configurable)
- Lock file (no parallel runs)

---

## 5. Prompt Evolution Pipeline

Inspired by Homer's "Organize then Retrieve" (Duke/Snowflake, 2026-06-10), the Prompt Evolution Loop automatically detects extraction failures, diagnoses root causes, drafts rules, and — after human approval — updates the distillation/graph-extraction prompt. The loop closes with degradation detection to catch rules that make things worse.

### Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────┐     ┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│  FAILURE DETECT  │────▶│  DIAGNOSIS       │────▶│ RULE CONSISTENCY │────▶│  RULE QUEUING   │────▶│  HUMAN REVIEW   │────▶│  PROMPT UPDATE   │
│  (Graph Valid.)  │     │  (LLM Root Cause) │     │ CHECK          │     │  (Approval Queue) │     │  (Mandatory HITL)│     │  (Versioned)     │
└─────────────────┘     └──────────────────┘     └──────────────┘     └─────────────────┘     └─────────────────┘     └─────────┬───────┘
                                                                                                                              │
         ◀──────────────────── METRIKEN-CHECK (Degradation-Detection) ◀───────────────────────────────────────────────────────┘
```

### Component 1: Failure Detector (`graph validate`)

Runs after every graph build or as a cron check. Validates five conditions:

| Rule | Condition | Threshold |
|------|-----------|----------|
| **Dangling Links** | Relations pointing to soft-deleted targets | Only if both endpoints existed at extraction time |
| **Over-Merged Entities** | >3 variants per entity via resolution | Only if caused by extraction, not post-hoc resolution |
| **Orphaned Entities** | Entities with zero relations | Only if ≥5 orphans per page |
| **Confidence Imbalance** | Too many weak-confidence relations | >40% weak (domain-dependent) |
| **Missing Relation Coverage** | Domain-specific relation types not used | Only when context clearly indicates (e.g., "Fusion", "Übernahme") |

Output: `{type, severity, evidence, page_source, entities_involved}`

### Component 2: Diagnosis Engine (`graph diagnose`, `graph evolve`)

Takes a failure event + wiki page + extraction prompt + extracted entities/relations and performs root-cause analysis via LLM with input sanitization (`<SOURCE_START>`/`<SOURCE_END>` markers isolate data from instructions).

Two-stage consistency check:
1. **Deterministic collision check** — does an existing rule cover this error type? Would the new direction conflict?
2. **LLM-based consistency verification** — independent confirmation that diagnosis matches evidence.

Template-based rule drafting transforms the diagnosis direction into a concrete prompt instruction without free-form LLM writing.

Output: `{root_cause, error_type: "exogenous"|"endogenous", rule_direction, drafted_rule_text}`

### Component 3: Rule Storage + Dedup + Queuing (`evolution_rules.json`)

Rules are stored in `evolution_rules.json` with the following structure:

```json
{
  "id": "uuid",
  "text": "rule instruction",
  "severity_weight": 0.8,
  "failures_resolved": ["failure-id-1", "failure-id-2"],
  "created": "2026-06-14T10:00:00Z",
  "status": "pending_approval",
  "originator": "auto",
  "diagnosis_summary": "...",
  "embedding_hash": "sha256:..."
}
```

**Dedup:** Embedding cosine similarity > 0.85 triggers a merge candidate; LLM fine-filter confirms whether they're the same rule. Merged rules increase severity weight instead of creating duplicates.

**Queuing:** New rules enter with `status: "pending_approval"`. Activation requires ≥2 identical failures (or SEVERITY_HIGH for single occurrences).

**Rate limiting:** Maximum 3 new rules per week. Queue blocks if exceeded.

### Component 4: Human-in-the-Loop Review (MANDATORY)

No auto-activation. Every new rule sends a Discord message to Owner in the #system channel:

```
🔄 ActiveWiki Prompt-Evolution — Neue Regel zum Review

⚠️ Problem: [FailureType] bei [Page] — [Count] mal vorkommend
📋 Diagnose: [Root Cause] ([exogenous|endogenous])
📝 Vorgeschlagene Regel: [drafted_rule_text]
📊 Evidenz: [Failure Event Details]

✅ Bestätigen oder ❌ Ablehnen
⏱️ TTL: 14 Tage (Reminder alle 2 Tage)
```

- **Approved** → `approved` → inserted into prompt on next graph build → `active`
- **Rejected** → `rejected`. Won't resurface until ≥3 more identical failures occur after rejection
- **TTL expired** → archived (not deleted). Evidence preserved for later review when Owner returns

### Component 5: Prompt Update + Versioning (`graph apply-prompt`)

Activated rules are appended to the prompt template as new `Good (...)` examples and explicit rule lines with `[AUTO]` markers.

**Versioning:**
- `prompt_history.json`: `{version, hash, applied_rules[], timestamp, author, previous_version, metrics_before{}}`
- Git-based versioning alongside JSON
- Old prompt backed up as `prompts/prompt_v12_backup.md`

### Component 6: Degradation Detection (`graph metrics`, `graph degradation-check`)

Metrics snapshot taken after rule activation (`failure_count`, `avg_confidence_weak_pct`, `orphan_rate`). Compared after 7 days of graph builds.

- **Next-day early warning:** If failure rate increases >50% the day after activation → immediate review request to Owner
- **7-day degradation signal:** If failure rate is equal or higher than before activation → rule automatically downgraded to `deprecated` + alert to Owner
- **Quarantine (two-stage):** `activated` → `quarantine` → `deprecated`
- **Spiral protection (`graph spiral-protection`):** If ≥3 rules degrade consecutively within a month → complete halt of the evolution process until manual release by Owner

### CLI Reference

| Command | Description |
|---------|-------------|
| `vectordb.py graph validate` | Run failure detector on current graph |
| `vectordb.py graph diagnose <failure-id>` | Root-cause analysis for a specific failure |
| `vectordb.py graph evolve` | Full diagnosis + rule drafting pipeline |
| `vectordb.py graph apply-prompt` | Insert approved rules into extraction prompt |
| `vectordb.py graph metrics` | Show current graph health metrics |
| `vectordb.py graph degradation-check` | Compare metrics before/after recent rule activations |
| `vectordb.py graph spiral-protection` | Check if evolution loop should be halted |
| `vectordb.py graph prompt-history` | List all prompt versions and applied rules |
| `vectordb.py graph prompt-backup` | Create backup of current prompt |

### Example Flow

1. Graph build → `graph validate` finds dangling link Volksbank Musterstadt → KREDIT_BEI → DZ BANK and missing fusion relation
2. `graph diagnose` → root cause: fusion context in document not extracted; exogenous error
3. `graph evolve` → no existing rule covers fusion; template-based drafting produces concrete rule text
4. Rule queued with SEVERITY_HIGH; Discord notification sent to Owner
5. Owner approves via Discord
6. `graph apply-prompt` → prompt extended with Good Fusion example + explicit FUSIONIERTE_MIT rule
7. Next graph build → fusion relation automatically extracted → no manual intervention needed ✓

---

## 4. Dependencies

### Python Packages

| Package | Where used | Purpose |
|---------|------------|---------|
| **numpy** | `vectordb.py` | Cosine similarity, matrix operations |
| **python-igraph** | `vectordb.py` | Community detection (Leiden algorithm) |
| **PyYAML** (`yaml`) | `vectordb.py`, `distill.py`, `split_pages.py` | YAML serialization |
| **docling** | `ingest.py` | Document ingestion (PDF/Images/DOCX → Markdown) |

**Minimal install (embedding + search only):**
```bash
pip install numpy pyyaml
```

**Full install (with OCR + community detection):**
```bash
pip install numpy pyyaml python-igraph docling
```

**Note:** `run_inbox.sh` uses a Python venv specified in `ocr.venv_path` (see `activewiki.json`). The venv must contain at least `numpy` and `pyyaml`; add `igraph` for community detection and `docling` for document ingestion.

---

## 3. Configuration

### activewiki.json

Copy `activewiki.example.json` to `activewiki.json` and adapt.

**All configurable options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `wikis_root` | string | _required_ | Root directory containing inbox/, sources/, wiki/, vectordb/ |
| `scopes.enabled` | string[] | [private,family,public] | Enabled scopes |
| `scopes.scopes_config` | string | _required_ | Path to scopes.json (scope gating) |
| `embeddings.backend` | string | ollama | Embedding backend (currently only Ollama) |
| `embeddings.model` | string | bge-m3 | Embedding model (also nomic-embed-text) |
| `embeddings.ollama_url` | string | http://localhost:11434 | Ollama API for embeddings |


| `embeddings.embed_dim` | int | 1024 | Dimensions (bge-m3=1024, nomic=768) |
| `embeddings.chunk_size` | int | 400 | Chunk size in characters |
| `embeddings.chunk_overlap` | int | 50 | Overlap between chunks |
| `embeddings.index_path` | string | vectordb/index.sqlite | SQLite path relative to wikis_root |
| `ocr.engine` | string | docling | OCR engine for PDF/images |
| `ocr.venv_path` | string | _optional_ | Python venv with Docling installed |
| `llm.backend` | string | ollama | LLM backend for distillation |
| `llm.model` | string | _required_ | LLM model name |
| `llm.ollama_url` | string | http://localhost:11434 | Ollama API for LLM |
| `llm.url` | string | http://127.0.0.1:8000/v1 | OpenAI-compatible endpoint (vLLM) |

**`llm.temperature`** (float, default `0.5`) — **Important:** Used centrally by all scripts (`vectordb.py`, `distill.py`, `split_pages.py`) via config — no longer hardcoded!

**`llm.max_tokens`** (int, default `4096`) — Max tokens per response

| Option | Type | Default | Description |

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `graph.build_incremental` | bool | true | KG incremental build |
| `graph.communities_enabled` | bool | true | Community detection (igraph) |
| `graph.communities_incremental_threshold` | int | 5 | Rebuild if more new entities |
| `distill.rollup_all` | bool | true | Bottom-up rollup of wiki hierarchy |
| `ingest.deadline` | string | "03:00" | Pipeline stops at this time |
| `ingest.timezone` | string | "Europe/Berlin" | Timezone for deadline |

Complete template with all options and comments: `activewiki.example.json`.

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
- Ollama running locally (for embeddings — `bge-m3` recommended)
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

### Knowledge Graph
- **Entity extraction** from documents via LLM (OpenAI) — automatically identifies people, places, organizations, and concepts
- **Relationship mapping** — entities are connected by typed, directed relationships (has_parent_object, part_of_project, owned_by, etc.)
- **Confidence tags** — every relationship carries a confidence level (`extracted`, `inferred`, `weak`) reflecting how reliably it was extracted. Visualized with color-coded links in the D3.js graph (green/yellow/gray)
- **Rationale entities** — the "why" behind connections is preserved as traceable rationale nodes, rendered as distinctive blue diamond shapes in the D3.js visualization
- **Community detection** — Louvain algorithm identifies clusters of related entities (e.g., "Real Estate Investment cluster", "Home Automation cluster")
- **Interactive HTML visualization** — explore the graph visually with D3.js: filter by type, search entities, adjust zoom, export as PNG/SVG
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
- **Two-pass extraction** — Pass 1 extracts entities per small chunk (~1500 chars) for granular provenance; Pass 2 extracts relationships per section (~2-5K chars) for sufficient context
- **Click-to-source** — every relationship stores the originating section header and source text snippet (max 200 chars), enabling traceability in the D3.js visualization
- **Entity-chunk mapping** — N:M junction table (`entity_chunks`) tracks which document chunks produced each entity, with source text excerpts
- **Discarded relations tracking** — relationships referencing unknown entities are stored in `discarded_relations` instead of creating stubs, preserving audit trail
- **Canonical ID resolution** — Pass 2 uses a global canonical_id_map combining local entities and EntityRegistry top-N entries for cross-document relationship extraction

### Temporal Filtering
- **Time-bound relationship queries** — `--since YYYY-MM-DD` and `--until YYYY-MM-DD` filter relationships by validity period
- **NULL-safe semantics** — timeless relationships (no date) always pass filters
- **Active vs. historical** — distinguish current relationships from past ones in graph search output

### Active Memory Integration
- OpenClaw plugin automatically injects wiki hits into every LLM response
- Scope-gated results (private/family/public) based on user authorization
- Hybrid search: vector hits bridge to graph entities for combined context

## Full Documentation

See [`plugin/README.md`](plugin/README.md) for complete reference documentation including all configuration options, pipeline phases, installation steps, troubleshooting, and FAQ.

## License

MIT — See [`LICENSE`](LICENSE) for details.

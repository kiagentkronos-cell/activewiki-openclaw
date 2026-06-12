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
- **Community detection** — Louvain algorithm identifies clusters of related entities (e.g., "Musterort real estate cluster", "Heimserver home automation cluster")
- **Interactive HTML visualization** — explore the graph visually with D3.js: filter by type, search entities, adjust zoom, export as PNG/SVG
- **God Nodes CLI** — identify the most connected entities: `vectordb.py graph god-nodes --top 10`

### Entity Resolution (Deduplication)
- **Hybrid matching** — cosine similarity (embeddings) + Jaccard string similarity (token overlap) catches duplicates that pure semantic similarity misses
- **Prevention** — new entities from LLM extraction are checked against existing ones before insertion (soft-delete with `valid_until`)
- **Batch deduplication** — retroactively clean existing graphs: `vectordb.py graph deduplicate`
- **Dry-run mode** — preview merges before applying: `--dry-run` flag shows what would be merged without changing anything
- **Transitive clustering** — Union-Find algorithm ensures A→B→C chains are resolved into single canonical entities

### Active Memory Integration
- OpenClaw plugin automatically injects wiki hits into every LLM response
- Scope-gated results (private/family/public) based on user authorization
- Hybrid search: vector hits bridge to graph entities for combined context

## Full Documentation

See [`plugin/README.md`](plugin/README.md) for complete reference documentation including all configuration options, pipeline phases, installation steps, troubleshooting, and FAQ.

## License

MIT — See [`LICENSE`](LICENSE) for details.

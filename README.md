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

## Full Documentation

See [`plugin/README.md`](plugin/README.md) for complete reference documentation including all configuration options, pipeline phases, installation steps, troubleshooting, and FAQ.

## License

MIT — See [`LICENSE`](LICENSE) for details.

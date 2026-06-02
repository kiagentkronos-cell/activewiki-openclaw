# ActiveWiki — OpenClaw Wiki Integration

**Status:** Stabil/Release | **Letzte Änderung:** 2026-05-30 | **Autor:** Max Mustermann

---

## 1. Überblick

### Was

ActiveWiki ist ein OpenClaw `MemoryCorpusSupplement`-Plugin, das eine Wiki-Vektordatenbank und einen Knowledge Graph automatisch in OpenClaws Active Memory Retrieval-Pipeline integriert.

### Warum

Vor dem Plugin konnte der Active Memory Subagent nur eigene `memory/`-Dateien und Session-Verläufe durchsuchen. Wiki-Inhalte (private, family, public) waren nur manuell über `memory_search corpus=wiki` erreichbar — nicht automatisch bei jeder Antwort.

**Jetzt:** Jede `memory_search`-Query durchsucht automatisch Session-Geschichte + Wiki-Vektordatenbank + Knowledge Graph. Ergebnisse werden gemerged, nach Score sortiert, und als `<active_memory_plugin>`-Block in den LLM-Prompt injiziert.

### Architektur

```
Nutzerfrage
    ↓
Active Memory Subagent → memory_search
    ↓
┌─── memory-core (eigene memory/ + Sessions)
├─── activewiki (Plugin)
│       ├── vectordb.py search      (Vektor-Chunks)
│       └── vectordb.py graph pages (KG-Entities + Beziehungen)
│       └── gemergt, scope-gegated
└─── Framework mergeMemorySearchCorpusResults()
    ↓
<active_memory_plugin>-Block im Prompt
```

---

## 2. Codebasis

### Struktur

```
activewiki/
├── activewiki.example.json   ← Config-Vorlage
├── scopes.json               ← Scope-Mapping Beispiel
├── scripts/                  ← Python-Pipeline
│   ├── config.py             ← Config-Loader
│   ├── ingest.py             ← Dokumente in Inbox importieren
│   ├── distill.py            ← Wiki-Seiten generieren (LLM)
│   ├── split_pages.py        ← Große Dokumente chunken
│   ├── vectordb.py           ← Vektordatenbank + Knowledge Graph
│   ├── graph_build.py        ← Entity-Extraktion + Graph-Build
│   └── run_inbox.sh          ← Master-Pipeline (alle Phasen)
└── plugin/                   ← TypeScript OpenClaw-Plugin
    ├── index.ts
    ├── lib/
    │   ├── cli-wrapper.ts    ← hybride Suche: Vektor→Graph-Bridge
    │   ├── scope-resolver.ts ← Scope-Gating
    │   ├── wiki-reader.ts    ← Wiki-Seiten lesen
    │   └── types.ts
    ├── openclaw.plugin.json
    ├── package.json
    ├── tsconfig.json
    └── knowledge-graph.md    ← Betriebshandbuch
```

### Plugin (TypeScript)

**`index.ts` — Plugin-Entry:**
- Registriert `MemoryCorpusSupplement` mit `search()` und `get()`
- **`register(api)` ist synchron** (OpenClaw-Anforderung)

**`lib/cli-wrapper.ts` — Hybride Suche:**
1. Vektorsuche überfetcht (`k = clamp(maxResults×3, min 12, max 30)`)
2. Top-Treffer → `wiki_page` extrahiert (max 8 Pages)
3. `graph pages` holt Entities + 1-Hop-Beziehungen
4. KG-Quota: ~⅓ der Slots für KG-Treffer reserviert

**Sicherheitsmaßnahmen:**
- `execFile` statt `exec` (keine Shell-Interpolation)
- Whitelist-ENV (keine Secrets an Subprozess)
- Timeouts: 30s Vektor, 10s Graph
- Buffer-Limits: 2MB / 1MB

**`lib/scope-resolver.ts` — Scope-Gating:**
- Liest Scopes-Config bei jeder Suche neu (Pfad aus `activewiki.json` oder `ACTIVEWIKI_SCOPES_CONFIG`)
- Substring-Matching: `sessionKey` gegen `sessionKeyPatterns`
- Subagent-Workaround: Strippt `:active-memory:` und `:subagent:` Suffixe

**`lib/wiki-reader.ts` — Seite lesen:**
- Line-based Slicing (`fromLine`, `lineCount`)
- Slug-Validierung: `^[a-z0-9\-]{1,100}$`
- `safeResolve()`: Pfad muss innerhalb `wiki/<scope>/` bleiben

### Scripts (Python)

**`config.py` — Zentraler Config-Loader:**
- Liest `activewiki.json` (Suchreihenfolge: `--config` → `ACTIVEWIKI_CONFIG` → auto-detect)
- Dot-notation Access: `get(config, "embeddings.ollama_url")`
- Helper-Funktionen: `wikis_root()`, `scopes()`, `ollama_url()`, `llm_model()` etc.

**`vectordb.py` — Vektordatenbank + Knowledge Graph:**
- Embedding über Ollama (bge-m3 oder konfiguriertes Modell)
- SQLite-Storage (`vectordb/index.sqlite`)
- Cosine Similarity Suche (numpy)
- Scope-aware (SQL-level Filterung)
- Knowledge Graph: Entities, Relationships, Communities
- Incremental Updates (content-hash basierend)

**`ingest.py` — Dokument-Import:**
- Docling OCR (PDF, Bilder, DOCX) → Markdown
- Scope-Erkennung aus `inbox/<scope>/`
- Content-Hashing (Duplikate vermeiden)

**`distill.py` — Wiki-Seiten-Generierung:**
- LLM-gestützte Extraktion (Docling Output → strukturierte Wiki-Seiten)
- Hierarchisch: Ordnerstruktur wird zur Wiki-Hierarchie
- Bottom-up Rollup: Parent-Seiten aus Child-Seiten synthetisiert

**`run_inbox.sh` — Master-Pipeline:**
- Koordiniert alle Phasen: Ingest → Distill → Vectordb → Graph
- Deadline-Respecting (konfigurierbar)
- Lock-File (keine parallelen Runs)

---

## 3. Konfiguration

### activewiki.json

Kopiere `activewiki.example.json` nach `activewiki.json` und passe an.

**Alle konfigurierbaren Optionen:**

| Option | Typ | Standardwert | Beschreibung |
|--------|-----|-------------|--------------|
| `wikis_root` | string | _required_ | Root-Verzeichnis mit inbox/, sources/, wiki/, vectordb/ |
| `scopes.enabled` | string[] | [private,family,public] | Aktivierte Scopes |
| `scopes.scopes_config` | string | _required_ | Pfad zu scopes.json (Scope-Gating) |
| `embeddings.backend` | string | ollama | Embedding-Backend (derzeit nur Ollama) |
| `embeddings.model` | string | bge-m3 | Embedding-Modell (auch nomic-embed-text) |
| `embeddings.ollama_url` | string | http://localhost:11434 | Ollama-API fuer Embeddings |
| `embeddings.embed_dim` | int | 1024 | Dimensionen (bge-m3=1024, nomic=768) |
| `embeddings.chunk_size` | int | 400 | Chunk-Groesse in Zeichen |
| `embeddings.chunk_overlap` | int | 50 | Overlap zwischen Chunks |
| `embeddings.index_path` | string | vectordb/index.sqlite | SQLite-Pfad relativ zu wikis_root |
| `ocr.engine` | string | docling | OCR-Engine fuer PDF/Bilder |
| `ocr.venv_path` | string | _optional_ | Python Venv mit Docling |
| `llm.backend` | string | ollama | LLM-Backend fuer Distillation |
| `llm.model` | string | _required_ | LLM-Modell-Name |
| `llm.ollama_url` | string | http://localhost:11434 | Ollama-API fuer LLM |
| `llm.url` | string | http://127.0.0.1:8000/v1 | OpenAI-kompatibler Endpoint (vLLM) |

**`llm.temperature`** (float, default `0.5`) — **Wichtig:** Von allen Scripts (`vectordb.py`, `distill.py`, `split_pages.py`) zentral ueber Config verwendet — nicht mehr hardcoded!

**`llm.max_tokens`** (int, default `4096`) — Max Tokens pro Antwort

| Option | Typ | Standardwert | Beschreibung |
|--------|-----|-------------|--------------|
| `graph.build_incremental` | bool | true | KG incremental build |
| `graph.communities_enabled` | bool | true | Community Detection (igraph) |
| `graph.communities_incremental_threshold` | int | 5 | Neubuild wenn mehr neue Entities |
| `distill.rollup_all` | bool | true | Bottom-up Rollup der Wiki-Hierarchie |
| `ingest.deadline` | string | "03:00" | Pipeline stoppt zu dieser Zeit |
| `ingest.timezone` | string | "Europe/Berlin" | Zeitzone fuer Deadline |

Vollstaendige Vorlage mit allen Optionen und Kommentaren: `activewiki.example.json`.
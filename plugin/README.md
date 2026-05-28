# ActiveWiki — OpenClaw Wiki Integration

**Status:** Entwicklung | **Letzte Änderung:** 2026-05-23 | **Autor:** Kronos

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

### aktivewiki.json

Kopiere `activewiki.example.json` nach `activewiki.json` und passe an:

```json
{
  "wikis_root": "/path/to/your/wikis",
  "embeddings": {
    "model": "bge-m3",
    "ollama_url": "http://localhost:11434"
  },
  "llm": {
    "model": "qwen3.6-fp8",
    "url": "http://127.0.0.1:8000/v1"
  },
  "scopes": {
    "scopes_config": "/path/to/scopes.json"
  }
}
```

Vollständige Config-Referenz in `activewiki.example.json`.

### scopes.json

Definiert wer welche Scopes sehen darf:

```json
{
  "entries": [
    {
      "name": "admin",
      "scopes": ["private", "family", "public"],
      "sessionKeyPatterns [": [":discord:", ":whatsapp:direct:+49..."]
    }
  ],
  "default": { "scopes": ["public"] }
}
```

**Wichtig:** Wird bei jeder Suche neu gelesen — keine Neustart nötig.

### Umgebungsvariablen

| Variable | Zweck |
|----------|-------|
| `ACTIVEWIKI_CONFIG` | Pfad zu `activewiki.json` |
| `ACTIVEWIKI_WIKIS_ROOT` | Override für `wikis_root` |
| `ACTIVEWIKI_SCOPES_CONFIG` | Override für `scopes.json` Pfad |
| `ACTIVEWIKI_PYTHON_BIN` | Python-Binary für Plugin |
| `ACTIVEWIKI_OCR_VENV` | Docling Venv-Pfad |

---

## 4. Installation

### Vorbereitungen

1. **Ollama** mit Embedding-Modell (`bge-m3` oder ähnlich)
2. **LLM-Backend** für Distillation (vLLM, Ollama, OpenAI-compatible)
3. **Docling** (für OCR) — Venv empfohlen
4. **OpenClaw** mit Plugin-Unterstützung

### Plugin Installieren

```bash
# TypeScript compilieren
cd plugin/
npm install
npm run build

# Plugin installieren (symbolischer Link)
openclaw plugins install --link /path/to/activewiki/plugin

# Config-Umgebungsvariablen setzen (z.B. in ~/.bashrc oder systemd Unit)
export ACTIVEWIKI_CONFIG="/path/to/activewiki/activewiki.json"
export ACTIVEWIKI_WIKIS_ROOT="/path/to/your/wikis"

# Gateway neu starten
systemctl --user restart openclaw-gateway.service
```

### Wikis-Verzeichnis anlegen

```bash
mkdir -p $WIKIS_ROOT/{inbox/{private,family,public},sources/{private,family,public},wiki/{private,family,public},vectordb,logs}
```

### Schnelltest

```bash
cd plugin/
node --input-type=module -e '
  import { searchWiki } from "./dist/lib/cli-wrapper.js";
  const r = await searchWiki("Testfrage?", ["public"], 6);
  console.log(JSON.stringify(r, null, 2));
'
```

---

## 5. Pipeline: Von Dokument zu Active Memory

### Der vollständige Workflow

```
inbox/<scope>/document.pdf
         ↓
   [INGEST]  ingest.py → OCR (Docling) → sources/<scope>/xxx/docling.md
         ↓
   [DISTILL] distill.py → LLM-Extraktion → wiki/<scope>/xxx.md
         ↓
   [VECTOR]  vectordb.py → Chunking + Embedding → index.sqlite
         ↓
   [GRAPH]   vectordb.py graph build → Entities + Relationships
         ↓
   [PLUGIN]  aktivewiki plugin → memory_search → <active_memory_plugin>
```

### Phase 1: Ingest

`scripts/ingest.py` verarbeitet Dateien aus `inbox/<scope>/`:
- Docling konvertiert PDF/Bilder/DOCX zu Markdown
- Content-Hash verhindert Duplikate
- Scope wird aus dem Ordner abgeleitet

**Gesteuert von:** `run_inbox.sh` (automatisch) oder manuell

### Phase 2: Distillation

`scripts/distill.py` generiert Wiki-Seiten:
- Docling-Output wird vom LLM strukturiert extrahiert
- Ordnerhierarchie → Wiki-Hierarchie
- Bottom-up Rollup: Parent-Seiten synthetisieren Child-Inhalte

**LLM-Anforderung:** Mindestens 7B Parameter, 32K+ Context. Empfohlen: qwen3.6-fp8 oder ähnliches.

### Phase 3: Vektordatenbank

`scripts/vectordb.py` baut die Such-Indizes:
- Wiki-Seiten werden in Chunks zerlegt
- Ollama-Embeddings (bge-m3)
- Incremental: nur geänderte/neue Seiten werden eingebettet

### Phase 4: Knowledge Graph

`scripts/vectordb.py graph build`:
- LLM extrahiert Entities (Personen, Orte, Organisationen) und deren Beziehungen
- Entities werden zu Graph-Knoten, Beziehungen zu Kanten
- Community Detection (igraph, optional)

### Phase 5: Active Memory Integration

Das TypeScript-Plugin macht alles verfügbar:
- Bei jeder `memory_search`-Query durchsucht das Plugin die Vektordatenbank
- Vektor→Graph-Bridge verbindet semantische Suche mit Beziehungsnetzwerk
- Scope-Gating stellt sicher, dass Nutzer nur erlaubte Daten sehen

---

## 6. Operationale Hinweise

### Täglicher Cron

`run_inbox.sh` sollte als Cron-Job laufen (empfohlen: nachts):
- Neue Dokumente in `inbox/` werden verarbeitet
- Vektordatenbank wird incremental aktualisiert
- Knowledge Graph wird neu gebaut (incremental)

### Wartung

**Scopes aktualisieren:**
`scopes.json` editieren — keine Neustart nötig

**KG-Quota ändern:**
`plugin/lib/cli-wrapper.ts` → Divisor in `Math.floor(maxResults / 3)` → `npm run build` → restart

**Status prüfen:**
```bash
openclaw plugins list | grep activewiki
systemctl --user is-active openclaw-gateway.service
```

### Troubleshooting

**Keine Wiki-Treffer obwohl Inhalt existiert?**
1. Gateway-Logs: `journalctl --user -u openclaw-gateway -n 50 | grep activewiki`
2. Scopes prüfen: `$PYTHON scripts/vectordb.py search --json -k 3 --scopes public "Test"`
3. Plugin aktiv? `openclaw plugins list | grep activewiki`

**Stille Fehlschläge:**
Das Plugin degradiert überall graceful (`return []` bei Vektor-Fehler, `return []` bei Graph-Fehler). Wenn alles schiefgeht, erscheinen einfach keine Wiki-Treffer. Das ist gewollt (crasht nicht), macht aber Debugging ohne Logs schwierig.

---

## 7. FAQ

**Warum eine Bridge statt direkt `graph search`?**
`graph search` macht nur `SQL LIKE %query%` gegen Entity-Labels. Das trifft natürlichsprachige Queries fast nie. Die Bridge nutzt Vektorsuche um relevante Seiten zu finden, dann holt der Graph die Beziehungen dieser Seiten.

**Wie viel RAM braucht die Vektordatenbank?**
Abhängig von der Chunk-Anzahl. ~10.000 Chunks → ~40MB RAM für Embeddings. ~50.000 → ~200MB. Keine Index-Struktur nötig (brute-force cosine similarity auf numpy arrays).

**Kann ich mehrere Wikis betreiben?**
Ja — jedes `wikis_root` ist isoliert. Du brauchst separate `activewiki.json` Instanzen.

**Was passiert bei OpenClaw-Updates?**
`MemoryCorpusSupplement` API könnte sich ändern. Nach jedem OpenClaw-Update prüfen:
1. `openclaw plugins list` → Plugin aktiv?
2. Logs → keine Ladefehler?
3. Funktionstest → Wiki-Frage stellen → Treffer?
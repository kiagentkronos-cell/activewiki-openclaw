# Wikis: Vektorsuche + Knowledge Graph in Active Memory

Begleitdoku zu `SKILL.md`. Erklärt, **wie** die Wiki-Treffer in den `<active_memory_plugin>`-Block kommen und wie der Knowledge Graph daran hängt. Stand: 2026-05-22.

## Das große Bild

Wenn ich antworte, baut der **active-memory**-Plugin von OpenClaw einen Erinnerungs-Kontext zusammen. Neben meinen eigenen Erinnerungen (`memory/`) fragt er **automatisch alle registrierten Memory-Corpus-Supplements** ab. Eines davon ist mein eigenes Plugin **`activewiki`** — es durchsucht die Wiki-Vektordatenbank UND den Knowledge Graph und liefert die Treffer scope-gegated zurück.

> **Wichtig:** Es gibt **keinen Schalter in der active-memory-Config** dafür. Das Supplement meldet sich beim Start selbst an (`registerMemoryCorpusSupplement`), und active-memory konsumiert es automatisch. Wer „im active memory einstellen will, dass auch das Wiki durchsucht wird", sucht an der falschen Stelle — die ganze Logik sitzt im Plugin, nicht in der Config.

```
Frage → active-memory → activewiki (Supplement)
                              │
                              ├─ vectordb.py search   (Vektor-Chunks, configured model)
                              └─ vectordb.py graph pages (KG-Entities + Beziehungen)
                              → gemergt, scope-gegated → <active_memory_plugin>-Block
```

## Wo das Plugin liegt

```
plugin//
├── index.ts              ← Supplement-Registrierung (search + get), Scope-Gating
├── lib/cli-wrapper.ts    ← ruft vectordb.py (Vektor + Graph), merged, KG-Quota
├── lib/scope-resolver.ts ← resolveScopes(sessionKey) → erlaubte Scopes
├── lib/wiki-reader.ts    ← get(): einzelne Wiki-Seite lesen
└── dist/                 ← KOMPILIERT — das ist der echte Entry, NICHT index.ts
```

Geladen über `plugins.load.paths` in `openclaw.json`; in `plugins.entries` als `activewiki: { enabled: true }`.

## Die hybride Suche — Vektor→Graph-Bridge

`graph search` von vectordb.py macht nur `SQL LIKE %query%` gegen Entity-Labels. Das trifft eine **natürlichsprachige Mehrwort-Query fast nie** (`"Mein-Begriff"` trifft, `"Testfrage?"` liefert `[]`). Deshalb hängt der KG **nicht** direkt an der Roh-Query, sondern an einer Bridge:

1. **Vektorsuche** (überfetcht, `k = clamp(maxResults*3, 12, 30)`) findet die semantisch relevantesten Wiki-Seiten.
2. Aus den Top-Treffern (`kind == "wiki"`) wird die `wiki_page = "<scope>/<ref>"` abgeleitet (verlustfrei — jede Entity-Seite entspricht genau einer Chunk-Seite).
3. **`vectordb.py graph pages`** holt die Entities **dieser Seiten** + ihre 1-Hop-Beziehungen.
4. **Merge mit KG-Quota:** ~⅓ der Plätze (`floor(maxResults/3)`) sind für KG-Treffer reserviert. Sonst würden die höher-scorenden Vektor-Chunks (0.5+) die KG-Entities (Fixscore 0.45) beim Slice komplett verdrängen. Nur Entities **mit** Beziehungen kommen durch (beziehungslose sind nur Seiten-Platzhalter).

So fließt bei jeder Frage sowohl der relevante Volltext-Chunk als auch das Beziehungsgeflecht der relevanten Entities in meine Antwort ein.

## Knowledge-Graph-CLI (manuelles Fallback / Debugging)

Der Graph lebt in derselben `vectordb/index.sqlite` (Tabellen `entities`, `relationships`, `communities`).

```bash
cd wikis_root
PY=wikis_root/venv/bin/python3

# Graph-Statistik (Anzahl Entities/Relationships, Typen, Orphans)
$PY scripts/vectordb.py graph stats

# Label-/Description-Suche (LIKE) — nur für EXAKTE Mein-Begriffe/Namen
$PY scripts/vectordb.py graph search --json --scopes private,family,public "Mein-Begriff"

# Vektor→Graph-Bridge: Entities + Beziehungen für konkrete Seiten
$PY scripts/vectordb.py graph pages --json \
    --scopes private,family,public \
    --pages "private/mein-thema,family/anderes-thema"

# Graph (neu) bauen — passiert normalerweise im nächtlichen Cron
$PY scripts/vectordb.py graph build            # vollständig
$PY scripts/vectordb.py graph build --incremental   # nur geänderte Seiten

# Communities (Phase 2, igraph)
$PY scripts/vectordb.py graph communities list
```

`graph pages` und `graph search` liefern dasselbe JSON-Format:
`{label, type, description, wiki_page, outgoing:[{relation_type,target,description}], incoming:[{relation_type,source,description}]}`.

**Scope-Disziplin gilt auch hier:** `--scopes` immer auf die für den aktuellen Chat erlaubten Scopes setzen. Die `wiki_page` trägt den Scope als Präfix — der Filter greift defensiv darüber.

## Wartung — Plugin-Fallen (hart erkauft am 2026-05-22)

- **Der Entry ist `dist/index.js`, nicht `index.ts`.** Nach **jedem** Edit an `index.ts` oder `lib/*.ts` zwingend neu bauen, sonst startet das Gateway nicht (`extension entry not found: dist/index.js`):
  ```bash
  cd plugin/ && npm run build
  ```
- **`tsconfig.json`** muss `module: ES2022` + `moduleResolution: bundler` + `ignoreDeprecations: "6.0"` nutzen (das Paket ist `type: module`). `commonjs`/`node` crasht unter TypeScript 6 (`TS5107`).
- Vor riskanten Umbauten den funktionierenden Stand sichern (Konvention: `*.bak-pre-<thema>-<ts>`); bei Defekt `index.ts` von dort restaurieren und neu bauen.
- Nach Plugin-Änderung Gateway neu starten und prüfen:
  ```bash
  systemctl --user restart openclaw-gateway.service
  systemctl --user is-active openclaw-gateway.service   # → active
  ```
- Schnelltest der Such-Kette ohne Gateway (zeigt, ob KG-Treffer ankommen):
  ```bash
  cd plugin/
  node --input-type=module -e '
    import { searchWiki } from "./dist/lib/cli-wrapper.js";
    const r = await searchWiki("Testfrage?", ["private","family","public"], 6);
    console.log(r.filter(x=>x.kind==="graph-entity"));'
  ```

## Was am 2026-05-22 passiert ist

1. Ich hatte das Plugin angefasst („kg"-Arbeit), `index.ts` + `tsconfig.json` umgeschrieben, aber **nicht neu gebaut** → `dist/index.js` fehlte → Gateway-Crash beim Start.
2. Fix: `tsconfig` auf ES2022/bundler zurück, funktionierende `index.ts` aus dem `pre-kg`-Backup restauriert (inkl. Scope-Gating), `npm run build`.
3. Danach die eigentlich gewollte KG-Funktion sauber gebaut: neuer `graph pages`-Befehl in `vectordb.py` (Helper `_entities_to_json` + `cmd_graph_pages`) und die Vektor→Graph-Bridge in `lib/cli-wrapper.ts` mit KG-Quota.

Seitdem fließen bei natürlichsprachigen Fragen echte Graph-Beziehungen in meine Antworten (z. B. „Testfrage?" → `Vertrag 12345 — Projekt Musterort FINANZIERT_VON`).

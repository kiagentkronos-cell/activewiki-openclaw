# Wikis: Vector Search + Knowledge Graph in Active Memory

Companion documentation for `SKILL.md`. Explains **how** wiki hits end up in the `<active_memory_plugin>` block and how the knowledge graph ties in. Status: 2026-05-22.

## The Big Picture

When I respond, OpenClaw's **active-memory** plugin assembles a memory context. Alongside my own memories (`memory/`), it **automatically queries all registered memory corpus supplements**. One of them is my own plugin **`activewiki`** — it searches the wiki vector database AND the knowledge graph and returns scope-gated results.

> **Important:** There is **no switch in the active-memory config** for this. The supplement registers itself at startup (`registerMemoryCorpusSupplement`), and active-memory consumes it automatically. If you're looking for where to "enable wiki search in active memory config", you're looking in the wrong place — all the logic lives in the plugin, not in config.

```
Question → active-memory → activewiki (supplement)
                              │
                              ├─ vectordb.py search   (vector chunks, configured model)
                              └─ vectordb.py graph pages (KG entities + relationships)
                              → merged, scope-gated → <active_memory_plugin> block
```

## Where the Plugin Lives

```
plugin/
├── index.ts              ← Supplement registration (search + get), scope gating
├── lib/cli-wrapper.ts    ← calls vectordb.py (vector + graph), merges, KG quota
├── lib/scope-resolver.ts ← resolveScopes(sessionKey) → allowed scopes
├── lib/wiki-reader.ts    ← get(): read single wiki page
└── dist/                 ← COMPILED — this is the real entry, NOT index.ts
```

Loaded via `plugins.load.paths` in `openclaw.json`; enabled in `plugins.entries` as `activewiki: { enabled: true }`.

## The Hybrid Search — Vector→Graph Bridge

`graph search` in vectordb.py only does `SQL LIKE %query%` against entity labels. This almost never matches a **natural-language multi-word query** (`"My-Concept"` matches, `"Test question?"` returns `[]`). So the KG is **not** directly attached to the raw query, but to a bridge:

1. **Vector search** (over-fetching, `k = clamp(maxResults*3, 12, 30)`) finds the most semantically relevant wiki pages.
2. From top hits (`kind == "wiki"`), derive `wiki_page = "<scope>/<ref>"` (lossless — every entity page corresponds to exactly one chunk page).
3. **`vectordb.py graph pages`** fetches entities **of these pages** + their 1-hop relationships.
4. **Merge with KG quota:** ~⅓ of slots (`floor(maxResults/3)`) are reserved for KG hits. Otherwise higher-scoring vector chunks (0.5+) would completely push out KG entities (fixed score 0.45) during slicing. Only entities **with** relationships pass through (relationship-less ones are just page placeholders).

This way, both the relevant full-text chunk and the relationship network of relevant entities flow into my responses for every question.

## Knowledge Graph CLI (Manual Fallback / Debugging)

The graph lives in the same `vectordb/index.sqlite` (tables `entities`, `relationships`, `communities`).

```bash
cd wikis_root
PY=wikis_root/venv/bin/python3

# Graph statistics (entity/relationship counts, types, orphans)
$PY scripts/vectordb.py graph stats

# Label/description search (LIKE) — only for EXACT concept names
$PY scripts/vectordb.py graph search --json --scopes private,family,public "My-Concept"

# Vector→Graph bridge: entities + relationships for specific pages
$PY scripts/vectordb.py graph pages --json \
    --scopes private,family,public \
    --pages "private/my-topic,family/another-topic"

# Build graph — normally happens in nightly cron
$PY scripts/vectordb.py graph build            # full rebuild
$PY scripts/vectordb.py graph build --incremental   # only changed pages

# Communities (phase 2, igraph)
$PY scripts/vectordb.py graph communities list
```

`graph pages` and `graph search` return the same JSON format:
`{label, type, description, wiki_page, outgoing:[{relation_type,target,description}], incoming:[{relation_type,source,description}]}`.

**Scope discipline also applies here:** always set `--scopes` to the scopes allowed for the current chat. The `wiki_page` carries the scope as a prefix — the filter applies defensively on top of that.

## Maintenance — Plugin Pitfalls (Hard-Learned on 2026-05-22)

- **The entry is `dist/index.js`, not `index.ts`.** After **every** edit to `index.ts` or `lib/*.ts`, you must rebuild or the gateway won't start (`extension entry not found: dist/index.js`):
  ```bash
  cd plugin/ && npm run build
  ```
- **`tsconfig.json`** must use `module: ES2022` + `moduleResolution: bundler` + `ignoreDeprecations: "6.0"` (the package is `type: module`). `commonjs`/`node` crashes under TypeScript 6 (`TS5107`).
- Before risky refactors, save the working state (convention: `*.bak-pre-<topic>-<timestamp>`); on failure restore from backup and rebuild.
- After plugin changes, restart the gateway and verify:
  ```bash
  systemctl --user restart openclaw-gateway.service
  systemctl --user is-active openclaw-gateway.service   # → active
  ```
- Quick test of the search chain without gateway (shows whether KG hits arrive):
  ```bash
  cd plugin/
  node --input-type=module -e '
    import { searchWiki } from "./dist/lib/cli-wrapper.js";
    const r = await searchWiki("Test question?", ["private","family","public"], 6);
    console.log(r.filter(x=>x.kind==="graph-entity"));'
  ```

## What Happened on 2026-05-22

1. I touched the plugin ("KG" work), rewrote `index.ts` + `tsconfig.json`, but **did not rebuild** → `dist/index.js` was missing → gateway crash at startup.
2. Fix: reverted `tsconfig` to ES2022/bundler, restored working `index.ts` from `pre-kg` backup (incl. scope gating), `npm run build`.
3. Then properly built the intended KG function: new `graph pages` command in `vectordb.py` (helper `_entities_to_json` + `cmd_graph_pages`) and the vector→graph bridge in `lib/cli-wrapper.ts` with KG quota.

Since then, real graph relationships flow into my responses for natural-language questions (e.g., "Test question?" → `Contract 12345 — Project Musterort FINANCED_BY`).

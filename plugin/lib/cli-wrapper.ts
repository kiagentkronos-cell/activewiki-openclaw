import { execFile } from "node:child_process";
import { readFileSync } from "node:fs";
import { promisify } from "node:util";
import { resolve, join } from "node:path";
import type { VectorSearchResult } from "./types.js";

const execFileAsync = promisify(execFile);

// ── Config resolution (from env, set by OpenClaw or deployment) ─────────────

/**
 * Resolve the Python binary path.
 * Priority: ACTIVEWIKI_PYTHON_BIN > ACTIVEWIKI_OCR_VENV/bin/python3 > python3
 */
function resolvePythonBin(): string {
  if (process.env.ACTIVEWIKI_PYTHON_BIN) {
    return process.env.ACTIVEWIKI_PYTHON_BIN;
  }
  const venvPath = process.env.ACTIVEWIKI_OCR_VENV;
  if (venvPath) {
    return join(venvPath, "bin", "python3");
  }
  return "python3";
}

/**
 * Resolve the wikis root directory.
 * Priority: ACTIVEWIKI_WIKIS_ROOT > read from ACTIVEWIKI_CONFIG > throw
 */
function resolveWikisRoot(): string {
  // Direct env override
  if (process.env.ACTIVEWIKI_WIKIS_ROOT) {
    return resolve(process.env.ACTIVEWIKI_WIKIS_ROOT);
  }

  // Read from activewiki.json
  const configPath = process.env.ACTIVEWIKI_CONFIG;
  if (configPath) {
    try {
      const cfg = JSON.parse(readFileSync(configPath, "utf-8"));
      const raw = cfg.wikis_root;
      if (raw) return resolve(raw);
    } catch {
      // Ignore parse errors, fall through
    }
  }

  throw new Error(
    "Cannot resolve wikis_root. Set ACTIVEWIKI_WIKIS_ROOT or ACTIVEWIKI_CONFIG."
  );
}

// ── Derived constants ────────────────────────────────────────────────────────

const PYTHON_BIN = resolvePythonBin();
const WIKIS_ROOT = resolveWikisRoot();
const VECTORDB_SCRIPT = join(WIKIS_ROOT, "scripts", "vectordb.py");
const SEARCH_TIMEOUT_MS = 30_000;
const MAX_BUFFER_BYTES = 2 * 1024 * 1024; // 2 MB

const GRAPH_TIMEOUT_MS = 10_000;
const GRAPH_MAX_BUFFER = 1 * 1024 * 1024; // 1 MB

/** Environment variables that are safe to pass through to the subprocess. */
const ALLOWED_ENV_KEYS = new Set([
  "PATH",
  "HOME",
  "LANG",
  "LC_ALL",
  "PYTHONUNBUFFERED",
  "PYTHONPATH",
  "OLLAMA_HOST",
  "OLLAMA_URL",
  "ACTIVEWIKI_DB_PATH",
  "ACTIVEWIKI_CONFIG",
  "ACTIVEWIKI_EMBEDDING_MODEL",
  "ACTIVEWIKI_WIKIS_ROOT",
  "ACTIVEWIKI_PYTHON_BIN",
  "ACTIVEWIKI_OCR_VENV",
  "ACTIVEWIKI_SCOPES_CONFIG",
]);

/** Build a safe environment object: only whitelisted keys from process.env. */
function buildSafeEnv(): NodeJS.ProcessEnv {
  const env: NodeJS.ProcessEnv = {};
  for (const [key, value] of Object.entries(process.env)) {
    if (ALLOWED_ENV_KEYS.has(key) && value !== undefined) {
      env[key] = value;
    }
  }
  // Ensure essential vars exist
  if (!env.PATH) env.PATH = process.env.PATH ?? "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin";
  if (!env.HOME) env.HOME = process.env.HOME ?? (process.platform === 'win32' ? process.env.USERPROFILE || '' : '');
  if (!env.LANG) env.LANG = "de_DE.UTF-8";
  env.PYTHONUNBUFFERED = "1";
  return env;
}

const SAFE_ENV = buildSafeEnv();

export interface GraphSearchEntity {
  label: string;
  type: string;
  description?: string;
  wiki_page?: string;
  outgoing?: GraphSearchRelation[];
  incoming?: GraphSearchRelation[];
}

export interface GraphSearchRelation {
  relation_type: string;
  target?: string;
  source?: string;
  description?: string;
}

// ── Raw JSON output shape from vectordb.py ───────────────────────────────────

interface RawVectordbHit {
  id: number;
  scope: string;
  kind: string;
  ref: string;
  section?: string;
  chunk_idx?: number;
  content: string;
  score: number;
}

// ── Public API ───────────────────────────────────────────────────────────────

/** How many top wiki pages from the vector search feed the graph bridge. */
const BRIDGE_MAX_PAGES = 8;

/**
 * Perform a hybrid search across the ActiveWiki corpus: semantic vector search
 * plus a vector→graph bridge that enriches results with knowledge-graph relations.
 *
 * Strategy:
 *   1. Vector search (over-fetched) finds the most relevant wiki pages.
 *   2. Those pages anchor a `graph pages` lookup — entities + 1-hop relations
 *      living on exactly the pages the query is semantically about. This makes
 *      the graph reachable from natural-language queries (raw label LIKE-matching
 *      via `graph search` almost never hits a multi-word query).
 *   3. Merge with a reserved KG quota so graph entities always survive into the
 *      injection instead of being out-ranked by vector chunks.
 *
 * Uses `execFile` (no shell interpolation), `--json` output, explicit scopes.
 */
export async function searchWiki(
  query: string,
  scopes: string[],
  maxResults: number
): Promise<VectorSearchResult[]> {
  // 1. Over-fetch vector hits so the bridge sees enough distinct pages.
  const vectorPool = await vectorSearch(
    query,
    scopes,
    Math.min(Math.max(maxResults * 3, 12), 30)
  );

  const adjusted = vectorPool
    .sort((a, b) => b.score - a.score);

  // 2. Bridge: derive the most relevant wiki pages (only wiki kind has entities)
  //    and pull their graph neighbourhood. wiki_page == "<scope>/<ref>".
  const pages: string[] = [];
  for (const h of adjusted) {
    if (h.kind !== "wiki") continue;
    const page = `${h.scope}/${h.ref}`;
    if (!pages.includes(page)) pages.push(page);
    if (pages.length >= BRIDGE_MAX_PAGES) break;
  }
  const graphEntities = pages.length ? await graphSearchByPages(pages, scopes) : [];

  // 3. Build KG hits, keeping only entities that actually carry relations
  //    (relation-less entities are page placeholders with no added value).
  const kgHits: VectorSearchResult[] = graphEntities
    .filter((e) => (e.outgoing?.length ?? 0) + (e.incoming?.length ?? 0) > 0)
    .map((entity) => ({
      id: hashCode(`kg:${entity.label}`),
      scope: (entity.wiki_page || "public").split("/")[0],
      kind: "graph-entity",
      ref: entity.wiki_page || entity.label.toLowerCase().replace(/[^a-z0-9]+/g, "-"),
      content: buildKgSnippet(entity),
      score: 0.45,
    }));

  // 4. Quota merge: reserve up to ~1/3 of slots for KG so it survives the cut.
  const kgQuota = Math.min(kgHits.length, Math.max(1, Math.floor(maxResults / 3)));
  const vecCount = Math.max(0, maxResults - kgQuota);
  return [...adjusted.slice(0, vecCount), ...kgHits.slice(0, kgQuota)];
}

/** Simple hash for consistent numeric IDs */
function hashCode(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) {
    h = ((h << 5) - h + s.charCodeAt(i)) | 0;
  }
  return h;
}

/** Render an entity + its relations into a compact, injection-friendly snippet. */
function buildKgSnippet(entity: GraphSearchEntity): string {
  const head = entity.type ? `${entity.label} [${entity.type}]` : entity.label;
  const desc = entity.description ? `: ${entity.description}` : "";
  const out = (entity.outgoing || [])
    .slice(0, 4)
    .map((r) => `${r.relation_type} → ${r.target}${r.description ? ` (${r.description})` : ""}`);
  const inc = (entity.incoming || [])
    .slice(0, 3)
    .map((r) => `${r.source} ${r.relation_type} →${r.description ? ` (${r.description})` : ""}`);
  const rels = [...out, ...inc];
  const relPart = rels.length ? ` — Beziehungen: ${rels.join("; ")}` : "";
  return `${head}${desc}${relPart}`;
}

/** Vector-only search (internal) */
async function vectorSearch(
  query: string,
  scopes: string[],
  maxResults: number
): Promise<VectorSearchResult[]> {
  const k = Math.min(Math.max(1, Math.floor(maxResults)), 100);
  const scopesArg = scopes.filter(Boolean).join(",");

  const args: string[] = [
    "search",
    "--json",
    `-k`,
    String(k),
    "--scopes",
    scopesArg,
    query,
  ];

  let stdout: string;
  try {
    const { stdout: raw } = await execFileAsync(
      PYTHON_BIN,
      [VECTORDB_SCRIPT, ...args],
      {
        timeout: SEARCH_TIMEOUT_MS,
        maxBuffer: MAX_BUFFER_BYTES,
        env: SAFE_ENV,
        cwd: WIKIS_ROOT,
      }
    );
    stdout = raw;
  } catch (err: unknown) {
    console.error(
      `[activewiki] vector search failed: ${err instanceof Error ? err.message : String(err)}`
    );
    return [];
  }

  let hits: RawVectordbHit[];
  try {
    const parsed = JSON.parse(stdout.trim());
    if (!Array.isArray(parsed)) {
      console.error(
        `[activewiki] unexpected JSON shape from vectordb.py (expected array)`
      );
      return [];
    }
    hits = parsed;
  } catch (err: unknown) {
    console.error(
      `[activewiki] JSON parse error: ${err instanceof Error ? err.message : String(err)}`
    );
    return [];
  }

  return hits.map((hit) => ({
    id: hit.id,
    scope: hit.scope ?? "unknown",
    kind: hit.kind ?? "wiki",
    ref: hit.ref ?? "",
    section: hit.section,
    content: hit.content ?? "",
    score: typeof hit.score === "number" ? hit.score : 0,
  }));
}

/**
 * Knowledge graph lookup anchored on wiki pages (internal).
 *
 * Calls `vectordb.py graph pages` with the pages surfaced by the vector search,
 * returning the entities living on those pages plus their 1-hop relations.
 * This is the vector→graph bridge that makes the KG reachable from
 * natural-language queries. Non-blocking: degrades to [] on any failure.
 */
async function graphSearchByPages(
  pages: string[],
  scopes: string[]
): Promise<GraphSearchEntity[]> {
  if (pages.length === 0) return [];
  const scopesArg = scopes.filter(Boolean).join(",");

  const args: string[] = [
    "graph",
    "pages",
    "--json",
    "--pages",
    pages.join(","),
    "--scopes",
    scopesArg,
  ];

  let stdout: string;
  try {
    const { stdout: raw } = await execFileAsync(
      PYTHON_BIN,
      [VECTORDB_SCRIPT, ...args],
      {
        timeout: GRAPH_TIMEOUT_MS,
        maxBuffer: GRAPH_MAX_BUFFER,
        env: SAFE_ENV,
        cwd: WIKIS_ROOT,
      }
    );
    stdout = raw;
  } catch (err: unknown) {
    console.warn(
      `[activewiki] graph pages failed (non-blocking): ${err instanceof Error ? err.message : String(err)}`
    );
    return []; // Graceful degradation
  }

  let entities: GraphSearchEntity[];
  try {
    const parsed = JSON.parse(stdout.trim());
    if (!Array.isArray(parsed)) return [];
    entities = parsed;
  } catch (err: unknown) {
    console.warn(
      `[activewiki] graph JSON parse failed: ${err instanceof Error ? err.message : String(err)}`
    );
    return [];
  }

  return entities;
}

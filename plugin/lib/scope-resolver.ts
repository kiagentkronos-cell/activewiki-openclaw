import { readFileSync } from "node:fs";
import { join, resolve } from "node:path";

// ── Constants ────────────────────────────────────────────────────────────────

/** Allowed scope names. */
const VALID_SCOPES = new Set(["private", "family", "public"]);

/** Allowed scope for a given session key. */
export type WikiScope = "private" | "family" | "public";

// ── Types ────────────────────────────────────────────────────────────────────

interface ScopesEntry {
  name?: string;
  scopes: string[];
  sessionKeyPatterns: string[];
}

interface ScopesConfig {
  entries: ScopesEntry[];
  default?: {
    scopes: string[];
  };
}

// ── Config resolution ────────────────────────────────────────────────────────

/**
 * Resolve the path to scopes.json from environment or activewiki.json.
 *
 * Priority:
 *   1. ACTIVEWIKI_SCOPES_CONFIG env var (explicit override)
 *   2. scopes.scopes_config from activewiki.json (read via ACTIVEWIKI_CONFIG env)
 *   3. Fallback: resolve from wikis_root (ACTIVEWIKI_WIKIS_ROOT env) / config / scopes.json
 */
function resolveScopesConfigPath(): string {
  // 1. Direct env override
  const explicit = process.env.ACTIVEWIKI_SCOPES_CONFIG;
  if (explicit) return explicit;

  // 2. Read from activewiki.json
  const configPath = process.env.ACTIVEWIKI_CONFIG;
  if (configPath) {
    try {
      const cfg = JSON.parse(readFileSync(configPath, "utf-8"));
      const raw = cfg.scopes?.scopes_config;
      if (raw) return resolve(raw);
    } catch {
      // Ignore parse errors, fall through
    }
  }

  // 3. Fallback relative to wikis_root
  const wikisRoot = process.env.ACTIVEWIKI_WIKIS_ROOT;
  if (wikisRoot) {
    return join(wikisRoot, "config", "scopes.json");
  }

  // Last resort — should not happen in production
  throw new Error(
    "Cannot resolve scopes config. Set ACTIVEWIKI_CONFIG or ACTIVEWIKI_SCOPES_CONFIG."
  );
}

// ── Config (read fresh per call — no cache, file is tiny) ───────────────────

function loadScopesConfig(): ScopesConfig {
  const path = resolveScopesConfigPath();
  try {
    const raw = readFileSync(path, "utf-8");
    return JSON.parse(raw);
  } catch (err: unknown) {
    console.error(
      `[activewiki] Failed to load scopes config: ${err instanceof Error ? err.message : String(err)}`
    );
    return {
      entries: [],
      default: { scopes: ["public"] },
    };
  }
}

// ── Public API ───────────────────────────────────────────────────────────────

/**
 * Resolve allowed wiki scopes for a given agent session key.
 *
 * Matches the sessionKey against patterns in scopes.json using
 * substring matching. Returns the first match's scopes, or the default
 * scopes if no pattern matches.
 *
 * @param sessionKey - The OpenClaw session key (may be undefined)
 * @returns Array of allowed scope names (e.g. ["private", "family", "public"])
 */
export function resolveScopes(sessionKey?: string): WikiScope[] {
  if (!sessionKey) {
    return ["public"];
  }

  const config = loadScopesConfig();

  // Try each entry — first match wins
  for (const entry of config.entries) {
    for (const pattern of entry.sessionKeyPatterns) {
      if (sessionKey.includes(pattern)) {
        return entry.scopes
          .filter((s) => VALID_SCOPES.has(s))
          .map((s) => s as WikiScope);
      }
    }
  }

  // Active Memory Subagent keys look like:
  //   agent:main:discord:channel:12345:active-memory:abcdef
  //   agent:main:subagent:abcdef
  // We need to match against the parent session, not the subagent key.
  // Strip everything after ":active-memory:" or ":subagent:"
  // and retry matching against the parent portion.
  const parentKey = sessionKey.split(":active-memory:")[0].split(":subagent:")[0];
  for (const entry of config.entries) {
    for (const pattern of entry.sessionKeyPatterns) {
      if (parentKey.includes(pattern)) {
        return entry.scopes
          .filter((s) => VALID_SCOPES.has(s))
          .map((s) => s as WikiScope);
      }
    }
  }

  // Fallback: default scopes
  const defaultScopes = config.default?.scopes ?? ["public"];
  return defaultScopes
    .filter((s) => VALID_SCOPES.has(s))
    .map((s) => s as WikiScope);
}

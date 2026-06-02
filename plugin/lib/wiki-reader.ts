import { readFileSync } from "node:fs";
import { resolve, join } from "node:path";
import type { WikiScope } from "./scope-resolver.js";

// ── Config resolution ────────────────────────────────────────────────────────

/**
 * Resolve the wiki root directory from environment or activewiki.json.
 * Priority: ACTIVEWIKI_WIKIS_ROOT env > read from ACTIVEWIKI_CONFIG > throw
 */
function resolveWikiRoot(): string {
  if (process.env.ACTIVEWIKI_WIKIS_ROOT) {
    return resolve(process.env.ACTIVEWIKI_WIKIS_ROOT);
  }

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
    "Cannot resolve wiki root. Set ACTIVEWIKI_WIKIS_ROOT or ACTIVEWIKI_CONFIG."
  );
}

// ── Constants ────────────────────────────────────────────────────────────────

/** Base directory containing wiki/ and sources/ subdirectories. */
const WIKI_ROOT = resolveWikiRoot();

/** Regex that slugs must fully match — prevents path traversal. */
const SLUG_REGEX = /^[a-z0-9\-]{1,100}$/;

/** Subdirectories to search (in priority order). */
const SUBDIRS: readonly string[] = ["wiki", "sources"];

// ── Helpers ──────────────────────────────────────────────────────────────────

/**
 * Sanitize and resolve a file path, ensuring it stays within the allowed
 * wiki root + scope subdirectory.
 *
 * Returns `null` if the path would escape the allowed prefix.
 */
function safeResolve(scope: WikiScope, subdir: string, slug: string): string | null {
  const prefix = resolve(WIKI_ROOT, subdir, scope);
  const candidate = resolve(prefix, `${slug}.md`);

  // Prefix check: the resolved path must start with the allowed prefix
  if (!candidate.startsWith(prefix + "/") && candidate !== prefix) {
    return null;
  }

  return candidate;
}

// ── Public API ───────────────────────────────────────────────────────────────

/** Result of reading a wiki page with line slicing. */
export interface WikiPageResult {
  content: string;
  path: string;
}

/**
 * Read a wiki page by scope + slug, applying line-based slicing.
 *
 * Searches both `wiki/<scope>/` and `sources/<scope>/` subdirectories.
 * Validates slug against a strict regex and protects against path traversal.
 *
 * @param scope - One of "private" | "family" | "public"
 * @param slug - URL-safe identifier (lowercase, hyphens, digits)
 * @param fromLine - 1-based starting line
 * @param lineCount - Maximum number of lines to return
 * @returns Page content + resolved path, or null if not found
 */
export function getWikiPage(
  scope: WikiScope,
  slug: string,
  fromLine: number,
  lineCount: number
): WikiPageResult | null {
  // Validate slug — reject anything that could be path traversal
  if (!SLUG_REGEX.test(slug)) {
    console.error(
      `[activewiki] Invalid slug rejected: "${slug}"`
    );
    return null;
  }

  // Normalize line params
  const start = Math.max(0, Math.floor(fromLine) - 1); // convert to 0-based
  const maxLines = Math.max(1, Math.min(lineCount, 5000));

  // Try each subdirectory
  for (const subdir of SUBDIRS) {
    const filePath = safeResolve(scope, subdir, slug);
    if (!filePath) {
      continue;
    }

    try {
      const raw = readFileSync(filePath, "utf-8");
      const lines = raw.split("\n");
      const sliced = lines.slice(start, start + maxLines);
      return {
        content: sliced.join("\n"),
        path: filePath,
      };
    } catch {
      // File doesn't exist or isn't readable — try next subdir
      continue;
    }
  }

  return null;
}

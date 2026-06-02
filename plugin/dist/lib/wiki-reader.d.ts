import type { WikiScope } from "./scope-resolver.js";
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
export declare function getWikiPage(scope: WikiScope, slug: string, fromLine: number, lineCount: number): WikiPageResult | null;

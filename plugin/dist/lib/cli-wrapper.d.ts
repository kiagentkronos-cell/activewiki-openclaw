import type { VectorSearchResult } from "./types.js";
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
export declare function searchWiki(query: string, scopes: string[], maxResults: number): Promise<VectorSearchResult[]>;

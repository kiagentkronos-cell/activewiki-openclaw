import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";
import { searchWiki } from "./lib/cli-wrapper.js";
import { resolveScopes } from "./lib/scope-resolver.js";
import { getWikiPage } from "./lib/wiki-reader.js";
export default definePluginEntry({
    id: "activewiki",
    name: "ActiveWiki",
    description: "Vector search + knowledge graph over a configurable wiki corpus (ActiveWiki)",
    register(api) {
        api.registerMemoryCorpusSupplement({
            async search(params) {
                // Determine which wiki scopes this session is allowed to access
                const scopes = resolveScopes(params.agentSessionKey);
                if (scopes.length === 0) {
                    return [];
                }
                // Delegate vector search to the Python script
                const results = await searchWiki(params.query, scopes, params.maxResults ?? 5);
                // Map to MemoryCorpusSearchResult format expected by OpenClaw
                return results.map((r) => ({
                    corpus: "custom-memory",
                    path: r.ref,
                    title: r.section
                        ? r.section.replace(/^_preamble_$|_/g, "").replace(/-/g, " ")
                        : undefined,
                    kind: r.kind,
                    score: r.score,
                    snippet: r.content.substring(0, 300),
                    id: String(r.id),
                    source: r.scope,
                }));
            },
            async get(params) {
                // Validate lookup parameter
                if (!params?.lookup || typeof params.lookup !== "string") {
                    return null;
                }
                // Parse lookup: "scope/slug" or just "slug" (defaults to public)
                const parts = params.lookup.split("/");
                let scope;
                let slug;
                if (parts.length === 2) {
                    scope = parts[0].toLowerCase();
                    slug = parts[1].toLowerCase();
                }
                else {
                    scope = "public";
                    slug = parts[0].toLowerCase();
                }
                // Restrict to known scopes
                const validScopes = ["private", "family", "public"];
                if (!validScopes.includes(scope)) {
                    return null;
                }
                // Authorize: check if this session is allowed to read the requested scope
                const allowed = resolveScopes(params.agentSessionKey);
                if (!allowed.some((s) => s === scope)) {
                    return null;
                }
                // Read the wiki page with line slicing
                const page = await getWikiPage(scope, slug, params.fromLine ?? 1, params.lineCount ?? 200);
                if (!page) {
                    return null;
                }
                return {
                    corpus: "custom-memory",
                    path: page.path,
                    content: page.content,
                    fromLine: params.fromLine ?? 1,
                    lineCount: params.lineCount ?? 200,
                    source: scope,
                };
            },
        });
    },
});

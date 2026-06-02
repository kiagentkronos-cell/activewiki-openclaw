/** Allowed scope for a given session key. */
export type WikiScope = "private" | "family" | "public";
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
export declare function resolveScopes(sessionKey?: string): WikiScope[];

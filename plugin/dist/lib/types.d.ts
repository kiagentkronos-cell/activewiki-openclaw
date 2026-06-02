/** Internal result shape from vectordb.py search. */
export interface VectorSearchResult {
    id: number;
    scope: string;
    kind: string;
    ref: string;
    section?: string;
    content: string;
    score: number;
}

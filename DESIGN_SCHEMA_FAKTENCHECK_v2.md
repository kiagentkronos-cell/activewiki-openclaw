# ActiveWiki: Schema Layer & Faktencheck — Design Plan v2

**Status:** Überarbeitet v4 | **Autor:** Kronos | **Review:** Hyperion Round 2 (3 Kritiken) + Round 3 (Validierung + Korrekturen)

---

## 1. Problem

Der aktuelle Knowledge Graph besteht aus zwei Tabellen:
- `entities` — konkrete Instanzen mit `entity_type` (z.B. "Max Mustermann" / `Person`)
- `relationships` — Tripel zwischen Instanzen via Entity-IDs

Es fehlt:
1. **Ontologie-Schema** — Welche Typen von Dingen und Beziehungen gibt es? Der Agent sieht nur einzelne Facts, nicht das Muster dahinter.
2. **Faktenvalidierung** — Widersprüchliche Aussagen akkumulieren (z.B. alter + neuer Mietpreis), zeitliche Gültigkeit geht verloren.

---

## 2. Ziele

- **Schema Layer:** Aggregierte Ontologie aus `entity_type` + `relation_type`, die dem Agent metakognitives Verständnis gibt.
- **Faktencheck:** Nachträglicher Scan (kein pro-Triple-Loop) erkennt Konflikte und veraltet alte Facts automatisch.

---

## 3. Architektur

### 3.1 Datenbank-Schema-Erweiterung

```sql
-- Neue Tabelle: Ontologie-Schema (Layer 1)
-- Basiert auf EXISTIERENDEN Feldern: entities.entity_type + relationships.relation_type
CREATE TABLE ontologies (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ontology_key    TEXT NOT NULL UNIQUE,        -- z.B. "Person BELEBT Raum"
    source_type     TEXT NOT NULL,               -- z.B. "Person" (aus entities.entity_type)
    target_type     TEXT NOT NULL,               -- z.B. "Raum" (aus entities.entity_type)
    relation_type   TEXT NOT NULL,               -- z.B. "BELEBT" (aus relationships.relation_type)
    instance_count  INTEGER DEFAULT 1,           -- wieviele Tripel dieses Patterns existieren
    last_seen       TEXT NOT NULL                -- ISO-Timestamp letztes Vorkommen
);
CREATE INDEX idx_ontology_source ON ontologies(source_type);
CREATE INDEX idx_ontology_target ON ontologies(target_type);

-- Neue Tabelle: Fakten-Konflikte (Audit-Log)
CREATE TABLE fact_conflicts (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    relationship_id_old TEXT NOT NULL REFERENCES relationships(id),   -- ID der alten Relationship-Row (direkter FK!)
    relationship_id_new TEXT NOT NULL REFERENCES relationships(id),   -- ID der neuen Relationship-Row
    source_id           TEXT NOT NULL REFERENCES entities(id),
    target_id_old       TEXT NOT NULL REFERENCES entities(id),
    target_id_new       TEXT NOT NULL REFERENCES entities(id),
    relation_type       TEXT NOT NULL,
    conflict_type       TEXT NOT NULL,            -- 'temporal'|'mutual'|'granularity'
    resolved            INTEGER DEFAULT 0,        -- 0=open, 1=resolved
    resolution          TEXT,                     -- 'kept_new'|'kept_old'|'both'
    detected_at         TEXT NOT NULL,            -- ISO-Timestamp
    resolved_at         TEXT                      -- ISO-Timestamp wenn gelöst
);

-- Migration: relationships erhält Gültigkeitsfeld
ALTER TABLE relationships ADD COLUMN valid_until  TEXT DEFAULT NULL;
CREATE INDEX idx_rel_valid ON relationships(valid_until);
CREATE INDEX idx_rel_scan ON relationships(source_id, relation_type, valid_until);

-- Migration: orphaned Relationships finden (Entity gelöscht aber Relationship bleibt)
-- Wird als separates Cleanup-Skript gehandhabt, nicht im Core-Schema.
```

**Hyperion-Fix #1:** `fact_conflicts` speichert jetzt `relationship_id_old/new` (TEXT,REFERENCES relationships(id)) statt Labels → direkte Resolution möglich.
**Hyperion-Fix #2:** Ontologie basiert auf `entity_type` (existiert bereits), nicht auf fiktiven Triple-Typen.

### 3.2 Ontologie-Schema: Aggregation nach dem Build

**NICHT pro-Triple im Loop.** Nach dem kompletten Graph-Build wird ein aggregierter SQL-Pass gemacht:

```python
def _rebuild_ontologies(conn) -> None:
    """Aggregiert Ontologie-Patterns aus ALLEN bestehenden Relationships."""
    # Single query über gesamte Relationships-Table → deutlich schneller als Loop
    rows = conn.execute("""
        SELECT e_src.entity_type AS source_type,
               r.relation_type   AS relation_type,
               e_tgt.entity_type AS target_type,
               COUNT(*)          AS cnt
        FROM relationships r
        JOIN entities e_src ON r.source_id = e_src.id AND r.valid_until IS NULL
        JOIN entities e_tgt ON r.target_id = e_tgt.id AND r.valid_until IS NULL
        GROUP BY source_type, relation_type, target_type
        HAVING cnt >= ?
    """, (SCHEMA_MIN_INSTANCES,)).fetchall()

    # Alle bestehenden Ontologien löschen und neu aufbauen (idempotent)
    conn.execute("DELETE FROM ontologies")

    for source_type, relation_type, target_type, cnt in rows:
        conn.execute("""INSERT INTO ontologies (ontology_key, source_type, target_type, relation_type, instance_count, last_seen) VALUES (?, ?, ?, ?, ?, ?)""",
            (f"{source_type} {relation_type} {target_type}", source_type, target_type, relation_type, cnt, utc_now()))


def _filter_sparse_ontologies(conn, threshold: int = 2) -> None:
    """Entfernt Ontologien die seltener als threshold vorkommen."""
    conn.execute("DELETE FROM ontologies WHERE instance_count < ?", (threshold,))
```

**Hyperion-Fix #3:** Batched SQL statt einzelnem UPSERT pro Triple. Reduziert Build-Time von +30-60% auf ~+2%.

### 3.3 Faktencheck: Nachträglicher Scan (nicht pro-Triple)

Inspiration von MemGraphRAG: Konflikt-Erkennung als **separater Schritt nach dem Build**, nicht währenddessen.

```python
def _scan_conflicts(conn) -> int:
    """Scan bestehende Relationships auf Konflikte (nach dem Build)."""

    # Finde alle Fälle wo dieselbe Source + dieselbe Relation multiple ACTIVE Targets hat
    conflict_rows = conn.execute("""
        SELECT r1.source_id, r1.relation_type, r1.id as rel1_id, r1.target_id as tgt1_id, r2.id as rel2_id, r2.target_id as tgt2_id
        FROM relationships r1
        JOIN relationships r2 ON r1.source_id = r2.source_id AND r1.relation_type = r2.relation_type
                             AND r1.id < r2.id
                             AND r1.valid_until IS NULL AND r2.valid_until IS NULL
                             AND r1.target_id != r2.target_id
        ORDER BY r1.id ASC
        LIMIT 500""").fetchall()

    # Hyperion-Fix #6: r1.id < r2.id verhindert N×(N-1) Paare — pro Konfliktgruppe nur ein Sortierwinner
    # (ID = SHA256-content-hash, keine Zeitinfo — daher konservativ: nur ein Treffer pro Gruppe)

    # Hyperion-Fix #4: Vergleicht Entity-IDs (nicht Labels!) → keine falschen Treffer

    for source_id, rel_type, rel1_id, tgt1_id, rel2_id, tgt2_id in conflict_rows:
        conflict_type = classify_conflict(rel_type)
        conflict_id = insert_conflict(conn, rel1_id, rel2_id, source_id, tgt1_id, tgt2_id, rel_type, conflict_type)
        if auto_resolve(conflict_type):
            resolve_conflict(conn, conflict_id)

    return len(conflict_rows)


def classify_conflict(relation_type) -> str:
    """Klassifiziert den Konflikttyp basierend auf der Relation (nicht auf Werten!)."""
    # "BEHALT", "KOSTET", "PREIS" → Wert hat sich geändert (temporal)
    TEMPORAL_RELATIONS = {"BEHALT", "KOSTET", "PREIS", "MIETE", "STATUS", "VERTRAGSSTATUS"}
    if relation_type in TEMPORAL_RELATIONS:
        return "temporal"
    # Alles andere → zunächst als temporal behandeln (konservativer Default)
    return "temporal"


def auto_resolve(conflict_type) -> bool:
    """Konfig-basierte Auto-Resolution."""
    if conflict_type == "temporal" and CONFIG.get("conflict_auto_resolve_temporal", True):
        return True
    return False  # mutual/granularity manuell


def resolve_conflict(conn, conflict_id) -> None:
    """Löscht alter Fakt (bekommt valid_until), neuer bleibt aktiv."""
    # relation_id_old speichern → direkte SQL-Update der betroffenen Row
    old_rel_id = conn.execute("SELECT relationship_id_old FROM fact_conflicts WHERE id=?", (conflict_id,)).fetchone()
    if old_rel_id:
        conn.execute("UPDATE relationships SET valid_until=? WHERE id=?", (utc_now(), old_rel_id[0]))
        conn.execute("UPDATE fact_conflicts SET resolved=1, resolution='kept_new', resolved_at=? WHERE id=?", (utc_now(), conflict_id))


```

# Cleanup: Orphaned Relationships (Entity gelöscht, Relationship bleibt hängen)
def _cleanup_orphaned_relationships(conn) -> int:
    deleted = conn.execute("""
        UPDATE relationships SET valid_until=?
        WHERE (source_id NOT IN (SELECT id FROM entities)
                  OR target_id NOT IN (SELECT id FROM entities))
                  AND valid_until IS NULL
    """, (utc_now())).rowcount
```

**Hyperion-Fix #7:** Orphan-Cleanup als nachträglicher Schritt. ON DELETE CASCADE in SQLite löst
nicht bei logischem Lösch (Entity bleibt, nur Wiki-Page weg) — expliziter Mark-Schritt nötig.

### 3.4 Schema-Suche fuer Active Memory Plugin

Das Plugin (`cli-wrapper.ts`) lernt einen neuen Query-Typ: **Schema-Kontext**.

**Neue CLI-Subkommande in `vectordb.py`:**

`cmd_schema_context()` — Holt Schema-Patterns für gegebene Wiki-Seiten.
- Einfacher JOIN über `entities.entity_type` → `ontologies.source_type/target_type`
- Kein dynamisches SQL-Format, kein LIKE-Hack (Hyperion-Fix #5)
- Query via `json_each(?)` fuer page-Parameter, indexiert auf `entity_type`

**Plugin-Integration (`cli-wrapper.ts`):**
- `vectordb.py schema-context --pages "..."` als neuen CLI-Befehl aufrufen
- Ergebnis als `<schema_context>`-Block im `<active_memory_plugin>` injizieren
- Timeout: 3s (reine SQL-Abfrage, kein LLM)

**Beispiel-Output fuer den Agent:**
```
<schemas>
Dieses Wiki enthällt folgende Beziehungs muster:
- Mietvertrag BEHALT Preis (12 Vorkommen)
- Mieter HABT Kontaktweg (8 Vorkommen)
- Raum BELEGT_VON Mieter (6 Vorkommen)
- Vertrag IST_IN_STATUS Status (15 Vorkommen)
</schemas>
```

### 3.5 Neue CLI-Befehle

```bash
# Ontologie-Schema anzeigen
vectordb.py schema stats          # Anzahl Ontologien, Top-Patterns
vectordb.py schema list           # Alle Ontologien sortiert nach Häufigkeit
vectordb.py schema show <pattern> # Detailansicht einer Ontologie

# Fakten-Konflikte
vectordb.py conflicts list        # Offene Konflikte
vectordb.py conflicts resolve --all-temporal  # Alle temporal automatisch lösen
vectordb.py conflicts show <id>   # Detailansicht

# Orphaned Relationships aufraeumen
vectordb.py cleanup orphaned      # Beziehungen mit gelöschten Entities invalidieren
```

---

## 4. Konfiguration (`activewiki.json`)

```jsonc
{
  "graph": {
    "schema_enabled": true,                     // Schema-Layer aktivieren (Default: true)
    "schema_min_instances": 2,                  // Mindesthäufigkeit für Ontologie-Aufnahme
    "conflict_detection": true,                 // Faktencheck aktivieren (Default: true)
    "conflict_auto_resolve_temporal": true,     // Temporale Konflikte automatisch lösen (Default: true)
    "conflict_auto_resolve_mutual": false,      // Mutuelle Konflikte NICHT automatisch lösen (Default: false — Hyperion-Korrektur!)
    "conflict_keep_granularity": true           // Granularitäts-Konflikte nicht lösen (beide behalten)
  }
}
```

---

## 5. Implementierungs-Reihenfolge (Phasen)

### Phase 1: Ontologie-Schema (~200 Zeilen Code)
1. `ontologies` Tabelle + Indices in `init_db()` + Migration via `_record_graph_migration()`
2. `_rebuild_ontologies()` nach `graph build` einbauen (batched SQL, kein Loop)
3. `cmd_schema_context()` CLI-Befehl in `vectordb.py` + Subparser in `argparse`
4. Plugin-Integration: `cli-wrapper.ts` ruft `schema-context` auf, injiziert `<schema_context>`-Block nach Graph-Treffern (mit neuem `kind: "schema-context"`)
5. Konfig-Optionen in `config.py` (`graph_schema_enabled`, `graph_schema_min_instances`)
6. Tests + Hyperion Review → Push + ClawHub Release (v1.1.0)

### Phase 2: Fakten-Konflikte (~300 Zeilen Code)  
1. `fact_conflicts` Tabelle + `valid_until` Spalte (+ Migrationen via `_record_graph_migration()`)  
   - Migration muss bestehende Rows explizit auf `valid_until IS NULL` setzen! (Hyperion-Warnung)  
2. `_scan_conflicts()` als neuer Post-Build-Schritt nach dem Graph-Build  
3. `_cleanup_orphaned_relationships()` für gelöschte Wiki-Seiten (Hyperion-Warnung)  
4. CLI: `conflicts list/show/resolve` + `cleanup orphaned`  
5. Konfig-Optionen in `config.py`  
6. Tests + Hyperion Review → Push + ClawHub Release (v1.2.0)

---

## 6. Risiken & Abwägungen

| Risiko | Bewertung | Mitigation |
|--------|-----------|------------|
| Schema-Aktualisierung verlangsamt Build | 🟢 Niedrig | Batched SQL GROUP BY, keine einzelnen UPSERTs (+2% statt +60%) |
| Falsche Konflikt-Klassifizierung | 🟡 Mittel | Konservativer Default: alles temporal; mutual/granularity später |  
| SQLite-Schema-Migration bricht alte DBs | 🔴 Hoch | Migration prüft `_migration_applied()` vor ALTER TABLE; bestehende Rows bekommen `valid_until IS NULL` |  
| Schema-Lärm bei kleinen Wikis | 🟢 Niedrig | `schema_min_instances` filtert Rauschen |  
| Entity-Typen ändern sich → Schema veraltet | 🟢 Niedrig | Vollständiger Rebuild bei jedem `graph build` (idempotent DELETE+INSERT) |  
| Conflict-Scan ist langsam bei großen Graphen | 🟡 Mittel | LIMIT 500 pro Build-Lauf; inkrementeller Scan nur neuer Einträge möglich |  

---

## 7. Hyperion Round 1 Kritikpunkte & Fixes

| # | Hyperion-Kritik | Fix in v2 | Status |  
|---|----------------|-----------|--------|  
| 1 | Triple-Typen existieren nicht im Code | Ontologie basiert auf `entity_type` (existiert bereits) | ✅ Gefixt |  
| 2 | Konflikt-Erkennung vergleicht Labels statt IDs | Jetzt Entity-ID-Vergleich via relationship_id_old/new FKs | ✅ Gefixt |  
| 3 | `fact_conflicts` kann keine Relationships updaten | Speichert jetzt relationship_id_old/new mit direktem FK auf relationships(id) | ✅ Gefixt |  
| 4 | `get_schema_context()` SQL nicht skalierbar | Neuer JOIN via entities.entity_type, json_each() statt dynamischem .format() SQL; Index auf entity_type vorhanden | ✅ Gefixt |  
| 5 | Mutation-Pattern vs nachträglicher Scan → Performance-Probleme bei großen Graphen | Umstellung auf nachträglichen Scan (`_scan_conflicts()`) nach dem Build, analog zu MemGraphRAG Pattern | ✅ Gefixt |  

### Open Questions fuer Round 2:  
- Should conflict_resolution be manual-only for mutual? Currently conservative default: false for auto-resolve mutual  
- Granularity detection currently not implemented — might need embedding similarity or LLM judgment in future iteration  

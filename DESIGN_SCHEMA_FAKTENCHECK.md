# ActiveWiki: Schema Layer & Faktencheck — Design Plan

**Status:** Entwurf | **Autor:** Kronos | **Review:** Hyperion (ausstehend)

---

## 1. Problem

Der aktuelle Knowledge Graph besteht aus zwei Tabellen:
- `entities` — konkrete Instanzen (z.B. "Max Mustermann", "DGX Spark")
- `relationships` — Tripel zwischen Instanzen

Es fehlt:
1. **Ontologie-Schema** — Welche Typen von Dingen und Beziehungen gibt es? Der Agent sieht nur einzelne Facts, nicht das Muster dahinter.
2. **Faktenvalidierung** — Widersprüchliche Aussagen akkumulieren (z.B. alter + neuer Mietpreis), zeitliche Gültigkeit geht verloren.

---

## 2. Ziele

- **Schema Layer:** Aggregierte Ontologie, die dem Agent metakognitives Verständnis gibt ("in diesem Wiki existieren Mietverträge mit Preis, Raum, Mieter").
- **Faktencheck:** Automatische Erkennung von Konflikten bei neuen Tripeln + Löschmechanismus für veraltete Facts.

---

## 3. Architektur

### 3.1 Datenbank-Schema-Erweiterung

```sql
-- Neue Tabelle: Ontologie-Schema (Layer 1)
CREATE TABLE ontologies (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ontology_key    TEXT NOT NULL UNIQUE,        -- z.B. "Mietvertrag BEHALT Preis"
    source_types    TEXT NOT NULL,               -- z.B. "Person -> Vertrag -> Float"
    relation_type   TEXT NOT NULL,               -- z.B. "BEHALT"
    instance_count  INTEGER DEFAULT 1,           -- wieviele Tripel dieses Patterns existieren
    last_seen       TEXT NOT NULL                -- ISO-Timestamp letztes Vorkommen
);

-- Neue Tabelle: Fakten-Konflikte (Audit-Log)
CREATE TABLE fact_conflicts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_a_id     TEXT NOT NULL REFERENCES entities(id),
    entity_b_id     TEXT NOT NULL REFERENCES entities(id),
    relation_type   TEXT NOT NULL,
    old_value       TEXT NOT NULL,               -- alter Fakt (Label der Target-Entity)
    new_value       TEXT NOT NULL,               -- neuer Fakt
    conflict_type   TEXT NOT NULL,               -- 'temporal'|'mutual'|'granularity'
    resolved        INTEGER DEFAULT 0,           -- 0=open, 1=resolved
    resolution      TEXT,                        -- 'kept_new'|'kept_old'|'merged'
    detected_at     TEXT NOT NULL,               -- ISO-Timestamp
    resolved_at     TEXT                         -- ISO-Timestamp wenn gelöst
);

-- Migration: relationships erhält gültigkeitsfeld
ALTER TABLE relationships ADD COLUMN valid_from  TEXT;  -- ISO-Timestamp oder NULL=immer
ALTER TABLE relationships ADD COLUMN valid_until  TEXT;  -- ISO-Timestamp oder NULL=aktuell
ALTER TABLE relationships ADD COLUMN conflict_id  INTEGER REFERENCES fact_conflicts(id);
```

### 3.2 Ontologie-Schema: Wie es gebaut wird

**Bestehender Graph-Build-Prozess bleibt erhalten.** Nach dem Extrahieren von Tripeln wird ein neuer Schritt eingefügt:

```python
def _update_ontologies(conn, triples: list[dict]) -> None:
    """Aggregiert Ontologie-Patterns aus neuen Tripeln."""
    for triple in triples:
        key = f"{triple['head_type']} {triple['relation']} {triple['tail_type']}"
        cursor = conn.execute("""
            INSERT INTO ontologies (ontology_key, source_types, relation_type, instance_count, last_seen)
            VALUES (?, ?, ?, 1, ?)
            ON CONFLICT(ontology_key) DO UPDATE SET
                instance_count = instance_count + 1,
                last_seen = excluded.last_seen
        """, (key, f"{triple['head_type']} -> {triple['tail_type']}", triple['relation'], utc_now()))

def _filter_sparse_ontologies(conn, threshold: int = 2) -> None:
    """Entfernt Ontologien die seltener als threshold vorkommen."""
    conn.execute("DELETE FROM ontologies WHERE instance_count < ?", (threshold,))
```

**Parameter:** `graph.schema_min_instances` in `activewiki.json` (Default: 2) — Ontologien mit weniger Vorkommen werden ignoriert (Rauschen).

### 3.3 Faktencheck: Konflikt-Erkennung

Bei jedem `graph build` wird jedes neue Tripel gegen bestehende Fakten geprüft:

```python
def _detect_conflict(conn, head_id, relation_type, new_tail_id):
    """Prüft ob ein neuer Fakt mit bestehenden Fakten kollidiert."""

    # Hole bestehenden Target derselben Relation von derselben Quelle
    existing = conn.execute("""
        SELECT e.label FROM relationships r JOIN entities e ON r.target_id = e.id
        WHERE r.source_id = ? AND r.relation_type = ? AND r.valid_until IS NULL
    """, (head_id, relation_type)).fetchone()

    if existing is None or existing[0] == new_tail_label:
        return None  # Kein Konflikt

    # Konflikt erkannt — Klassifizieren:
    conflict_type = classify_conflict(relation_type, existing[0], new_tail_label)

    # Temporäre Speicherung bis Resolution:
    conflict_id = insert_conflict(conn, ...)

    return conflict_id


def classify_conflict(relation_type, old_value, new_value) -> str:
    """Klassifiziert den Konflikttyp."""

    # Mutual: direkte Gegensätze (z.B. STATUS "AKTIV" vs "BEENDET")
    if relation_type in ("STATUS", "TYPE", "STATE", "IST"):
        return "mutual"

    # Granularity: selbes Thema, unterschiedliches Detailniveau
    if old_value in new_value or new_value in old_value:
        return "granularity"

    # Default: temporal (Wert hat sich geändert)
    return "temporal"


def resolve_temporal_conflict(conn, conflict_id):
    """Temporale Konflikte: alter Fakt bekommt valid_until."""
    conn.execute("""UPDATE relationships SET valid_until = ? WHERE ...""", (utc_now(),))


def resolve_mutual_conflict(conn, conflict_id):
    """Mutuelle Konflikte: neuer Fakt gewinnt (letztes Schreiben zählt)."""
    resolve_temporal_conflict(conn, conflict_id)  # gleiches Verhalten initially


def resolve_granularity_conflict(conn, conflict_id):
    """Granularitäts-Konflikte: beide Facts behalten."""
    pass  # Keine Löschung — beides ist wahr auf unterschiedlichem Level. Granularity-Konflikte bleiben im Audit-Log.
```

### 3.4 Schema-Suche für Active Memory Plugin

Das Plugin (`cli-wrapper.ts`) lernt einen neuen Query-Typ: **Schema-Kontext**.

Wenn die Vektorsuche Top-Pages identifiziert:

```python
def get_schema_context(conn, wiki_pages: list[str]) -> dict:
    """Holt relevante Ontologie-Patterns für gegebene Wiki-Seiten."""
    patterns = conn.execute("""
        SELECT o.* FROM ontologies o
        JOIN entities e ON e.entity_type LIKE '%' || substr(o.source_types, 1, instr(o.source_types, ' -> ')) || '%'
        JOIN (SELECT DISTINCT wiki_page FROM chunks WHERE wiki_page IN ({}) ) cp
        ON e.wiki_page = cp.wiki_page
        WHERE o.instance_count >= 2
        ORDER BY o.instance_count DESC
        LIMIT 10
    """.format(','.join('?' * len(wiki_pages))), wiki_pages).fetchall()
    return [row_to_dict(r) for r in patterns]
```

**Plugin-Integration (`cli-wrapper.ts`):**
- Nach Vektor+Graph-Bridge: `get_schema_context()` für Top-Pages aufrufen
- Als `<schema_context>`-Block im `<active_memory_plugin>` injizieren
- Timeout: 3s (reine SQL-Abfrage, kein LLM)

**Beispiel-Output für den Agent:**
```
<schemas>
Dieses Wiki enthält folgende Beziehungsmuster:
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
```

---

## 4. Konfiguration (`activewiki.json`)

```json
{
  "graph": {
    "schema_enabled": true,                     // Schema-Layer aktivieren
    "schema_min_instances": 2,                  // Mindesthäufigkeit für Ontologie-Aufnahme
    "conflict_detection": true,                 // Faktencheck aktivieren
    "conflict_auto_resolve_temporal": true,     // Temporale Konflikte automatisch lösen
    "conflict_auto_resolve_mutual": true,       // Mutuelle Konflikte automatisch lösen (neuer Wert gewinnt)
    "conflict_keep_granularity": true           // Granularitäts-Konflikte nicht lösen (beide behalten)
  }
}
```

---

## 5. Implementierungs-Reihenfolge (Phasen)

### Phase 1: Ontologie-Schema (ca. 200 Zeilen Code)
1. `ontologies` Tabelle in `init_db()` hinzufügen (+ Migration)
2. `_update_ontologies()` in `_extract_entities_from_page()` einbauen
3. `_filter_sparse_ontologies()` am Ende des Builds aufrufen
4. `get_schema_context()` + Plugin-Integration in `cli-wrapper.ts`
5. CLI: `schema stats/list/show`
6. Konfig-Optionen in `config.py`
7. Tests + Hyperion Review → Push + ClawHub Release (v1.1.0)

### Phase 2: Fakten-Konflikte (ca. 300 Zeilen Code)
1. `fact_conflicts` Tabelle + `valid_from/until` Spalten (+ Migrationen)
2. `_detect_conflict()` + `classify_conflict()` + Auto-Resolution in Build-Pipeline integrieren
3. CLI: `conflicts list/show/resolve`
4. Konfig-Optionen in `config.py`
5. Tests + Hyperion Review → Push + ClawHub Release (v1.2.0)

---

## 6. Risiken & Abwägungen

| Risiko | Bewertung | Mitigation |
|--------|-----------|------------|
| Schema-Aktualisierung verlangsamt Build | 🟡 Mittel | Pure SQL UPSERTs, keine externen Calls |
| Falsche Konflikt-Klassifizierung | 🟡 Mittel | Initial nur temporal/mutual; granularity später |
| SQLite-Schema-Migration bricht alte DBs | 🔴 Hoch | Migration prüft `_migration_applied()` vor ALTER TABLE |
| Schema-Lärm bei kleinen Wikis | 🟢 Niedrig | `schema_min_instances` filtert Rauschen |
| Entity-Typen ändern sich → Schema veraltet | 🟡 Mittel | Periodischer Rebuild der Ontologie bei graph build |


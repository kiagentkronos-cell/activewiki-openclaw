# ActiveWiki: Phase 2 — Confidence Tags, Why Nodes, HTML Visualisierung

**Status:** Entwurf | **Autor:** Kronos | **Review:** Hyperion (3 Runden)

---

## 1. Confidence Tags

### Problem
Jede Relationship wird gleich gewichtet — aber eine direkte Referenz in einer Wiki-Page (z.B. "Max wohnt in Gruenwald") ist eine harte Tatsache, während "Owner ist wohl in der Immobilienbranche" eine LLM-Inferenz ist. Der Agent kann das nicht unterscheiden.

### Lösung
`confidence` Spalte auf `relationships`: `extracted`, `inferred`, `ambiguous`

**Wie bestimmt wird:**
- Der LLM-Prompt für `_ollama_extract()` bekommt eine neue Anweisung: jede Tripel bekommt ein Confidence-Level.
- `extracted` = direkte Aussage im Text ("Mietpreis: 720€")
- `inferred` = logische Schlussfolgerung ("Owner hat Immobilie → er investiert")  
- `ambiguous` = mehrdeutig oder schwach belegt ("Simba könnte ein Hund sein")

**Prompt-Spezifikation:**
- Confidence als separates JSON-Feld: `{"source": "x", "target": "y", "type": "BESITZT", "confidence": "extracted"}`
- Mindestens 3 Few-Shot-Beispiele pro Level in den Good Examples
- Nur bei einfachen, klaren Fällen extrahieren — komplexe Mehrdeutigkeiten → `ambiguous`

**Schema:**
```sql
ALTER TABLE relationships ADD COLUMN confidence TEXT DEFAULT 'inferred';
```

**Migration:** Alle bestehenden Relationships bekommen `confidence = 'inferred'` (konservativer Default — alles bisherige wurde vom LLM extrahiert).

**⚠️ Historische Daten als 'inferred' (TODO):** Nach der Migration sind alle alten Facts als `inferred` markiert — auch harte Fakten wie Mietpreise aus Mietverträgen.

**⚠️ Rel-ID-Stabilität:** Das Confidence-Upgrade-Design setzt konsistente Entitäts-Namensgebung über Prompt-Versionen hinweg voraus. Falls "Haus A" später "Gruenwald" heißt, ändert sich die Rel-ID und es entsteht eine parallele Kante statt eines Upgrades. Späterer `--reclassify-historical` Befehl könnte alte Daten nochmal durch den neuen Prompt schicken für korrekte Confidence-Klassifizierung.

**⚠️ Rel-ID Konflikt (Hyperion-Kritik):** Die Rel-ID = SHA256 von `{src_id}::{tgt_id}::{rel_type}` — gleiche Relation mit anderer Confidence wird ignoriert.

**Fix: SELECT-first Pattern in Transaktion** (nicht INSERT OR IGNORE + nachträglich UPDATE):
```python
conn.execute("BEGIN")
try:
    existing = conn.execute("SELECT confidence FROM relationships WHERE id=?", (rel_id,)).fetchone()
    if existing is None:
        conn.execute(INSERT_WITH_CONFIDENCE, (rel_id, src, tgt, rel_type, desc, confidence))
    elif CONFIDENCE_ORDER[confidence] > CONFIDENCE_ORDER[existing.confidence]:
        conn.execute("UPDATE relationships SET confidence=? WHERE id=?", (confidence, rel_id))
    # else: ignorieren (neue Confidence nicht höher)
    conn.execute("COMMIT")
except:
    conn.execute("ROLLBACK")
```
`CONFIDENCE_ORDER = {'extracted': 3, 'inferred': 2, 'ambiguous': 1}`

**INSERT-Fix (Hyperion-Kritik):** INSERT muss um `confidence` erweitert werden:
```python
"INSERT OR IGNORE INTO relationships(id, source_id, target_id, relation_type, description, confidence)"
"VALUES (?, ?, ?, ?, ?, ?)"
```
Und Parsing des LLM-Outputs muss das `confidence` Feld extrahieren und mitliefern.

**Dedup-Fix (Hyperion-Kritik):** `_deduplicate_relationships()` muss Confidence priorisieren.

**Konkrete SQL-Logik:** Dedup löscht NUR wenn ältere Row ≤ gleiche Confidence hat. Join auf **alle 3 Spalten** (source + target + type):
```sql
-- NUR soft-delete wenn gleiche source+target+type und ältere Zeile gleiche oder niedrigere Confidence hat
UPDATE relationships SET valid_until = datetime('now')
WHERE id IN (
    SELECT r1.id FROM relationships r1
    INNER JOIN relationships r2
        ON r1.source_id = r2.source_id
        AND r1.target_id = r2.target_id          -- ← alle 3 Spalten!
        AND r1.relation_type = r2.relation_type
        AND r1.valid_until IS NULL AND r2.valid_until IS NULL
    WHERE r1.rowid < r2.rowid
        AND CASE r1.confidence WHEN 'extracted' THEN 3 WHEN 'inferred' THEN 2 ELSE 1 END
         <= CASE r2.confidence WHEN 'extracted' THEN 3 WHEN 'inferred' THEN 2 ELSE 1 END
);
```
In Python implementiert als Dict `CONFIDENCE_ORDER = {'extracted': 3, 'inferred': 2, 'ambiguous': 1}` — `extracted` kann **niemals** durch `inferred` oder `ambiguous` ersetzt werden.

**Nutzung im Agent:** Der Graph-Bridge-Output zeigt Confidence mit an:
```
[score=0.85 confidence=extracted] Owner BELEGT Gruenwald
[score=0.72 confidence=inferred] Owner HABT_ZIEL Immobilienportfolio
```

Der Agent kann sich darauf verlassen: `extracted` = harte Fakten, `inferred` = plausible Ableitungen, `ambiguous` = mit Vorsicht verwenden.

### Aufwand: ~60 Zeilen (Prompt-Anpassung + Migration + Bridge-Output + Dedup-Fix)

---

## 2. Why Nodes

### Problem
Der Graph speichert nur Zustände, nicht Rationale. "Warum ist die Miete 720€?" → muss neu recherchiert werden statt aus dem Graph zu kommen.

### Lösung
Design-Rationale als eigene Entity-Typen (`Rationale`, `Note`, `Hack`) extrahieren und an die Fakten-Entities linken.

**Beispiel Wiki-Text:**
> Mietpreis wurde von 650€ auf 720€ erhöht, da die Nebenkosten sich 2024 verdoppelt haben.

**Extraktion:**
- Entity: `Mietpreis_Gruenwald` (Typ: `Wert`, Label: "720€") → relationship: BELEGT_VON Gruenwald
- Entity: `Nebenkosten_Verdopplung_2024` (Typ: `Rationale`, Label: "Nebenkosten verdopplten sich 2024") → relationship: ERKLÄRT_MIT Mietpreis_Gruenwald

**Schema:** Kein neues Schema nötig — nutzt bestehende `entities` + `relationships`. Neuer Entity-Typ `Rationale` und neue Relationstypen: `ERKLÄRT_MIT`, `HINTET_AUF`, `WEGEN`.

**Extraktion:** LLM-Prompt bekommt neue Anweisung: wenn ein Text einen Grund oder Kommentar liefert ("da", "weil", "Hinweis:", "# NOTE:"), extrahiere ihn als Rationale-Entity und verlinke mit der Fakten-Entity.

**⚠️ Nur bei einfachen Fällen (Hyperion-Kritik):** Nur wenn Ursache + Wirkung im selben Satz stehen und eindeutig zugeordnet sind. Komplexe Fälle ("Miete stieg wegen X, aber Versicherung fiel wegen Y") → NICHT extrahieren statt falsche Links zu erzeugen.

### Aufwand: ~80 Zeilen (Prompt-Erweiterung + Rationale-Filter in Bridge-Query)

---

## 3. HTML Visualisierung

### Problem
Der Graph ist nur über CLI sichtbar. Zum Debuggen, Verstehen, und Präsentieren brauch man eine visuelle Darstellung.

### Lösung
Interaktives HTML mit D3.js Force-Directed Graph. Eine Seite, ein Klick — kein Server nötig.

**Output:** `/home/user/wikis/vectordb/graph.html` — statische HTML-Datei, direkt im Browser öffnen.

**Features:**
- Force-Directed Layout (Knoten ziehen, Zoom, Pan)
- Knoten-Farbe nach Entity-Typ (Person=blau, Raum=grün, Wert=gelb, Rationale=lila)
- Kantendicke nach Confidence (extracted=dick, inferred=mittel, ambiguous=dünn/gepunktet)
- Klick auf Knoten → zeigt Label + ausgehende Beziehungen im Sidebar
- Suche nach Label
- Filter: Nur bestimmte Entity-Typen anzeigen / Nur extracted/inferred zeigen

**Technik:**  
- Python generiert JSON aus der DB (`nodes.json` + `edges.json`)  
- **D3.js INLINE im HTML eingebettet** (~250KB minifiziert, ~115KB gzipped) — kein CDN, kein externer Request. Subset (d3-force + d3-selection + d3-zoom ≈ ~90KB min) wäre alternativ möglich erfordert aber Build-Tooling.  
- Kein Server nötig — rein statisch  

**⚠️ Hard Limit: max 200 Knoten / 500 Kanten** — mit `--top-degree N` nur hoch-konnektierte Knoten nehmen.

**⚠️ Datenschutz im Export:** `chmod 600 graph.html` beim Generieren setzen + "CONFIDENTIAL" Banner im HTML-Header rendern wenn private Daten enthalten sind.

**CLI-Befehl:**
```bash
vectordb.py graph export-html          # generiert graph.html
vectordb.py graph export-html --min-confidence extracted  # nur harte Fakten
```

### Aufwand: ~150 Zeilen Python + ~400 Zeilen HTML/JS (D3.js Template)

---

## 4. Implementierungs-Reihenfolge

### Phase 2A: Confidence Tags (~60 LOC)
1. Migration: `ALTER TABLE relationships ADD COLUMN confidence TEXT DEFAULT 'inferred'`  
2. LLM-Prompt erweitern (neue Output-Anweisung für confidence pro Triple + Few-Shot-Beispiele)  
3. `_insert_triples()` — SELECT-first Pattern in Transaktion (INSERT mit confidence ODER UPDATE für Upgrades)  
4. **Dedup Fix: `_deduplicate_relationships()` Confidence priorisieren (konkrete SQL)**  
5. `_entities_to_json()` / Bridge-Query — confidence im Output anzeigen  
6. CLI: `graph stats` zeigt Confidence-Distribution  
7. **Manuelle Validierung: Stichprobe ~20 Relationships prüfen**  
8. Hyperion Review → Wiki-Portierung  

### Phase 2B: Why Nodes (~80 LOC)  
(Startet erst nach erfolgreicher Stabilisierung von Phase 2A — Prompt-Komplexität muss gemessen werden)  
1. LLM-Prompt erweitern (Rationale-Erkennung + Extraktion)  
2. Neuen Entity-Typ `Rationale` im Prompt definieren  
3. Keine Migration nötig — nutzt bestehende Tables  
4. Bridge-Query erweitert: Rationale-Nodes im Output enthalten (optional filterbar)  
5. CLI: `graph search --with-rationale` → zeigt auch den Grund mit an  
6. Hyperion Review → Wiki-Portierung  

### Phase 2C: HTML Visualisierung (~550 LOC incl. Template)  
1. `_export_graph_data()` — nodes.json + edges.json aus DB generieren  
2. D3.js Template als String in Python hardcoden (inline ~250KB)  
3. HTML rendern → `vectordb/graph.html` (+ chmod 600 + CONFIDENTIAL-Banner)  
4. CLI: `graph export-html [--min-confidence extracted] [--scope private]`  
5. Hyperion Review → Wiki-Portierung  

---

## 5. Risiken & Abwägungen

| Risiko | Bewertung | Mitigation |  
|--------|-----------|------------|  
| Confidence-Tags sind subjektiv | 🟡 Mittel | Klare Definitionen im Prompt; konsistenter Default = 'inferred'; Few-Shot-Beispiele |  
| Rationale-Erklärung wird zu oft extrahiert → Rauschen | 🟡 Mittel | Nur bei expliziten Markern + nur einfache Fälle; komplexe Fälle ignorieren |  
| D3.js HTML lädt langsam bei großen Graphen | 🟡 Mittel | **Hard Limit max 200 Knoten / 500 Kanten**; Cluster-Knoten möglich |  
| Prompt wird länger → höhere Parse-Fehler-Rate | 🟡 Mittel | Separate JSON-Felder; messen nach Phase 2A bevor Phase 2B startet |  
| graph.html enthält sensible Daten | 🟡 Mittel | chmod 600 + CONFIDENTIAL Banner im HTML |  

---

## 6. Kein externer Datenfluss  
Alles lokal — keine API-Calls außer bestehende Ollama/vLLM Extraktion. D3.js inline eingebettet, kein CDN. ⚡


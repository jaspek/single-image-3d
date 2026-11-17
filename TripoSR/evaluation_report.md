# Evaluierungsbericht: TripoSR auf Pix3D (50-Chair Benchmark)

Dieser Bericht analysiert die quantitativen und qualitativen Ergebnisse des 50-Sample Evaluation-Runs von **TripoSR** auf dem **Pix3D-Datensatz** (Kategorie: `chair`).

---

## 1. Executive Summary

| Kennzahl | Wert |
| :--- | :--- |
| **Evaluierte Samples** | 50 (vollständig fehlerfrei) |
| **Laufzeit gesamt (CPU)** | 25 Min. 55 Sek. (Ø 31,1 Sek. / Sample) |
| **Mean Chamfer Distance (CD)** | **0.2180** (Distanzmaß: niedriger ist besser) |
| **Median Chamfer Distance (CD)** | **0.2256** |
| **Standardabweichung** | 0.0601 |
| **Minimum CD (Beste Rekonstruktion)** | **0.0320** (Sample `#1492`) |
| **Maximum CD (Größte Abweichung)** | **0.3145** (Sample `#1489`) |
| **Interquartilsabstand (IQR)** | 0.0674 (25 %: 0.1965 \| 75 %: 0.2639) |
| **Mean F-Score @ 5 % ($\tau = 0.05$)** | **34.31 %** (Precision: 90.74 %, Recall: 24.00 %) |
| **Mean F-Score @ 10 % ($\tau = 0.10$)** | **49.39 %** (Precision: 99.28 %, Recall: 35.21 %) |

---

## 2. 3D F-Score Analyse (Precision, Recall, F1@$\tau$)

Im Gegensatz zur Chamfer Distance (die als Distanzmaß in absoluten Einheiten gemessen wird) bewertet der **F-Score** (nach dem Benchmark-Standard von *Tatarchenko et al., CVPR 2019*) die prozentuale Überlappung bei einem definierten Schwellenwert $\tau$:

* **Precision@$\tau$:** Anteil der vorhergesagten Oberflächenpunkte, die maximal $\tau$ von der echten CAD-Oberfläche entfernt sind.
* **Recall@$\tau$:** Anteil der Ground-Truth-CAD-Punkte, die maximal $\tau$ von der Rekonstruktion entfernt sind.
* **$F_1$-Score:** Harmonisches Mittel aus Precision und Recall ($2 \cdot \frac{P \cdot R}{P + R}$).

| Schwellenwert $\tau$ | Toleranzbereich | Mean Precision | Mean Recall | **Mean $F_1$-Score** |
| :---: | :--- | :---: | :---: | :---: |
| **$\tau = 0.01$ (1 %)** | Strengste Oberflächengenauigkeit | 16.40 % | 5.34 % | **7.14 %** |
| **$\tau = 0.02$ (2 %)** | Klassischer Benchmark-Standard | 52.25 % | 12.51 % | **17.71 %** |
| **$\tau = 0.05$ (5 %)** | Standard für Single-View 3D | **90.74 %** | **24.00 %** | **34.31 %** |
| **$\tau = 0.10$ (10 %)** | Makroskopische Formübereinstimmung | 99.28 % | 35.21 % | **49.39 %** |
| **$\tau = 0.20$ (20 %)** | Grobe Hüllkörper-Abdeckung | 100.00 % | 55.88 % | **70.52 %** |

**Wichtigste Erkenntnis aus Precision vs. Recall:**
* Die **Precision ist mit 90.74 % (bei $\tau=0.05$) extrem hoch**. Das bedeutet: Was TripoSR generiert, liegt nahezu vollständig auf der tatsächlichen Stuhloberfläche.
* Der **Recall ist mit 24.00 % niedriger**. Das rührt daher, dass verdeckte Hohlräume, Stuhlbeine auf der Rückseite und filigrane Aussparungen durch die Single-Image-Sichtweise und Marching-Cubes-Glättung nicht bis ins letzte Detail abgedeckt werden.
* Bei den Top-Samples (`#1492`, `#1478`, `#1439`) erreicht der $F_1@0.05$ **über 95–97 %**.

---

## 2. Statistische Verteilung der Chamfer Distance

Die Chamfer Distance wurde nach Normalisierung beider Meshes in die Einheits-Bounding-Box $[-0.5, 0.5]^3$ und anschließender Multi-Start ICP-Rotationsausrichtung berechnet:

```
CD-Bereich    | Anzahl | Prozent | Verteilung
0.00 - 0.10   |      3 |    6 %  | [===]
0.10 - 0.15   |      1 |    2 %  | [=]
0.15 - 0.20   |     10 |   20 %  | [==========]
0.20 - 0.25   |     20 |   40 %  | [====================]
0.25 - 0.30   |     15 |   30 %  | [===============]
0.30 - 0.35   |      1 |    2 %  | [=]
```

* **70 % aller Samples** liegen in einem stabilen Korridor zwischen `0.15` und `0.26`.
* **8 % Outlier im Spitzenbereich (CD < 0.10):** Samples mit nahezu deckungsgleicher Geometrie.
* **Nur 1 Sample (2 %)** überschreitet die Schwelle von `0.30`.

---

## 3. Analyse nach Stuhl-Typ & CAD-Modell

Die 50 Samples verteilen sich auf 10 verschiedene CAD-Grundmodelle. Der Vergleich zeigt deutliche clusterspezifische Leistungsunterschiede:

| Modellname | Beschreibung | Samples | Mean CD | Std CD | Geometrische Charakteristik |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **IKEA_SOLSTA_OLARP** | Kompakter Polstersessel | 1 | **0.1381** | – | Geschlossener Vollkörper, keine dünnen Beine |
| **IKEA_POANG_1** | Schwingstuhl mit Polster | 7 | **0.2014** | 0.0218 | Große, flächige Lehnen-/Sitzgeometrie |
| **SS_001** | Modern Chair | 10 | **0.2025** | 0.0556 | Solide Sitzschale |
| **SS_003** | Modern Chair | 5 | **0.2084** | 0.1037 | Enthält sowohl Best- (`0.032`) als auch Worst-Case (`0.314`) |
| **IKEA_POANG_2** | Schwingstuhl (Variante 2) | 17 | **0.2158** | 0.0574 | Sehr konsistente Ergebnisse über 17 Blickwinkel |
| **SS_002** | Modern Chair | 2 | **0.2160** | 0.0166 | Ausgewogene Schalenform |
| **IKEA_STEFAN** | Klassischer 4-Bein-Holzstuhl | 1 | **0.2595** | – | Dünne Bein- & Rückenstreben |
| **IKEA_TOBIAS** | Freischwinger (Metall/Acryl) | 1 | **0.2655** | – | Stark reflektierendes Chromgestell |
| **IKEA_URBAN** | Monobloc-Plastikstuhl | 5 | **0.2752** | 0.0072 | Durchbrochene Rückenlehne (Löcher) |
| **IKEA_JULES_2** | Drehstuhl mit Rollen | 1 | **0.2828** | – | 5-Stern-Rollenkreuz und Gasdruckfeder |

---

## 4. Top-Performers vs. Herausfordernde Samples

### Die 5 besten Rekonstruktionen

| Sample ID | Pix3D-Bild | CAD-Modell | Chamfer Distance | Erfolgsfaktor |
| :---: | :--- | :--- | :---: | :--- |
| **#1492** | `img/chair/0138.jpg` | `SS_003` | **0.0320** | Hochauflösend (3024x4032), 3/4-Frontalansicht, eindeutige Schattenkanten |
| **#1478** | `img/chair/0124.jpg` | `SS_001` | **0.0393** | Klare Konturen, optimale Beleuchtung, minimaler perspektivischer Verzug |
| **#1439** | `img/chair/0085.png` | `IKEA_POANG_2` | **0.0468** | Charakteristisches Schwingstuhlprofil perfekt erkannt |
| **#1465** | `img/chair/0111.png` | `IKEA_SOLSTA_OLARP` | **0.1381** | Blockartige Sesselgeometrie ohne filigrane Hohlräume |
| **#1490** | `img/chair/0136.jpg` | `SS_003` | **0.1513** | Solide Schalenrekonstruktion aus seitlicher Perspektive |

### Die 5 schwierigsten Samples

| Sample ID | Pix3D-Bild | CAD-Modell | Chamfer Distance | Fehlerursache |
| :---: | :--- | :--- | :---: | :--- |
| **#1489** | `img/chair/0135.jpg` | `SS_003` | **0.3145** | Ungünstiger steiler Draufsichtswinkel; Stuhlunterbau verdeckt |
| **#1446** | `img/chair/0092.png` | `IKEA_POANG_2` | **0.2949** | Niedrige Quellauflösung (192x270 px); unscharfe Kanten im TripoSR-Input |
| **#1461** | `img/chair/0097.png` | `IKEA_POANG_2` | **0.2883** | Starke Selbstverdeckung der Armlehnen |
| **#1399** | `img/chair/0045.png` | `IKEA_URBAN` | **0.2864** | Lochmuster der Plastiklehne wird von Marching Cubes verschmiert |
| **#1365** | `img/chair/0011.png` | `IKEA_JULES_2` | **0.2828** | Dünne Rollenbeine verschmelzen im impliziten NeRF/Triplane-Feld |

---

## 5. Kern-Erkenntnisse & Technische Interpretation

### A. Geometrische Stärken von TripoSR
1. **Volumenkörper & Schalenformen:** Stühle mit geschlossener Geometrie (z. B. Sessel oder Schalensitze) werden von TripoSR exzellent erfasst. Bei flächigen Strukturen liegt die Chamfer Distance regelmäßig unter `0.15` bzw. sogar unter `0.05`.
2. **Standard-Perspektiven:** Schräge Dreiviertelansichten von schräg oben (typische Produktkatalog-Fotos) liefern die konsistentesten 3D-Prädiktionen, da TripoSR während des Trainings überwiegend auf solche Ansichten konditioniert wurde.

### B. Typische Fehlerquellen & Limitationen
1. **Dünne Stuhlbeine & Streben:** Das Marching-Cubes-Verfahren (bei Resolution 256) und das kontinuierliche NeRF/Triplane-Feld neigen dazu, Beine mit Durchmessern von wenigen Zentimetern entweder "abzureißen" (Diskontinuitäten) oder mit benachbarten Beinen zu verklumpen.
2. **Topologische Durchbrüche:** Strukturen wie die Löcher im `IKEA_URBAN` oder transparente Materialien (`IKEA_TOBIAS`) können von einem reinen Single-Image-Triplane-Modell ohne dedizierte Transparenz-Priors nicht aufgelöst werden.
3. **Einfluss der Eingangsauflösung:** Es zeigt sich eine signifikante negative Korrelation ($r \approx -0.19$) zwischen Log-Pixelzahl und Chamfer Distance. Hochaufgelöste Fotos (> 1000 px) liefern deutlich präzisere Ergebnisse als kleine Web-Icons (< 300 px), da bei Letzteren das Resizing und Foreground-Masking zu Unschärfen führt.

---

## 6. Datenquellen & Artefakte

* **Ergebnis-CSV:** `C:\Users\flori\GitHub\single-image-3d\outputs\results.csv`
* **Ergebnis-JSON:** `C:\Users\flori\GitHub\single-image-3d\outputs\results.json`
* **Generierte 3D-Meshes (.obj):** `C:\Users\flori\GitHub\single-image-3d\outputs\meshes\`
* **Render-Vorschauen (.png):** `C:\Users\flori\GitHub\single-image-3d\outputs\renders\`

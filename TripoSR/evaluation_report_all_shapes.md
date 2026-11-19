# Umfassender Evaluierungsbericht: TripoSR auf allen Pix3D-Kategorien (Multi-Shape Benchmark)

Dieser Bericht präsentiert die quantitative und qualitative Evaluation von **TripoSR** über **alle 9 Objektformen / Kategorien** des **Pix3D-Datensatzes** (`bed`, `bookcase`, `chair`, `desk`, `misc`, `sofa`, `table`, `tool`, `wardrobe`). Untersucht wurden 90 unokkludierte, nicht-trunkierte Bildsamples (10 Samples je Kategorie) bezüglich der **Chamfer Distance (CD)** sowie der **3D F-Scores** nach dem Benchmark-Standard von *Tatarchenko et al. (CVPR 2019)* über die Schwellenwerte $\tau \in \{0.01, 0.02, 0.05, 0.10, 0.20\}$.

---

## 1. Executive Summary & Ranking der Objektformen

| Rang | Kategorie | Deutsche Bezeichnung | Mean CD ↓ | F1@0.02 ↑ | **F1@0.05 ↑** | **F1@0.10 ↑** | Prec@0.05 ↑ | Rec@0.05 ↑ | Geometrischer Hauptcharakter |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | **wardrobe** | Kleiderschrank | **0.1087** | **27.8 %** | **61.9 %** | **84.3 %** | 76.9 % | **53.6 %** | Quaderförmiger Vollkörper, große geschlossene Flächen |
| **2** | **misc** | Sonstiges | **0.1742** | **35.0 %** | **51.2 %** | **63.6 %** | 82.9 % | **45.4 %** | Kompakte Design-Objekte mit klarer Kontur |
| **3** | **table** | Tisch | **0.2010** | 23.8 % | **48.0 %** | **63.0 %** | 76.9 % | 38.4 % | Große Tischplatte, Beine/Zargen teils vereinfacht |
| **4** | **tool** | Werkzeug | **0.1602** | 22.7 % | **46.7 %** | **64.7 %** | 88.3 % | 38.0 % | Kompakte Griffe und Funktionsteile |
| **5** | **bed** | Bett | **0.2074** | 18.3 % | **33.7 %** | **47.5 %** | 95.5 % | 21.5 % | Großflächige Liegefläche, verdeckte Unterbauten |
| **6** | **sofa** | Sofa | **0.2236** | 15.5 % | **28.7 %** | **43.2 %** | 95.4 % | 17.3 % | Flächige Polster, hohle Sitzecken & Rückseiten |
| **7** | **desk** | Schreibtisch | **0.2855** | 17.2 % | **28.5 %** | **40.8 %** | 90.1 % | 17.0 % | Dünne Platten, Beinfreiheit, Container |
| **8** | **chair** | Stuhl | **0.2323** | 12.5 % | **25.5 %** | **41.9 %** | 91.8 % | 16.2 % | Filigrane Streben, 4 dünne Beine, Armlehnen |
| **9** | **bookcase** | Bücherregal | **0.2547** | 12.8 % | **20.6 %** | **30.9 %** | **98.0 %** | **12.5 %** | Dünne Regalböden, offene Fächer / Kavitäten |
| – | **OVERALL** | *Gesamtdurchschnitt* | **0.2053** | **20.6 %** | **38.3 %** | **53.3 %** | **88.4 %** | **28.9 %** | *90 Samples (10 pro Kategorie)* |

---

## 2. Detaillierte 3D F-Score Analyse ($\tau$-Sweep)

Der $F_1$-Score ($2 \cdot \frac{P \cdot R}{P + R}$) bewertet die Schnittmenge zwischen rekonstruierter Oberfläche und Ground-Truth-CAD-Modell bei variierenden Toleranzradien $\tau$ (bezogen auf die normalisierte Bounding-Box-Kantenlänge $1.0$):

### Kategorieweiser $F_1$-Score über alle Schwellenwerte

| Kategorie | $F_1@0.01$ (1 %) | $F_1@0.02$ (2 %) | $F_1@0.05$ (5 %) | $F_1@0.10$ (10 %) | $F_1@0.20$ (20 %) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **wardrobe** | 9.9 % | 27.8 % | **61.9 %** | **84.3 %** | **93.8 %** |
| **misc** | **16.2 %** | **35.0 %** | **51.2 %** | **63.6 %** | 77.5 % |
| **table** | 8.0 % | 23.8 % | **48.0 %** | 63.0 % | 78.8 % |
| **tool** | 9.2 % | 22.7 % | **46.7 %** | 64.7 % | 80.5 % |
| **bed** | 8.4 % | 18.3 % | 33.7 % | 47.5 % | 71.2 % |
| **sofa** | 7.4 % | 15.5 % | 28.7 % | 43.2 % | 66.2 % |
| **desk** | 9.3 % | 17.2 % | 28.5 % | 40.8 % | 60.8 % |
| **chair** | 6.3 % | 12.5 % | 25.5 % | 41.9 % | 66.6 % |
| **bookcase** | 7.0 % | 12.8 % | 20.6 % | 30.9 % | 61.0 % |
| **Gesamt** | **9.1 %** | **20.6 %** | **38.3 %** | **53.3 %** | **72.9 %** |

---

## 3. Kern-Erkenntnis: Precision vs. Recall Diskrepanz

Eine der markantesten Beobachtungen über alle Formen hinweg ist die extreme Asymmetrie zwischen **Precision** und **Recall**:

* **Überragende Precision ($\varnothing\ 88.4\ \%$ bei $\tau=0.05$):**
  Bei geschlossenen Möbeln wie `bookcase` (98.0 %), `bed` (95.5 %), `sofa` (95.4 %) und `chair` (91.8 %) liegt nahezu jeder Punkt, den TripoSR erzeugt, tatsächlich auf der echten Objektoberfläche. Das Modell "halluziniert" kaum freischwebende Geometrie außerhalb der wahren Konturen.
* **Niedriger Recall ($\varnothing\ 28.9\ \%$ bei $\tau=0.05$):**
  TripoSR leidet unter einem signifikanten Vollständigkeitsdefizit bei:
  1. **Verdeckten Regionen:** Die Rückseite, Unterseiten von Tischen/Betten oder Hohlräume hinter Möbelfronten sind im 2D-Einzelfoto nicht sichtbar und werden vom NeRF-Feld geglättet oder verschlossen.
  2. **Dünnen Platten und Fachböden:** Ein Bücherregal (`bookcase`) besitzt Regalböden mit Stärken von 1–2 cm. Im $256^3$- bzw. $128^3$-Gitter des Marching Cubes und dem kontinuierlichen Triplane-Feld verschmelzen diese Böden oder werden unvollständig aufgelöst, was den Recall auf magere **12.5 %** drückt.

---

## 4. Geometrische Ursachenanalyse nach Objektklassen

### A. Spitzenreiter: `wardrobe` ($F_1@0.05 = 61.9\ \%$, CD = $0.1087$)
* **Warum TripoSR hier brilliert:** Ein Kleiderschrank ist im Wesentlichen ein massiver Quader (konvexer Vollkörper). Es gibt keine dünnen Beine, keine durchbrochenen Gitter und keine tiefen inneren Schatten.
* **Ergebnis:** Das Marching-Cubes-Verfahren rekonstruiert saubere ebene Flächen. Recall (53.6 %) und Precision (76.9 %) sind harmonisch ausbalanciert, und bei $\tau=0.10$ wird ein F1-Score von **84.3 %** erreicht.

### B. Solide Mittelfeld-Performers: `table` ($48.0\ %$) und `tool` ($46.7\ %$)
* **`table`:** Die Tischplatte bietet eine riesige, eindeutige Referenzfläche. Punktwolken-Samples treffen zu einem hohen Prozentsatz diese Platte, wodurch CD und F1-Score stabil bleiben, selbst wenn die Tischbeine leicht abdriften.
* **`tool`:** Werkzeuge sind meist kompakt und massiv ohne komplexe Hinterschneidungen.

### C. Polstermöbel: `bed` ($33.7\ %$) und `sofa` ($28.7\ %$)
* **Charakteristik:** Sofas und Betten haben eine sehr hohe Precision (> 95 %), aber der Recall fällt auf ~17–21 % ab. 
* **Ursache:** Die sichtbaren Kissen- und Lehnenflächen werden exzellent modelliert. Da Betten und Sofas jedoch oft flach auf dem Boden stehen oder davor Teppiche/Bettdecken liegen, kann das implizite Feld den Bettkasten oder die Rückseite nur als undifferenzierten Block schätzen.

### D. Die schwierigsten Geometrien: `chair` ($25.5\ %$) und `bookcase` ($20.6\ %$)
* **`chair`:** Stühle sind der klassische Härtetest für 3D-Rekonstruktion. Dünne Streben (z. B. IKEA Stefan oder Metallbeine), durchbrochene Rückenlehnen (IKEA Urban) und 4 separate Beine fordern das kontinuierliche NeRF-Triplane heraus. Oft "verbinden" sich Beine oder brechen ab.
* **`bookcase` (Schlusslicht):** Ein Regal besteht fast ausschließlich aus Hohlräumen und millimeterdünnen Brettern. Da TripoSR als feed-forward Netzwerk auf geschlossene wasserdichte Oberflächen trainiert ist, füllt es Fächer teilweise auf oder lässt Regalböden verschwinden. Dadurch verfehlen die CAD-Punkte im Regalinneren die Vorhersage, was den F1-Score auf $20.6\ \%$ drückt.

---

## 5. Extremwert-Analyse: Top 5 vs. Flop 5 Samples

### Top 5 Einzelrekonstruktionen

| Sample ID | Kategorie | Bildpfad | CAD-Modell | Chamfer Distance ↓ | F1@0.05 ↑ | Geometrischer Erfolgsfaktor |
| :---: | :--- | :--- | :--- | :---: | :---: | :--- |
| **#5903** | `misc` | `img/misc/0010.jpg` | `SS_02` | **0.0260** | **99.5 %** | Nahezu perfekte Übereinstimmung einer kompakten Vollform |
| **#5901** | `misc` | `img/misc/0008.jpg` | `SS_02` | **0.0260** | **99.0 %** | Identisches CAD-Modell aus optimaler 3/4-Perspektive |
| **#5900** | `misc` | `img/misc/0007.jpg` | `SS_02` | **0.0320** | **96.8 %** | Scharfe Kanten und eindeutige Tiefenkonturen |
| **#9831** | `wardrobe` | `img/wardrobe/0006.png` | `IKEA_ANEBODA_1` | **0.0464** | **91.4 %** | Geradliniger Kleiderschrank ohne filigrane Hohlräume |
| **#7930** | `table` | `img/table/0022.png` | `IKEA_LACK_3` | **0.0536** | **90.1 %** | Kompakter Beistelltisch (IKEA Lack), dicke Beine |

### Flop 5 Einzelrekonstruktionen

| Sample ID | Kategorie | Bildpfad | CAD-Modell | Chamfer Distance ↓ | F1@0.05 ↓ | Geometrische Fehlerursache |
| :---: | :--- | :--- | :--- | :---: | :---: | :--- |
| **#1004** | `bookcase` | `img/bookcase/0011.jpg` | `IKEA_BILLY_1` | **0.2879** | **8.2 %** | Tiefes Billy-Regal: Regalböden werden nicht erfasst |
| **#1005** | `bookcase` | `img/bookcase/0012.jpg` | `IKEA_BILLY_1` | **0.2855** | **8.4 %** | Starke frontale Schatten, leere Fächer verschmiert |
| **#0996** | `bookcase` | `img/bookcase/0003.jpg` | `IKEA_BILLY_1` | **0.2868** | **8.5 %** | Geringer Kontrast im Innenraum, Hohlräume blockiert |
| **#5902** | `misc` | `img/misc/0009.jpg` | `SS_02` | **0.2848** | **9.8 %** | Ungünstiger steiler Draufsichtswinkel, Verkippung |
| **#1370** | `chair` | `img/chair/0016.png` | `IKEA_URBAN` | **0.2880** | **11.0 %** | Monobloc-Plastikstuhl mit Löchern; Marching-Cubes-Glättung |

---

## 6. Zusammenfassendes Fazit

1. **Starke Abhängigkeit von der Oberflächentopologie:**
   TripoSR erzielt hervorragende Ergebnisse bei **konvexen, geschlossenen Vollkörpern** (`wardrobe`: F1@0.05 = $61.9\ \%$, F1@0.10 = $84.3\ %$). Sobald **Hohlräume, dünne Platten oder Aussparungen** dominieren (`bookcase`: F1@0.05 = $20.6\ %$, `chair`: $25.5\ %$), bricht der Recall ein.
2. **Precision als Qualitätssiegel:**
   Mit einer mittleren Precision von **$88.4\ \%$** beweist TripoSR, dass es zuverlässig die sichtbaren Oberflächenanteile trifft und kaum "Geistergeometrien" erzeugt.
3. **Vergleich zu Stühlen:**
   Der ursprüngliche Chair-Benchmark bildete mit die schwierigste Kategorie ab. Kategorien wie Kleiderschränke, Tische oder Werkzeuge erzielen in TripoSR um **Faktor 2 bis 3 höhere F1-Werte**.

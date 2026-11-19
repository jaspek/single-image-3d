# TripoSR Evaluation on Pix3D (Multi-Category Benchmark)

Dieses Verzeichnis bündelt alle Skripte, Auswertungen und generierten Daten der **TripoSR-Evaluation** auf dem **Pix3D-Datensatz** über alle 9 Objekt-Kategorien / Formen (`bed`, `bookcase`, `chair`, `desk`, `misc`, `sofa`, `table`, `tool`, `wardrobe`).

---

## 📁 Struktur

```text
single-image-3d/TripoSR/
├── eval_pipeline.py                 # Multi-Category Benchmark-Pipeline (Inferenz, ICP & CD/F1-Scores)
├── calc_fscore.py                   # 3D F-Score Analyse-Tool (Precision, Recall, F1@tau über alle Kategorien)
├── evaluation_report.md             # Ursprünglicher Bericht (Kategorie: chair, 50 Samples)
├── evaluation_report_all_shapes.md  # Umfassender Vergleichsbericht aller 9 Pix3D-Kategorien
├── README.md                        # Diese Dokumentation
└── outputs/                         # Generierte Artefakte (von Git ignoriert)
    ├── inputs/                      # Maskierte & vorverarbeitete 2D-Eingabebilder (<id>_input.png)
    ├── meshes/                      # Generierte 3D-Meshes von TripoSR (<id>_pred.obj)
    ├── renders/                     # 2D-Vorschau-Renderings der 3D-Meshes (<id>_render.png)
    ├── results.csv                  # Detaillierte tabellarische Auswertung aller Samples
    └── results.json                 # Zusammenfassung mit kategorieweisen & globalen Metriken
```

---

## 🚀 Ausführung

Die Skripte können sowohl aus dem Hauptverzeichnis `single-image-3d` als auch direkt aus diesem Ordner gestartet werden:

### 1. Benchmark über alle Kategorien ausführen

```powershell
# Vollständiger Multi-Category Benchmark (alle 9 Formen, z. B. 10 Samples pro Form)
python TripoSR/eval_pipeline.py --category all --num-samples 10

# Einzelne Kategorie evaluieren (z. B. chair, sofa, bed, table etc.)
python TripoSR/eval_pipeline.py --category chair --num-samples 50

# Nur Evaluierung bereits generierter Meshes (überspringt Inferenz)
python TripoSR/eval_pipeline.py --eval-only
```

### 2. 3D F-Scores analysieren & vergleichen

```powershell
# Liest results.json und gibt tabellarischen F-Score-Vergleich über alle Kategorien aus
python TripoSR/calc_fscore.py

# Optional: Neuberechnung direkt aus den .obj-Meshes erzwingen
python TripoSR/calc_fscore.py --recompute
```

---

## 📊 Metriken & Auswertungs-Details

* **Symmetric Chamfer Distance (CD):** Nach Bounding-Box-Zentrierung, Einheits-Skalierung und Multi-Start ICP.
* **3D F-Score nach Tatarchenko et al. (CVPR 2019):**
  * **Precision@$\tau$:** Prozentsatz vorhergesagter Oberflächenpunkte innerhalb von Distanz $\tau$ zur CAD-Referenz.
  * **Recall@$\tau$:** Prozentsatz der Ground-Truth-CAD-Punkte innerhalb von Distanz $\tau$ zur Rekonstruktion.
  * **$F_1$-Score@$\tau$:** Harmonisches Mittel $2 \cdot \frac{P \cdot R}{P + R}$ für Schwellenwerte $\tau \in \{0.01, 0.02, 0.05, 0.10, 0.20\}$.

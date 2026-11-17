# TripoSR Evaluation on Pix3D

Dieses Verzeichnis bündelt alle Skripte, Auswertungen und generierten Daten der **TripoSR-Evaluation** auf dem **Pix3D-Datensatz** (Kategorie `chair`, 50 Samples).

---

## 📁 Struktur

```text
single-image-3d/TripoSR/
├── eval_pipeline.py        # Haupt-Evaluierungspipeline (Inferenz, ICP-Alignment & Chamfer Distance)
├── calc_fscore.py          # Skript zur Berechnung der 3D F-Scores (Precision, Recall, F1@tau)
├── evaluation_report.md    # Umfassender Evaluierungsbericht mit Detailanalyse
├── README.md               # Diese Dokumentation
└── outputs/                # Generierte Daten (von Git ignoriert)
    ├── inputs/             # Maskierte & vorverarbeitete 2D-Eingabebilder (<id>_input.png)
    ├── meshes/             # Generierte 3D-Meshes von TripoSR (<id>_pred.obj)
    ├── renders/            # 2D-Vorschau-Renderings der 3D-Meshes (<id>_render.png)
    ├── results.csv         # Tabellarische Auswertung aller 50 Samples
    └── results.json        # Zusammenfassung & Chamfer-Distance-Statistiken
```

---

## 🚀 Ausführung

Die Skripte können sowohl aus dem Hauptverzeichnis `single-image-3d` als auch direkt aus diesem Ordner gestartet werden:

### 1. Chamfer-Distance-Pipeline ausführen

```powershell
# Vollständiger Durchlauf (50 Stühle)
python TripoSR/eval_pipeline.py

# Schneller Testlauf mit 1 oder 5 Samples
python TripoSR/eval_pipeline.py --num-samples 1

# Nur Evaluierung der existierenden Meshes (ohne erneute Inferenz)
python TripoSR/eval_pipeline.py --eval-only
```

### 2. 3D F-Scores berechnen

```powershell
python TripoSR/calc_fscore.py
```

---

## 📊 Benchmark-Ergebnisse (50 Pix3D-Stühle)

* **Mean Chamfer Distance:** `0.2180` (Median: `0.2256`)
* **Bester Einzelwert (CD):** `0.0320` (Sample `#1492`)
* **Mean F-Score @ 5 % ($\tau = 0.05$):** `34.31 %` (Precision: `90.74 %`, Recall: `24.00 %`)
* **Mean F-Score @ 10 % ($\tau = 0.10$):** `49.39 %` (Precision: `99.28 %`, Recall: `35.21 %`)

Details und Fehlerursachenanalysen finden sich in `evaluation_report.md`.

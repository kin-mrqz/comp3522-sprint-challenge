# HKO context — citation only

These files are copied from the course project so every teammate can check a unit, a key, or a published number without the OneDrive path. They are not extra training data. The sprint brief disallows other datasets unless the scenario says otherwise. If the 1:00 scenario is not about HKO, do not cite this folder.

The weather tables, the Hugging Face cache, the slide decks, and the collector scripts were left out on purpose.

| File | Use it for |
|---|---|
| `GRAPH_REPORT.md` | Graph definition, the 2024-07-01 / 2025-01-01 split, and test CSI / PR-AUC |
| `data_dictionary_v0.md` | Hong Kong time, millimetres, amber/red/black at 30/50/70 mm, and the rule that maintenance code `M` stays missing |
| `collection_README.md` | Sources, station keys, and the 2023-01-01 to 2025-12-31 window |
| `EDA_report.md` | What the early station feed could support (temperature) and what it could not (a rain forecast) |
| `tables/graph_metrics.csv` | The full published metric rows behind the report |
| `tables/graph_vs_statistical.csv` | Graph models next to the statistical baselines |
| `tables/graph_summary.csv` | Node and edge counts |
| `tables/task1_moran_by_lead.csv` | Temperature error and Moran's I by lead day |
| `figures/csi_territory_amber_by_lead.png` | Territory CSI by lead, for the link slide |
| `figures/cowin_graph_map.png` | The station graph, for the link slide |
| `figures/shap_xgb_graph.png` | The project's XGBoost SHAP bar, for the link slide |

The five sentences that are safe to put on a slide are in `project_anchors.md` at the repository root. The dataset the project was trained on is `NotASI/COMP3522-HKO` on Hugging Face. Do not download it into this sprint unless the scenario explicitly allows other datasets.

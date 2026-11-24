# Team clock

Three people, one deck. The teaching team will ask each of you about the part you own, so the split below is also the oral map.

Stay on `main`. Commit messages name the notebook: `01 profile`, `03 rate tables`.

## Together, 1:00–1:20

Open `notebooks/00_question_contract.ipynb` and fill `outputs/question_contract.json`.

- Write the 2 or 3 questions, the unit of a row, and the sentence that would count as an answer.
- Mark the decision time, so a later column cannot leak into a feature.
- Name the join keys and the entity risk. The project failure mode is the same station under two codes.
- Set `native_graph` to true or false. A native graph is an edge list, coordinates on a real entity (station, site, gauge), or a shared real-world id plus one of those. A kNN graph on feature columns is a no.
- Set `model_needed`. A descriptive question does not need a fit.
- If the split should be by time, set `train_end` and `val_end`. Otherwise notebook 04 uses a stratified 70/15/15 split.
- Assign each question to A, B, or C.

Person B may sketch feature code during this window. Nobody fits a model yet.

## Diverge, 1:20–2:20

**Person A — data trust.** Notebooks 01 and 02.

- Profile, duplicates, missingness, and the columns tagged `drop`.
- Normalize ids, write the alias map, and list unmatched keys. Fuzzy pairs are written and not merged.
- Write `data/processed/analysis_frame.parquet` and push it. Say the match rate out loud if someone asks: what fraction of keys matched, and what is still ambiguous.

**Person B — structure and tabular models.** Notebooks 03 and 04. Pull A's frame before fitting.

- Drop junk, build an id from descriptor columns, and write one two-way table. That table is the multidimensional pattern the rubric asks for.
- PCA in notebook 03 is a picture of the full frame. It is not a model feature.
- LDA runs only when the target is a class label. Otherwise the notebook records why it skipped.
- Notebook 04 writes `splits/split.npz` once. Aggregates for the model use training rows only.
- Fit linear or logistic regression and XGBoost. LightGBM and CatBoost stay off unless the question is predictive and the time is still before 14:05 (`RUN_LGBM`, `RUN_CATBOOST`).
- If asked: which configuration won on validation, and why that number is not a test score.

**Person C — graph, explanation, deck.** Notebooks 05 and 06, plus `slides/deck_outline.md`.

- Start the slide titles at 1:20 so the deck is not born at 2:40.
- If the gate is no, notebook 05 writes `outputs/graph_skipped.md` and you move on. If the gate is yes, degree features and one validation XGBoost. Neighbour label rates are refused unless you pass a training mask at node level.
- Project-link numbers come from `project_anchors.md` and `context/hko/`. Use them only when the metric family matches the question.
- SHAP runs in notebook 06 only after B has saved a model. At 2:10 an unstarted SHAP run is dropped.

## Pull points

1. A pushes `outputs/column_roles.json` and `data/processed/analysis_frame.parquet`.
2. B and C pull that commit before notebooks 03, 04, or 05 fit or build graph features.
3. Everyone pulls again before editing `outputs/answers.md` or the deck.

## Converge, 2:20–2:40

Edit `outputs/answers.md` together. Each question gets one claim, one chart, one limitation.

- A checks that the rows in the claim still exist after entity resolution.
- B checks that no quoted score used the test split or a column from after the decision time.
- C exports the PDF. Content slides stay at or under 12. Cover and references do not count.

2:40–2:50 is Moodle only: the PDF, plus the notebooks and `outputs/`.

## Hard stop, 2:10

If LightGBM, CatBoost, or SHAP has not been started, do not start it. The deck can stand on the profile, the match rate, and the two-way table. One logistic regression is enough when a question is actually predictive.

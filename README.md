# COMP3522 sprint challenge

Private repository: https://github.com/kin-mrqz/comp3522-sprint-challenge

`avcode-exe` and `dancheung555` have write invitations. Accept the invite, then clone and stay on `main`.

Working repo for the 24 November sprint. The graded piece is a PDF of at most 12 slides, plus these notebooks as the workings. Read `TEAM.md` before 1:00. The clock, the three tracks, and the file each person must push are in there.

## Setup

Windows:

```powershell
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m ipykernel install --user --name comp3522-sprint --display-name "comp3522-sprint"
```

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name comp3522-sprint --display-name "comp3522-sprint"
```

Use the `comp3522-sprint` kernel. Run notebooks from the repository root, or from `notebooks/` — the first cell finds `src/` either way.

## Notebook order

| Notebook | Who | Writes |
|---|---|---|
| `00_question_contract.ipynb` | all three, together | question contract, draft roles, graph gate |
| `01_eda.ipynb` | A | profiles and missingness figures |
| `02_cleaning_entity.ipynb` | A | analysis frame, alias map |
| `03_features.ipynb` | B | feature table, two-way table, descriptive PCA |
| `04_tabular.ipynb` | B | frozen split, linear model, XGBoost |
| `05_graph_features.ipynb` | C | graph features, or a written skip |
| `06_explain_answers.ipynb` | C | SHAP and importance, if a model exists |

`context/hko/` is the course project write-up for citation. It is not extra training data. The sprint brief disallows other datasets unless the scenario says otherwise.

## One branch

Stay on `main`. Person A pushes the analysis frame before B and C fit or build graph features. Pull before editing `outputs/question_contract.json`, `outputs/column_roles.json`, `outputs/answers.md`, or the slide deck.

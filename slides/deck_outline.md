# Slide outline

Cover and a references slide sit outside the 12. If a slot has nothing to show, delete it. Do not fill it with method trivia.

The rubric scores two things: a multidimensional pattern with a number behind it, and a deck that is short, visual, and owned by the whole team.

| Slot | Slide | Owner | What must be on it |
|---|---|---|---|
| 1 | Questions and datasets, one line each | all | The question sentence from `outputs/question_contract.json` |
| 2 | Kept and dropped, with counts | A | Rows in, rows out, columns dropped and why |
| 3 | Entity resolution | A | Match rate and what stayed unmatched |
| 4 | Multidimensional pattern | B | The two-way table, or the descriptive PCA plot, with the scope written on the figure |
| 5 | Question 1 | question owner | One chart, one number, one sentence |
| 6 | Question 2 | question owner | One chart, one number, one sentence |
| 7 | Question 3, or delete if there are only two | question owner | One chart, one number, one sentence |
| 8 | Model comparison, or a second pattern if no model was fit | B | Validation metric only. Say that the test split was not scored |
| 9 | Graph gate | C | The graph in one picture, or one sentence on why it was skipped |
| 10 | Link to the HKO project | C | Two numbers from `project_anchors.md`, same metric family as the question |
| 11 | Caveats | all | Time, leakage, keys that did not match |
| 12 | Spare for the question staff push on | all | Delete if it is still empty at 2:35 |

Export PDF at 2:40. Notebooks and `outputs/` go up with the deck as the workings.

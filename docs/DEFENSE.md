# Defense Pack

A tight, honest brief for the thesis defense. The strongest story here is not
"I built a big app" — it's "I found and fixed real ML methodology defects, and I
can *prove* the fix."

---

## 1. What it is (one breath)
A no-code data-mining web app (Streamlit) that takes a non-technical user from a
raw spreadsheet to a trained, downloadable ML model through a guided 6-step
workflow: **upload → data-quality assessment → preprocessing → visualization →
dashboards → machine learning** (train, evaluate, export, and predict on new data).
~12,000 lines of Python; 46 automated tests; verified end-to-end in a browser.

## 2. The engineering story — before → after
| Defect (before) | Fix (after) |
|---|---|
| Classification **crashed** on text/categorical columns | sklearn `Pipeline` + `ColumnTransformer` (one-hot encoding) |
| No feature scaling (SVM/LogReg/Ridge invalid) | `StandardScaler` inside the pipeline |
| **Data leakage** — preprocessing fit on the full dataset | fit on the **training split only**; CV refits per fold |
| Exported models unusable on raw data | the **fitted pipeline** (incl. preprocessing) is saved |
| Wrong multiclass ROC-AUC; fake "average precision" | corrected to scikit-learn's definitions |
| XSS / JS-injection via uploaded column names | escaped + sanitized |
| 0 tests | 46 pytest tests + a real-browser smoke test |

## 3. Prove it, don't claim it  ⭐
Run this live:
```bash
python scripts/show_no_leakage.py
```
It demonstrates two things:
- **Demo 1** — the fitted scaler's mean equals the *training* data's mean, **not**
  the full dataset's. The transformer never saw the test set.
- **Demo 2** — on a **random target** (no real signal), a common *leaky* approach
  (selecting features on the whole dataset before cross-validation) scores ~**0.65**
  (falsely optimistic), while this pipeline scores ~**0.50** (honest chance).

That single contrast is the most convincing thing you can show: leakage *inflates*
results; the rebuilt pipeline stays honest.

## 4. "Why not just use KNIME / Orange / RapidMiner / Excel?"  (you WILL be asked)
A three-part honest answer:
1. **It's not trying to beat them on features.** It's a focused, guided,
   *browser-only* workflow for non-technical users — no install, no node-graph,
   no formulas. Zero-friction is the point.
2. **The differentiator is correctness-by-construction.** The pipeline *enforces*
   leakage-free methodology — exactly the mistake general-purpose tools happily let
   you make. (See §3; it's a live demo, not a slogan.)
3. **Honest scope.** As a capstone, the value is the end-to-end engineering and the
   methodological rigor, not market displacement.

## 5. Honest limitations (say them first)
- Single developer; **no maintenance future** beyond graduation.
- The target user ("non-technical people who want data mining") is real but **not
  yet validated** with real users.
- Page-level feature engineering (e.g. datetime extraction) isn't part of the saved
  model, so the Predict tab needs those engineered columns in the uploaded file.
- High-cardinality categoricals are naively one-hot encoded (a known ML caveat).

## 6. Likely questions → short answers
- **Overfitting?** Train/test split, **stratified** CV, regularized defaults, and the
  pipeline refits preprocessing per fold (no leakage).
- **Categorical data?** One-hot in the `ColumnTransformer`, fit on train only.
- **Imbalanced data?** Stratified split + CV; `class_weight='balanced'` is a
  documented next step.
- **Deployed?** Deploy-ready — see `DEPLOY.md`; runs free on Streamlit Community Cloud.
- **Tested?** 46 pytest tests (unit + an integration smoke test) plus a real-browser
  end-to-end walkthrough (`app/tests/test_smoke_workflow.py` mirrors it in code).

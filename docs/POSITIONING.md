# Positioning — "Rigorous ML you can't get wrong"

A one-pager on how to frame this project (from the LLM-council synthesis). The
engineering is strong; the positioning is what turns it from "a project" into
"a thing with a reason to exist."

## The wedge
**Rigorous ML you can't get wrong.** Most no-code ML tools are methodologically
sloppy (they let you leak data, skip scaling, mis-encode categoricals). Most
*rigorous* tools require code. This sits in the gap: a no-code surface over a
pipeline that **enforces** correctness (leakage-free, scaled, encoded — see
`scripts/show_no_leakage.py`). That enforcement is the product, not a feature.

## First real user: an educator
The most plausible and reachable first user is **a professor or TA**, not a
business buyer. A tool where students *can't cheat the methodology* is what
data-science teachers want. One instructor adopting it for a course gives you the
three things a resume can't: **real users, real usage data, and a named
institution** — which doubles as defense evidence and a job story.

→ Concrete first move: pitch it to your own department for an intro course.

## What it is NOT competing on
Not feature count vs. KNIME / Power BI / Tableau. It competes on **correctness +
zero friction** (browser-only, no install, guided 6 steps, "try sample data" in
one click). Trying to out-feature mature incumbents is the losing game.

## Honest 12-month arc
- **Validate** with one captive cohort (a class) on the deployed URL — watch real
  people use it, fix the friction.
- **Iterate** toward whatever that cohort actually needs; consider a narrow
  vertical where "show your work / reproducibility" matters (clinical, ESG, ops).
- **If no adoption** materializes, it remains a strong, deployed **portfolio
  piece** with a real engineering story — a perfectly good outcome for a capstone.

## One-line elevator pitch
> "Upload a spreadsheet and build a prediction model in your browser — with the
> data-science rigor (no leakage, proper validation) that other no-code tools skip."

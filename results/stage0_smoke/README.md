# Stage 0 Smoke Results

This folder is a lightweight, checked-in record of Stage 0 smoke-test outcomes.

Use it for:
- quick zero-shot smoke runs
- small training smoke runs
- short experiment comparisons you want to keep alongside the repo

Keep this folder simple:
- store the raw run summaries as JSON
- store one short markdown summary per smoke session
- duplicate `TEMPLATE.md` for future runs

Suggested naming:
- raw JSON: `YYYY-MM-DD_<notebook_or_scope>_smoke.json`
- summary markdown: `YYYY-MM-DD_<scope>_summary.md`

Recommended workflow for future smoke tests:
1. Copy `TEMPLATE.md` to a dated file.
2. Paste the exact command you ran.
3. Paste the raw JSON run summaries into a dated JSON file.
4. Fill in the markdown summary with ranking, notes, and next step.

This keeps smoke tracking reviewable without expanding the experiment framework.

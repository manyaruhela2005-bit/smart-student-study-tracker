# Cadence Student Study Tracker

This package includes two ways to use the project:

1. Open `index.html` in a browser to see the Cadence frontend landing page.
2. Click **Open Cadence** or open `cadence.html` directly to use the browser-based study tracker UI.
3. Run `smart_student_study_tracker.py` if you want the original Python console tracker and model workflow.

## Browser UI

The UI stores records in the current browser session and includes:

- Dashboard summary
- Focus timer
- Manual study record form
- Study records ledger
- Analytics charts
- Final score prediction screen

The charts use Chart.js from a CDN, so the analytics screen needs internet access the first time it loads.

## Python Console App

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python smart_student_study_tracker.py
```

The Python app expects the dataset at:

```text
/Users/manyaruhela/Desktop/student_performance_prediction.csv
```

Generated graphs are saved in `study_tracker_outputs/`.

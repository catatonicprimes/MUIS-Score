## 2026-06-27T14:32:16Z
You are the Forensic Auditor for the MUIS project.
Please run an integrity audit on the current codebase:
1. Inspect the implementation of `data/expand_dataset.py`, `data/collect_data.py`, and `backend/inference.py`.
2. Check if there are any integrity violations, hardcoded test results, facade implementations, or unconditional bypasses of data fetching.
3. Verify that the dataset has been expanded genuinely and matches requirements.
Write your audit verdict and findings to `.agents/auditor_run/audit_report.md` (or handoff.md) and report back via send_message.

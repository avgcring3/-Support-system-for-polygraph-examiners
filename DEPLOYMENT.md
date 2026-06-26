# Deployment

GitHub Pages can only host static HTML/CSS/JS. This project needs a Python
runtime because the app parses uploaded files, builds features, loads/trains a
model, writes reports, and runs Streamlit. Use Streamlit Community Cloud for a
shareable public demo link.

## Streamlit Community Cloud

1. Open https://share.streamlit.io/.
2. Choose this repository:
   `avgcring3/-Support-system-for-polygraph-examiners`.
3. Branch: `master`.
4. Main file path:

```text
ui/streamlit_app.py
```

5. Python version is pinned in `runtime.txt`.
6. Dependencies are installed from `requirements.txt`.

On first launch the app builds generated artifacts from `data/raw` if
`models/best_model.pkl` is absent. This is needed because model and report files
are intentionally git-ignored.

## Local smoke check

```powershell
python -m pytest -q
streamlit run ui/streamlit_app.py
```

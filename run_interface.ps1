$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot
python -m streamlit run "$projectRoot\ui\streamlit_app.py" --server.port 8501

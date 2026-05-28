$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

Write-Host "[run_all] tests"
python -m pytest -q

Write-Host "[run_all] preprocess"
python src/preprocess.py

Write-Host "[run_all] train"
python src/train.py

Write-Host "[run_all] inference"
python src/predict.py --input-features data/processed/polygram_features.csv --model-path models/best_model.pkl --output-path reports/predictions_inference.csv

Write-Host "[run_all] done"

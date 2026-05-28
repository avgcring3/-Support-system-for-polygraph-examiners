param(
    [Parameter(Mandatory = $true)]
    [string]$XdexPath
)

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

python "$projectRoot\src\process_single_xdex.py" --xdex-path "$XdexPath" --project-root "$projectRoot"

$stem = [System.IO.Path]::GetFileNameWithoutExtension($XdexPath)
$reportPath = Join-Path $projectRoot "reports\single_xdex\$stem\report.html"

Write-Host ""
Write-Host "HTML-отчет:" $reportPath
Write-Host "PNG-схема:" (Join-Path $projectRoot "reports\single_xdex\$stem\question_route.png")
Write-Host "CSV-вопросов:" (Join-Path $projectRoot "reports\single_xdex\$stem\questions.csv")

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$releaseDir = Join-Path $projectRoot "release"
if (-not (Test-Path -LiteralPath $releaseDir)) {
    New-Item -ItemType Directory -Path $releaseDir | Out-Null
}

$dateTag = Get-Date -Format "yyyy-MM-dd_HHmm"
$zipName = "polygraph_dss_release_$dateTag.zip"
$zipPath = Join-Path $releaseDir $zipName
if (Test-Path -LiteralPath $zipPath) {
    Remove-Item -LiteralPath $zipPath -Force
}

$stagingRoot = Join-Path $releaseDir "_staging_$dateTag"
if (Test-Path -LiteralPath $stagingRoot) {
    Remove-Item -LiteralPath $stagingRoot -Recurse -Force
}
New-Item -ItemType Directory -Path $stagingRoot | Out-Null

$include = @(
    "api",
    "src",
    "ui",
    "tests",
    "data\raw",
    "data\processed",
    "data\examples",
    "models",
    "reports",
    "README.md",
    "requirements.txt",
    "docker-compose.yml",
    "Dockerfile",
    "run_all.ps1",
    "run_interface.ps1",
    "run_single_xdex.ps1",
    "stop_interface.ps1",
    "create_desktop_shortcut.ps1",
    "launch_ui_hidden.vbs",
    "Makefile",
    "pytest.ini",
    "package_release.ps1"
)

foreach ($rel in $include) {
    $src = Join-Path $projectRoot $rel
    if (-not (Test-Path -LiteralPath $src)) {
        continue
    }
    $dst = Join-Path $stagingRoot $rel
    if (Test-Path -LiteralPath $src -PathType Container) {
        New-Item -ItemType Directory -Path $dst -Force | Out-Null
        robocopy $src $dst /E /R:1 /W:1 /NFL /NDL /NJH /NJS /NP /XD __pycache__ .pytest_cache | Out-Null
        if ($LASTEXITCODE -gt 7) {
            throw "robocopy failed for $rel with code $LASTEXITCODE"
        }
    } else {
        $parent = Split-Path -Parent $dst
        if (-not (Test-Path -LiteralPath $parent)) {
            New-Item -ItemType Directory -Path $parent -Force | Out-Null
        }
        Copy-Item -LiteralPath $src -Destination $dst -Force
    }
}

$manifestPath = Join-Path $stagingRoot "RELEASE_CONTENTS.txt"
@(
    "Polygraph DSS Release Package",
    "Generated at: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")",
    "",
    "Included top-level entries:"
) | Set-Content -Path $manifestPath -Encoding UTF8

Get-ChildItem -LiteralPath $stagingRoot | Sort-Object Name | ForEach-Object {
    Add-Content -Path $manifestPath -Value $_.Name
}

Compress-Archive -Path (Join-Path $stagingRoot "*") -DestinationPath $zipPath -Force
Remove-Item -LiteralPath $stagingRoot -Recurse -Force

Write-Output "RELEASE_ZIP:$zipPath"

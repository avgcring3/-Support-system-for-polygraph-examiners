$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$desktopPath = [Environment]::GetFolderPath("Desktop")
$shortcutPath = Join-Path $desktopPath "Polygraph DSS.lnk"

if (Test-Path -LiteralPath $shortcutPath) {
    Remove-Item -LiteralPath $shortcutPath -Force
}

$wscriptExe = Join-Path $env:WINDIR "System32\wscript.exe"
$launcherVbs = Join-Path $projectRoot "launch_ui_hidden.vbs"

$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $wscriptExe
$shortcut.Arguments = "`"$launcherVbs`""
$shortcut.WorkingDirectory = $projectRoot
$shortcut.IconLocation = "$env:SystemRoot\System32\imageres.dll,102"
$shortcut.Description = "Launch Polygraph DSS interface"
$shortcut.Save()

Write-Host "Shortcut created:" $shortcutPath

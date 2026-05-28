$targets = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -like "*streamlit*ui\\streamlit_app.py*" }

if (-not $targets) {
    Write-Host "Interface process not found."
    exit 0
}

foreach ($proc in $targets) {
    try {
        Stop-Process -Id $proc.ProcessId -Force -ErrorAction Stop
        Write-Host "Stopped PID:" $proc.ProcessId
    } catch {
        Write-Host "Failed to stop PID:" $proc.ProcessId
    }
}

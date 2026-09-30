# Starts Docker Desktop after an unclean shutdown.
# A crash leaves Unix-socket files behind that Windows can neither open nor delete, and
# Docker Desktop's backend crashes trying to rename them. Moving their folders aside
# lets Docker recreate them. The moved folders are harmless; delete them after a reboot.

$ErrorActionPreference = "Stop"
$docker = "$env:LOCALAPPDATA\Programs\DockerDesktop\Docker Desktop.exe"
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"

Get-Process | Where-Object { $_.ProcessName -match "^(Docker Desktop|com\.docker\..*|docker)$" } |
    Stop-Process -Force -Confirm:$false -ErrorAction SilentlyContinue
Start-Sleep -Seconds 3

foreach ($dir in "$env:LOCALAPPDATA\Docker\run", "$env:LOCALAPPDATA\docker-secrets-engine") {
    if (Test-Path $dir) {
        Rename-Item $dir "$(Split-Path $dir -Leaf).stale-$stamp"
        Write-Host "Moved aside: $dir"
    }
}

Start-Process $docker
Write-Host "Waiting for the Docker engine..."
# docker writes to stderr while the engine is starting; that must not abort the script
$ErrorActionPreference = "Continue"
for ($i = 0; $i -lt 60; $i++) {
    cmd /c "docker info >nul 2>&1"
    if ($LASTEXITCODE -eq 0) { Write-Host "Docker is up."; exit 0 }
    Start-Sleep -Seconds 4
}
Write-Error "Docker did not start within 4 minutes; see $env:LOCALAPPDATA\Docker\log\host"

[CmdletBinding()]
param(
  [int]$ApiPort = 8000,
  [int]$UiPort = 3000,
  [int]$UiPortMax = 3010,
  [int]$ApiTimeoutSec = 120,
  [int]$UiTimeoutSec = 180,
  [switch]$SkipTests
)

$ErrorActionPreference = 'SilentlyContinue'
$repoRoot = Split-Path -Parent $PSScriptRoot
$logsDir = Join-Path $repoRoot '.riper/logs'
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null
Write-Host "Repo Root: $repoRoot"
Write-Host "Logs: $logsDir"

function Get-PidsOnPort {
  param([int]$Port)
  $pids = @()
  try {
    $pids += (Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue | Select-Object -Expand OwningProcess) | Sort-Object -Unique
  } catch {}
  if (-not $pids -or $pids.Count -eq 0) {
    try {
      $lines = netstat -ano | Select-String ":$Port" | ForEach-Object { $_.ToString() }
      foreach ($ln in $lines) {
        if ($ln -match "LISTENING\s+(\d+)$") { $pids += [int]$Matches[1] }
      }
      $pids = $pids | Sort-Object -Unique
    } catch {}
  }
  return $pids
}

function Stop-Port {
  param([int]$Port)
  Write-Host "Stopping any processes on port $Port ..."
  $pids = Get-PidsOnPort -Port $Port
  foreach ($procId in $pids) {
    try {
      if ($procId -and (Get-Process -Id $procId -ErrorAction SilentlyContinue)) {
        Write-Host "  Killing PID $procId"
        Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
      }
    } catch {}
  }
  Start-Sleep -Milliseconds 500
  # Double-check and attempt once more
  $pids2 = Get-PidsOnPort -Port $Port
  foreach ($pid2 in $pids2) {
    try { Stop-Process -Id $pid2 -Force -ErrorAction SilentlyContinue } catch {}
  }
}

function Test-Url {
  param([string]$Url, [int]$TimeoutSec = 60)
  $deadline = (Get-Date).AddSeconds($TimeoutSec)
  do {
    try {
# Find a free UI port if requested port is busy
function Get-FreePortInRange {
  param([int]$Start, [int]$End)
  for ($p = $Start; $p -le $End; $p++) {
    $busy = Get-PidsOnPort -Port $p
    if (-not $busy -or $busy.Count -eq 0) { return $p }
  }
  return $null
}

      $resp = Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 5
      if ($resp.StatusCode -ge 200 -and $resp.StatusCode -lt 500) {
        return $true
      }
    } catch {}
    Start-Sleep -Milliseconds 800
  } while ((Get-Date) -lt $deadline)
  return $false
}

# 1) Clean ports first
Stop-Port -Port $ApiPort
Stop-Port -Port $UiPort

# 2) Start API
Write-Host "Starting API on port $ApiPort ..."
$apiOut = Join-Path $logsDir 'api.out.log'
$apiErr = Join-Path $logsDir 'api.err.log'
$apiProc = Start-Process -FilePath "py" -ArgumentList "-3","-u","ui_api.py" -WorkingDirectory $repoRoot -RedirectStandardOutput $apiOut -RedirectStandardError $apiErr -PassThru
if (-not (Test-Url -Url "http://localhost:$ApiPort/api/health" -TimeoutSec $ApiTimeoutSec)) {
  Write-Error "API did not become healthy in $ApiTimeoutSec seconds."; exit 1
}
Write-Host "API healthy. PID: $($apiProc.Id)"

# 3) Start Next.js dev (ensure API base)
$env:NEXT_PUBLIC_API_BASE = "http://localhost:$ApiPort/api"
Write-Host "Starting Next.js dev on port $UiPort (npm run dev) ..."
$uiWorkingDir = (Join-Path $repoRoot 'ui')
$uiOut = Join-Path $logsDir 'ui.out.log'
$uiErr = Join-Path $logsDir 'ui.err.log'
$uiProc = Start-Process -FilePath "npm" -ArgumentList "run","dev","--","-p",$UiPort -WorkingDirectory $uiWorkingDir -RedirectStandardOutput $uiOut -RedirectStandardError $uiErr -PassThru
if (-not (Test-Url -Url "http://localhost:$UiPort/" -TimeoutSec $UiTimeoutSec)) {
  Write-Warning "npm run dev did not become reachable. Trying fallback: npx next dev ..."
  if ($uiProc) { try { Stop-Process -Id $uiProc.Id -Force } catch {} }
  $uiProc = Start-Process -FilePath "npx" -ArgumentList "next","dev","-p",$UiPort -WorkingDirectory $uiWorkingDir -RedirectStandardOutput $uiOut -RedirectStandardError $uiErr -PassThru
  if (-not (Test-Url -Url "http://localhost:$UiPort/" -TimeoutSec ($UiTimeoutSec + 30))) {
    Write-Warning "npx fallback did not work. Trying node binary fallback ..."
    if ($uiProc) { try { Stop-Process -Id $uiProc.Id -Force } catch {} }
    $nextBin = Join-Path $uiWorkingDir 'node_modules/next/dist/bin/next'
    if (-not (Test-Path $nextBin)) { $nextBin = Join-Path $uiWorkingDir 'node_modules/.bin/next.cmd' }
    $uiProc = Start-Process -FilePath "node" -ArgumentList $nextBin,"dev","-p",$UiPort -WorkingDirectory $uiWorkingDir -RedirectStandardOutput $uiOut -RedirectStandardError $uiErr -PassThru
  }
  if (-not (Test-Url -Url "http://localhost:$UiPort/" -TimeoutSec ($UiTimeoutSec + 60))) {
    Write-Error "UI did not become reachable after fallback attempts.";
    if ($apiProc) { try { Stop-Process -Id $apiProc.Id -Force } catch {} }
    if ($uiProc) { try { Stop-Process -Id $uiProc.Id -Force } catch {} }
    exit 1
  }
}
Write-Host "UI reachable. PID: $($uiProc.Id)"

if ($SkipTests) { Write-Host "SkipTests set; leaving servers running."; exit 0 }

# 4) Run Playwright tests (servers already running; config will reuse)
Push-Location (Join-Path $repoRoot 'ui')
try {
  Write-Host "Running Playwright e2e tests ..."
  npm run test:e2e
  $code = $LASTEXITCODE
} finally {
  Pop-Location
}

# 5) Cleanup servers
Write-Host "Cleaning up servers ..."
if ($uiProc) { try { Stop-Process -Id $uiProc.Id -Force } catch {} }
if ($apiProc) { try { Stop-Process -Id $apiProc.Id -Force } catch {} }

exit $code


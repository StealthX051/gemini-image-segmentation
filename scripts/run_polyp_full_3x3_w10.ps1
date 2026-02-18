param(
  [string]$RunId = ("polyp_full_3x3_w10_{0}" -f (Get-Date -Format "yyyyMMdd-HHmmss")),
  [switch]$NoLiveMonitor
)

$ErrorActionPreference = "Stop"

# Load .env into current process environment
Get-Content .env | ForEach-Object {
  if ($_ -match '^\s*#' -or $_ -match '^\s*$') { return }
  $n, $v = $_ -split '=', 2
  Set-Item -Path "Env:$n" -Value $v.Trim().Trim('"').Trim("'")
}

# Full-dataset override for polyp (no sample_size)
@"
defaults:
  workers: 10
  rate_limit: 0.5
datasets:
  - name: polyp
    root: segmented-images
"@ | Set-Content configs/benchmarks/polyp_full_w10.local.yaml

$batchDir = "results/batches/$RunId"
$statusPath = "$batchDir/job_status.jsonl"
$launcherLog = "$batchDir/launcher.log"
$launcherErr = "$batchDir/launcher.err.log"
$logsDir = "$batchDir/logs"

New-Item -ItemType Directory -Force $batchDir | Out-Null

$batchArgs = @(
  "-m", "gemini_segmentation.batch",
  "--config", "configs/benchmarks/ablation_robotics_canonical.yaml",
  "--overrides", "configs/benchmarks/polyp_full_w10.local.yaml",
  "--only-dataset", "polyp",
  "--run-id", $RunId
)

$proc = Start-Process `
  -FilePath "python" `
  -ArgumentList $batchArgs `
  -NoNewWindow `
  -PassThru `
  -RedirectStandardOutput $launcherLog `
  -RedirectStandardError $launcherErr

Write-Host "Started batch process PID=$($proc.Id)"
Write-Host "Launcher log: $launcherLog"
Write-Host "Status file:  $statusPath"
Write-Host "Job logs dir:  $logsDir"
Write-Host "Live monitor: status events + heartbeat every 30s + active log tail"

function Write-NewStatusEntries {
  param(
    [string]$Path,
    [int]$SeenCount
  )

  if (-not (Test-Path $Path)) {
    return @{
      Seen = $SeenCount
      CompletedDelta = 0
      FailedDelta = 0
    }
  }

  $lines = @(Get-Content $Path)
  if ($lines.Count -le $SeenCount) {
    return @{
      Seen = $SeenCount
      CompletedDelta = 0
      FailedDelta = 0
    }
  }

  # Select-Object -Skip always returns line items (not chars), even for a single new line.
  $newLines = @($lines | Select-Object -Skip $SeenCount)
  $completedDelta = 0
  $failedDelta = 0

  foreach ($line in $newLines) {
    $lineText = [string]$line
    if ([string]::IsNullOrWhiteSpace($lineText)) { continue }
    if ($lineText.Trim() -notmatch '^\{.*\}$') { continue }
    try {
      $entry = $lineText | ConvertFrom-Json
      $msg = "[{0}] phase={1} job={2} status={3} exit={4} duration_s={5}" -f `
        (Get-Date -Format "HH:mm:ss"), `
        $entry.phase, `
        $entry.job_id, `
        $entry.status, `
        $entry.exit_code, `
        $entry.duration_s
      Write-Host $msg

      if ($entry.status -eq "succeeded" -or $entry.status -eq "failed") {
        $completedDelta += 1
        if ($entry.status -eq "failed") {
          $failedDelta += 1
        }
      }
    } catch {
      Write-Host ("[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $lineText)
    }
  }

  return @{
    Seen = $lines.Count
    CompletedDelta = $completedDelta
    FailedDelta = $failedDelta
  }
}

if (-not $NoLiveMonitor) {
  $seen = 0
  $monitorStart = Get-Date
  $lastHeartbeat = Get-Date
  $heartbeatSeconds = 30
  $completed = 0
  $failed = 0
  $lastActiveLogPath = ""
  $lastActiveLogSize = -1
  $lastActiveTail = ""

  while (-not $proc.HasExited) {
    $statusUpdate = Write-NewStatusEntries -Path $statusPath -SeenCount $seen
    $seen = [int]$statusUpdate.Seen
    $completed += [int]$statusUpdate.CompletedDelta
    $failed += [int]$statusUpdate.FailedDelta

    if (Test-Path $logsDir) {
      $activeLog = Get-ChildItem $logsDir -Filter "*.log" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
      if ($null -ne $activeLog) {
        if ($activeLog.FullName -ne $lastActiveLogPath) {
          $lastActiveLogPath = $activeLog.FullName
          $lastActiveLogSize = -1
          $lastActiveTail = ""
          Write-Host ("[{0}] active_log={1}" -f (Get-Date -Format "HH:mm:ss"), $activeLog.Name)
        }

        if ($activeLog.Length -ne $lastActiveLogSize) {
          $tail = Get-Content $activeLog.FullName -Tail 1 -ErrorAction SilentlyContinue
          if ($tail -and -not [string]::IsNullOrWhiteSpace($tail) -and $tail -ne $lastActiveTail) {
            $displayTail = $tail
            if ($displayTail.Length -gt 220) {
              $displayTail = $displayTail.Substring(0, 220) + "..."
            }
            Write-Host ("[{0}] log_tail={1}" -f (Get-Date -Format "HH:mm:ss"), $displayTail)
            $lastActiveTail = $tail
          }
          $lastActiveLogSize = $activeLog.Length
        }
      }
    }

    $now = Get-Date
    if ((New-TimeSpan -Start $lastHeartbeat -End $now).TotalSeconds -ge $heartbeatSeconds) {
      $elapsed = (New-TimeSpan -Start $monitorStart -End $now)
      $elapsedFmt = "{0:00}:{1:00}:{2:00}" -f $elapsed.Hours, $elapsed.Minutes, $elapsed.Seconds
      Write-Host ("[{0}] heartbeat elapsed={1} completed={2}/3 failed={3}" -f `
        (Get-Date -Format "HH:mm:ss"), $elapsedFmt, $completed, $failed)
      $lastHeartbeat = $now
    }

    Start-Sleep -Seconds 2
    $proc.Refresh()
  }

  $statusUpdate = Write-NewStatusEntries -Path $statusPath -SeenCount $seen
  $seen = [int]$statusUpdate.Seen
  $completed += [int]$statusUpdate.CompletedDelta
  $failed += [int]$statusUpdate.FailedDelta
} else {
  $proc.WaitForExit()
}

# Ensure process metadata is finalized before reading ExitCode.
$proc.WaitForExit()
$proc.Refresh()

$exitCode = $proc.ExitCode
if ($null -eq $exitCode -and (Test-Path "$batchDir/summary.json")) {
  try {
    $summary = Get-Content "$batchDir/summary.json" | ConvertFrom-Json
    if ($summary.segment_jobs_failed -gt 0 -or $summary.fairness_jobs_failed -gt 0) {
      $exitCode = 1
    } else {
      $exitCode = 0
    }
  } catch {
    $exitCode = 1
  }
}
if ($null -eq $exitCode) { $exitCode = 1 }

if ($exitCode -ne 0) {
  Write-Error "Batch run failed with exit code $exitCode."
  Write-Host "Check logs:"
  Write-Host "  stdout: $launcherLog"
  Write-Host "  stderr: $launcherErr"
  exit $exitCode
}

Write-Host "Run complete: $RunId"
Write-Host "Summary: $batchDir/summary.json"
Write-Host "Status:  $statusPath"
Write-Host "Stdout:  $launcherLog"
Write-Host "Stderr:  $launcherErr"

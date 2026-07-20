$ErrorActionPreference = "Continue"

Set-Location -LiteralPath "D:\NVM_20235783\20252\GR1\UWSN\src"

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = Join-Path (Get-Location) "outputs\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$statusPath = Join-Path $logDir "overnight_status.txt"
$convLog = Join-Path $logDir "overnight_convergence_$timestamp.log"
$outputLog = Join-Path $logDir "overnight_outputs_$timestamp.log"
$plotLog = Join-Path $logDir "overnight_plot_$timestamp.log"

"STARTED $timestamp" | Out-File -FilePath $statusPath -Encoding utf8
"convergence_log=$convLog" | Out-File -FilePath $statusPath -Encoding utf8 -Append
"outputs_log=$outputLog" | Out-File -FilePath $statusPath -Encoding utf8 -Append
"plot_log=$plotLog" | Out-File -FilePath $statusPath -Encoding utf8 -Append

python run_convergence_first_refresh.py --seed-start 4000 --runs 1 --flush-every 10 *> $convLog
"CONVERGENCE_DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File -FilePath $statusPath -Encoding utf8 -Append

python run_experiment_inputs_parallel.py --workers 6 --flush-every 10 *> $outputLog
"OUTPUTS_DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File -FilePath $statusPath -Encoding utf8 -Append

python plot_experiment_outputs.py --include-leach *> $plotLog
"PLOT_DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File -FilePath $statusPath -Encoding utf8 -Append

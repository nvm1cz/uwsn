$ErrorActionPreference = "Continue"

Set-Location -LiteralPath "D:\NVM_20235783\20252\GR1\UWSN\src"

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = Join-Path (Get-Location) "outputs\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$statusPath = Join-Path $logDir "priority_status.txt"
$convLog = Join-Path $logDir "priority_convergence_$timestamp.log"
$outputLog = Join-Path $logDir "priority_outputs_$timestamp.log"
$plotLog = Join-Path $logDir "priority_plot_$timestamp.log"

"STARTED $timestamp" | Out-File -FilePath $statusPath -Encoding utf8
"scope=priority report figures, seeds=4000..4002, parameter_set=baseline, distribution=all, R=all, packet=4000/6400, energy=0.5/1.0" | Out-File -FilePath $statusPath -Encoding utf8 -Append
"convergence_log=$convLog" | Out-File -FilePath $statusPath -Encoding utf8 -Append
"outputs_log=$outputLog" | Out-File -FilePath $statusPath -Encoding utf8 -Append
"plot_log=$plotLog" | Out-File -FilePath $statusPath -Encoding utf8 -Append

python run_convergence_first_refresh.py --seed-start 4000 --runs 3 --parameter-set baseline --flush-every 10 *> $convLog
"CONVERGENCE_DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File -FilePath $statusPath -Encoding utf8 -Append

python run_experiment_inputs_parallel.py --workers 6 --flush-every 5 --seed-start 4000 --runs 3 --parameter-set baseline *> $outputLog
"OUTPUTS_DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File -FilePath $statusPath -Encoding utf8 -Append

python plot_experiment_outputs.py --include-leach --parameter-set baseline --figure-dir outputs/figures/priority_report *> $plotLog
"PLOT_DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File -FilePath $statusPath -Encoding utf8 -Append

# Define the path to the actions-runner folder
$runnerPath = (Resolve-Path -Path "../../../actions-runner").Path

# Verify the runner directory exists
if (-Not (Test-Path -Path $runnerPath)) {
    Write-Host "Error: The actions-runner directory does not exist at: $runnerPath" -ForegroundColor Red
    Exit 1
}

# Verify the run.cmd file exists
if (-Not (Test-Path -Path "$runnerPath/run.cmd")) {
    Write-Host "Error: The 'run.cmd' file does not exist in the actions-runner directory." -ForegroundColor Red
    Exit 1
}

# Start the GitHub Actions runner
Write-Host "Starting GitHub Actions runner from: $runnerPath" -ForegroundColor Green
& "$runnerPath/run.cmd"

# Keep the script running indefinitely (if required)
Write-Host "GitHub Actions runner is running. Press Ctrl+C to stop." -ForegroundColor Green
Start-Sleep -Seconds 3600
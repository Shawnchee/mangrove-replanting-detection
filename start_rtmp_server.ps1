# Start RTMP Server Script
# This script starts the nginx RTMP server for drone streaming

Write-Host "Starting RTMP Server for Mangrove Detection..." -ForegroundColor Green
Write-Host ""

# Set nginx path
$nginxPath = "C:\nginx-rtmp-win32-1.2.1"
$nginxExe = "$nginxPath\nginx.exe"

# Check if nginx exists
if (-not (Test-Path $nginxExe)) {
    Write-Host "Error: nginx.exe not found at $nginxExe" -ForegroundColor Red
    Write-Host "Please install nginx-rtmp-win32-1.2.1 in C:\" -ForegroundColor Yellow
    exit 1
}

# Check if nginx is already running
$nginxProcess = Get-Process -Name nginx -ErrorAction SilentlyContinue
if ($nginxProcess) {
    Write-Host "nginx is already running. Killing processes..." -ForegroundColor Yellow
    Stop-Process -Name nginx -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
}

# Start nginx in background
Write-Host "Starting nginx RTMP server..." -ForegroundColor Cyan
$startInfo = New-Object System.Diagnostics.ProcessStartInfo
$startInfo.FileName = $nginxExe
$startInfo.WorkingDirectory = $nginxPath
$startInfo.UseShellExecute = $false
$startInfo.CreateNoWindow = $true
[System.Diagnostics.Process]::Start($startInfo) | Out-Null

# Wait a moment for nginx to start
Start-Sleep -Seconds 2

# Check if nginx started successfully
$nginxProcess = Get-Process -Name nginx -ErrorAction SilentlyContinue
if ($nginxProcess) {
    Write-Host ""
    Write-Host "RTMP Server started successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "RTMP Stream URL: rtmp://localhost:1935/drone/dji" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "To stream from DJI drone, use the URL above in your streaming app." -ForegroundColor Cyan
    Write-Host "To stop the server, run: .\stop_rtmp_server.ps1" -ForegroundColor Cyan
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "Failed to start nginx. Check nginx error logs." -ForegroundColor Red
    exit 1
}
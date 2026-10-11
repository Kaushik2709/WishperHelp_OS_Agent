# WisprHelp Portable USB OS Agent PowerShell Launcher
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$UsbRoot = $ScriptDir

Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host "  WisprHelp Portable USB OS Agent" -ForegroundColor Green
Write-Host "  Mount Point: $UsbRoot" -ForegroundColor Yellow
Write-Host "===================================================================" -ForegroundColor Cyan

$PythonExe = "python"
if (Test-Path "$UsbRoot\runtime\python\python.exe") {
    $PythonExe = "$UsbRoot\runtime\python\python.exe"
    Write-Host "  Runtime    : Portable Python (USB)" -ForegroundColor Green
} else {
    Write-Host "  Runtime    : Host System Python" -ForegroundColor Green
}

$env:PYTHONPATH = $UsbRoot
$env:WISPRHELP_USB_ROOT = $UsbRoot

Write-Host "Starting WisprHelp in voice mode..." -ForegroundColor Cyan
& $PythonExe "$UsbRoot\wisprhelp\main.py" --voice $args

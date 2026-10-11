@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "USB_ROOT=%SCRIPT_DIR%"

echo ===================================================================
echo   WisprHelp Portable USB OS Agent
echo   Mount Point: %USB_ROOT%
echo ===================================================================

set "PYTHON_EXE=python"
if exist "%USB_ROOT%runtime\python\python.exe" (
    set "PYTHON_EXE=%USB_ROOT%runtime\python\python.exe"
    echo   Runtime    : Portable Python (USB)
) else (
    echo   Runtime    : Host System Python
)

set "PYTHONPATH=%USB_ROOT%"
set "WISPRHELP_USB_ROOT=%USB_ROOT%"

echo Starting WisprHelp...
"%PYTHON_EXE%" "%USB_ROOT%wisprhelp\main.py" --voice %*

if errorlevel 1 (
    echo.
    echo WisprHelp exited with code %ERRORLEVEL%
    pause
)

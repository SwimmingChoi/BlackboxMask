@echo off
cd /d "%~dp0"
"runtime\python.exe" "tools\diagnose.py"
pause

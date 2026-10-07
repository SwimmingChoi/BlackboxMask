@echo off
setlocal
cd /d "%~dp0"
echo Installing Intel NPU support into the bundled Python runtime.
echo Local npu-wheels are used when present. Otherwise internet access is required.
"runtime\python.exe" tools\install_npu.py
if errorlevel 1 goto fail
"runtime\python.exe" tools\diagnose.py
if errorlevel 1 goto fail
echo Installation completed. Select Intel NPU in the application and analyze a short video.
pause
exit /b 0
:fail
echo Installation or diagnostic failed. See the error above.
pause
exit /b 1

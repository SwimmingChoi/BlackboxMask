@echo off
setlocal
cd /d "%~dp0"
echo Run on an approved internet-connected PC with Python and pip.
py -m pip download --dest npu-wheels --platform win_amd64 --python-version 312 --implementation cp --abi cp312 --only-binary=:all: pip "openvino==2026.3.0"
if errorlevel 1 goto fail
echo Copy the entire npu-wheels folder to BlackboxMask on the target PC.
pause
exit /b 0
:fail
echo Download failed. Use an approved package source or ask your IT team.
pause
exit /b 1

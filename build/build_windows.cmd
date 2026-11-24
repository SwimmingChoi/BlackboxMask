@echo off
setlocal
cd /d "%~dp0.."
py -3.12 -m venv .build-venv
if errorlevel 1 goto fail
".build-venv\Scripts\python.exe" -m pip install -r requirements-npu.txt pyinstaller==6.22.0
if errorlevel 1 goto fail
".build-venv\Scripts\python.exe" -m unittest discover -s tests -v
if errorlevel 1 goto fail
".build-venv\Scripts\python.exe" -m PyInstaller --noconfirm --clean --windowed --onedir --name BlackboxMask --add-data "models;models" --add-data "licenses;licenses" --collect-all openvino --collect-all onnxruntime --collect-all av --collect-all cv2 launch.py
if errorlevel 1 goto fail
copy README_KO.txt dist\BlackboxMask\
copy THIRD_PARTY_NOTICES.md dist\BlackboxMask\
echo Build completed: dist\BlackboxMask\BlackboxMask.exe
echo Test the complete dist\BlackboxMask folder on a clean Windows PC before distribution.
pause
exit /b 0
:fail
echo Build failed. See the error above.
pause
exit /b 1

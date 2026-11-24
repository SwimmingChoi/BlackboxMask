@echo off
setlocal
rem Builds dist\BlackboxMask-<version>-Setup.exe from the portable bundle in this repo.
rem Requires NSIS 3.x (https://nsis.sourceforge.io). winget install NSIS.NSIS
cd /d "%~dp0.."
set "VERSION=0.1.2"
set "MAKENSIS=makensis"
where makensis >nul 2>nul || set "MAKENSIS=%ProgramFiles(x86)%\NSIS\makensis.exe"
if not exist dist mkdir dist
"%MAKENSIS%" /INPUTCHARSET UTF8 /DVERSION=%VERSION% build\installer.nsi
if errorlevel 1 goto fail
echo Installer created: dist\BlackboxMask-%VERSION%-Setup.exe
pause
exit /b 0
:fail
echo Installer build failed. See the error above.
pause
exit /b 1

; Blackbox Mask Windows installer (NSIS 3.x, Unicode).
; Build:  makensis -DVERSION=0.1.2 build/installer.nsi
; Output: dist/BlackboxMask-<VERSION>-Setup.exe
;
; Per-user install (no administrator rights), matching the portable build's policy.
; Default location: %LOCALAPPDATA%\Programs\BlackboxMask

Unicode true
SetCompressor /SOLID lzma
SetCompressorDictSize 64
ManifestDPIAware true

!ifndef VERSION
  !define VERSION "0.1.2"
!endif
!define APPNAME "Blackbox Mask"
!define APPKEY "BlackboxMask"
!define UNINSTKEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPKEY}"
!define SRC ".."

Name "${APPNAME} ${VERSION}"
OutFile "${SRC}\dist\BlackboxMask-${VERSION}-Setup.exe"
InstallDir "$LOCALAPPDATA\Programs\${APPKEY}"
InstallDirRegKey HKCU "Software\${APPKEY}" "InstallDir"
RequestExecutionLevel user
BrandingText "${APPNAME} ${VERSION}"

VIProductVersion "${VERSION}.0"
VIAddVersionKey /LANG=1042 "ProductName" "${APPNAME}"
VIAddVersionKey /LANG=1042 "FileDescription" "${APPNAME} 설치 프로그램"
VIAddVersionKey /LANG=1042 "FileVersion" "${VERSION}"
VIAddVersionKey /LANG=1042 "ProductVersion" "${VERSION}"
VIAddVersionKey /LANG=1042 "LegalCopyright" "See THIRD_PARTY_NOTICES.md"

!include "MUI2.nsh"
!include "x64.nsh"
!include "LogicLib.nsh"
!include "FileFunc.nsh"

!define MUI_ABORTWARNING
!define MUI_COMPONENTSPAGE_SMALLDESC
!define MUI_FINISHPAGE_RUN "$INSTDIR\BlackboxMask.exe"
!define MUI_FINISHPAGE_SHOWREADME "$INSTDIR\README_KO.txt"
!define MUI_FINISHPAGE_SHOWREADME_NOTCHECKED

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "${SRC}\THIRD_PARTY_NOTICES.md"
!insertmacro MUI_PAGE_COMPONENTS
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

!insertmacro MUI_LANGUAGE "Korean"

Function .onInit
  ${IfNot} ${RunningX64}
    MessageBox MB_ICONSTOP "Blackbox Mask는 64비트(x64) Windows 10/11에서만 설치할 수 있습니다."
    Abort
  ${EndIf}
FunctionEnd

Section "Blackbox Mask (필수)" SecMain
  SectionIn RO
  SetOutPath "$INSTDIR"
  ; Upgrading in place: drop old Python sources/caches so stale files do not linger.
  RMDir /r "$INSTDIR\__pycache__"

  File "${SRC}\BlackboxMask.exe"
  File "${SRC}\launch.py"
  File "${SRC}\app.py"
  File "${SRC}\engine.py"
  File "${SRC}\inference.py"
  File "${SRC}\detection_checks.py"
  File "${SRC}\npu_diagnostics.py"
  File "${SRC}\requirements.txt"
  File "${SRC}\requirements-npu.txt"
  File "${SRC}\Diagnose.cmd"
  File "${SRC}\Install_NPU.cmd"
  File "${SRC}\README_KO.txt"
  File "${SRC}\NPU_README_KO.txt"
  File "${SRC}\PATCH_README_KO.txt"
  File "${SRC}\VALIDATION.txt"
  File "${SRC}\NPU_VALIDATION.txt"
  File "${SRC}\THIRD_PARTY_NOTICES.md"
  File "${SRC}\BUILD_MANIFEST.json"
  File /r "${SRC}\models"
  File /r "${SRC}\licenses"
  File /r /x "__pycache__" "${SRC}\tools"
  File /r /x "__pycache__" "${SRC}\runtime"

  WriteUninstaller "$INSTDIR\Uninstall.exe"

  ; Start menu
  CreateDirectory "$SMPROGRAMS\${APPNAME}"
  CreateShortcut "$SMPROGRAMS\${APPNAME}\${APPNAME}.lnk" "$INSTDIR\BlackboxMask.exe" "" "$INSTDIR\BlackboxMask.exe" 0
  CreateShortcut "$SMPROGRAMS\${APPNAME}\문제 진단.lnk" "$INSTDIR\Diagnose.cmd"
  CreateShortcut "$SMPROGRAMS\${APPNAME}\사용 설명서.lnk" "$INSTDIR\README_KO.txt"
  CreateShortcut "$SMPROGRAMS\${APPNAME}\${APPNAME} 제거.lnk" "$INSTDIR\Uninstall.exe"

  ; Apps & features entry (current user)
  WriteRegStr HKCU "Software\${APPKEY}" "InstallDir" "$INSTDIR"
  WriteRegStr HKCU "${UNINSTKEY}" "DisplayName" "${APPNAME}"
  WriteRegStr HKCU "${UNINSTKEY}" "DisplayVersion" "${VERSION}"
  WriteRegStr HKCU "${UNINSTKEY}" "DisplayIcon" "$INSTDIR\BlackboxMask.exe"
  WriteRegStr HKCU "${UNINSTKEY}" "InstallLocation" "$INSTDIR"
  WriteRegStr HKCU "${UNINSTKEY}" "UninstallString" '"$INSTDIR\Uninstall.exe"'
  WriteRegStr HKCU "${UNINSTKEY}" "QuietUninstallString" '"$INSTDIR\Uninstall.exe" /S'
  WriteRegDWORD HKCU "${UNINSTKEY}" "NoModify" 1
  WriteRegDWORD HKCU "${UNINSTKEY}" "NoRepair" 1
SectionEnd

Section "바탕화면 바로가기" SecDesktop
  CreateShortcut "$DESKTOP\${APPNAME}.lnk" "$INSTDIR\BlackboxMask.exe" "" "$INSTDIR\BlackboxMask.exe" 0
SectionEnd

Section /o "Intel NPU 지원 (오프라인 설치)" SecNPU
  SetOutPath "$INSTDIR"
  File /r "${SRC}\npu-wheels"
  CreateShortcut "$SMPROGRAMS\${APPNAME}\Intel NPU 지원 설치.lnk" "$INSTDIR\Install_NPU.cmd"
  DetailPrint "Intel NPU 지원(OpenVINO)을 설치하는 중..."
  nsExec::ExecToLog '"$INSTDIR\runtime\python.exe" "$INSTDIR\tools\install_npu.py"'
  Pop $0
  ${If} $0 != 0
    MessageBox MB_ICONEXCLAMATION "Intel NPU 지원 설치에 실패했습니다 (코드 $0).$\n프로그램은 CPU로 사용할 수 있습니다.$\n나중에 시작 메뉴의 'Intel NPU 지원 설치'를 다시 실행하세요."
  ${EndIf}
SectionEnd

Section "-EstimatedSize"
  ; Measured after all selected sections so the size includes optional NPU files.
  ${GetSize} "$INSTDIR" "/S=0K" $0 $1 $2
  WriteRegDWORD HKCU "${UNINSTKEY}" "EstimatedSize" "$0"
SectionEnd

!insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
  !insertmacro MUI_DESCRIPTION_TEXT ${SecMain} "프로그램, Windows 실행 환경(Python), AI 모델"
  !insertmacro MUI_DESCRIPTION_TEXT ${SecDesktop} "바탕화면에 실행 아이콘을 만듭니다."
  !insertmacro MUI_DESCRIPTION_TEXT ${SecNPU} "Intel NPU(OpenVINO) 가속을 인터넷 없이 설치합니다. Intel Core Ultra PC에서만 필요합니다. (약 90MB)"
!insertmacro MUI_FUNCTION_DESCRIPTION_END

Section "Uninstall"
  ; Remove only what the installer created; never RMDir /r $INSTDIR itself,
  ; in case the user picked a shared folder as the install location.
  Delete "$INSTDIR\BlackboxMask.exe"
  Delete "$INSTDIR\launch.py"
  Delete "$INSTDIR\app.py"
  Delete "$INSTDIR\engine.py"
  Delete "$INSTDIR\inference.py"
  Delete "$INSTDIR\detection_checks.py"
  Delete "$INSTDIR\npu_diagnostics.py"
  Delete "$INSTDIR\requirements.txt"
  Delete "$INSTDIR\requirements-npu.txt"
  Delete "$INSTDIR\Diagnose.cmd"
  Delete "$INSTDIR\Install_NPU.cmd"
  Delete "$INSTDIR\README_KO.txt"
  Delete "$INSTDIR\NPU_README_KO.txt"
  Delete "$INSTDIR\PATCH_README_KO.txt"
  Delete "$INSTDIR\VALIDATION.txt"
  Delete "$INSTDIR\NPU_VALIDATION.txt"
  Delete "$INSTDIR\THIRD_PARTY_NOTICES.md"
  Delete "$INSTDIR\BUILD_MANIFEST.json"
  RMDir /r "$INSTDIR\__pycache__"
  RMDir /r "$INSTDIR\models"
  RMDir /r "$INSTDIR\licenses"
  RMDir /r "$INSTDIR\tools"
  RMDir /r "$INSTDIR\runtime"
  RMDir /r "$INSTDIR\npu-wheels"
  Delete "$INSTDIR\Uninstall.exe"
  RMDir "$INSTDIR"

  Delete "$DESKTOP\${APPNAME}.lnk"
  RMDir /r "$SMPROGRAMS\${APPNAME}"

  DeleteRegKey HKCU "${UNINSTKEY}"
  DeleteRegKey HKCU "Software\${APPKEY}"
  ; User logs in %LOCALAPPDATA%\BlackboxMask and saved projects/videos are kept.
SectionEnd

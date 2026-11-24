#define UNICODE
#define _UNICODE
#include <windows.h>
#include <wchar.h>

int WINAPI wWinMain(HINSTANCE instance,HINSTANCE previous,PWSTR args,int show) {
    wchar_t root[32768], exe[32768], script[32768], command[65536];
    DWORD count=GetModuleFileNameW(NULL,root,32768);
    if (!count || count>=32768) return 1;
    wchar_t *slash=wcsrchr(root,L'\\');
    if (!slash) return 1;
    *slash=0;
    if (wcslen(root)>30000) return 1;
    swprintf(exe,32768,L"%ls\\runtime\\pythonw.exe",root);
    swprintf(script,32768,L"%ls\\launch.py",root);
    swprintf(command,65536,L"\"%ls\" \"%ls\"",exe,script);
    STARTUPINFOW si={0}; si.cb=sizeof(si);
    PROCESS_INFORMATION pi={0};
    if (!CreateProcessW(exe,command,NULL,NULL,FALSE,CREATE_NO_WINDOW,NULL,root,&si,&pi)) {
        MessageBoxW(NULL,L"Unable to start. Extract the entire ZIP first. Keep runtime, models and source files beside BlackboxMask.exe. See README_KO.txt.",L"Blackbox Mask",MB_ICONERROR);
        return 1;
    }
    CloseHandle(pi.hThread);CloseHandle(pi.hProcess);
    return 0;
}

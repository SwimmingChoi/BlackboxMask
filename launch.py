"""Desktop entrypoint, with startup error capture for a windowed Windows app."""
import os
import sys
import traceback
from pathlib import Path

root=Path(__file__).resolve().parent
sys.path.insert(0,str(root))
handles=[]
if os.name=="nt":
    for directory in [root/"runtime",root/"runtime/Lib/site-packages/PySide6"]:
        if directory.is_dir():handles.append(os.add_dll_directory(str(directory)))
log_dir=Path(os.getenv("LOCALAPPDATA",str(Path.home())))/"BlackboxMask"
log_dir.mkdir(parents=True,exist_ok=True)
log=log_dir/"startup.log"
sys.stdout=open(log,"a",encoding="utf-8",buffering=1)
sys.stderr=sys.stdout
try:
    from app import main
    sys.exit(main())
except Exception:
    traceback.print_exc()
    if os.name=="nt":
        import ctypes
        ctypes.windll.user32.MessageBoxW(None,f"프로그램을 시작하지 못했습니다.\n오류 기록: {log}","Blackbox Mask",0x10)
    raise

"""Rebuild the portable bundle from an official embedded ZIP and Windows wheels.

Example (from any OS with Zig installed):
  python build/package_windows.py --python-zip python-3.12.10-embed-amd64.zip \
      --wheels windows_wheels --zig zig
The supplied wheel directory must contain every requirement and its dependencies,
downloaded using pip download --platform win_amd64 --python-version 312
--implementation cp --abi cp312 --only-binary=:all: -r requirements.txt msvc-runtime.
"""
import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        while data:=f.read(4*1024*1024):h.update(data)
    return h.hexdigest()


def build(python_zip,wheels,zig):
    root=Path(__file__).resolve().parents[1]
    runtime=root/'runtime';site=runtime/'Lib/site-packages';site.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(python_zip) as z:z.extractall(runtime)
    inputs={Path(python_zip).name:digest(python_zip)}
    for wheel in sorted(Path(wheels).glob('*.whl')):
        inputs[wheel.name]=digest(wheel)
        with zipfile.ZipFile(wheel) as z:
            for name in z.namelist():
                if name.endswith('/'):continue
                if '.data/' in name:
                    category,_,relative=name.split('.data/',1)[1].partition('/')
                    if category in ('platlib','purelib'):target=site/relative
                    elif category=='data':
                        target=runtime/(Path(relative).name if relative.startswith('Scripts/') and relative.endswith('.dll') else relative)
                    else:continue
                else:target=site/name
                if root not in target.resolve().parents:raise ValueError('Unsafe archive path')
                target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(name))
    (runtime/'python312._pth').write_text('python312.zip\n.\n..\nLib/site-packages\nimport site\n',encoding='utf-8')
    subprocess.run([zig,'cc','-target','x86_64-windows-gnu','-municode','-Wl,--subsystem,windows',
                    '-O2','-s',str(root/'build/launcher.c'),'-o',str(root/'BlackboxMask.exe')],check=True)
    files={str(p.relative_to(root)).replace('\\','/'):digest(p) for p in sorted(root.rglob('*'))
           if p.is_file() and '__pycache__' not in p.parts and p.name!='BUILD_MANIFEST.json'}
    (root/'BUILD_MANIFEST.json').write_text(json.dumps({'inputs':inputs,'files':files},indent=2),encoding='utf-8')


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--python-zip',required=True);a.add_argument('--wheels',required=True);a.add_argument('--zig',default='zig')
    args=a.parse_args();build(args.python_zip,args.wheels,args.zig)

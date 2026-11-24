"""Install optional NPU dependencies using the bundled embedded Python."""
import importlib.util
import subprocess
import sys
import tempfile
import traceback
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def validate_bootstrap(data):
    prefix = data.lstrip()[:1024].lower()
    if b"<html" in prefix or b"<!doctype" in prefix or b"<iframe" in prefix:
        raise RuntimeError("Download returned an HTML page (possibly an organization network block). "
                           "Use approved offline wheels in npu-wheels; do not execute this download.")
    try:
        compile(data, "get-pip.py", "exec")
    except (SyntaxError, ValueError) as error:
        raise RuntimeError("Downloaded pip bootstrap is not valid Python. Use approved offline wheels.") from error


def install():
    print("Python:", sys.executable, flush=True)
    wheels = ROOT / "npu-wheels"
    offline = wheels.is_dir() and any(wheels.glob("*.whl"))
    if offline:
        print("Installing from local offline wheels:", wheels, flush=True)
        if importlib.util.find_spec("pip") is None:
            pip_wheels = sorted(wheels.glob("pip-*-py3-none-any.whl"))
            if len(pip_wheels) != 1:
                raise RuntimeError("Place exactly one pip-*-py3-none-any.whl in npu-wheels.")
            # pip runs directly from its wheel, so embedded Python needs no bootstrap.
            sys.path.insert(0, str(pip_wheels[0]))
        from pip._internal.cli.main import main as pip_main
        result = pip_main(["install", "--no-index", "--find-links", str(wheels),
                           "--only-binary=:all:", "--no-warn-script-location", "openvino==2026.3.0"])
        if result:
            raise RuntimeError("Offline installation failed. Check that all dependency wheels are present.")
        print("Offline OpenVINO installation completed.", flush=True)
        return
    if importlib.util.find_spec("pip") is None:
        print("Downloading pip bootstrap from bootstrap.pypa.io...", flush=True)
        # A temporary script lets embedded Python install pip into its own runtime.
        # HTTPS uses the system certificate store; do not bypass TLS verification.
        with tempfile.TemporaryDirectory(prefix="blackbox-npu-install-") as directory:
            bootstrap = Path(directory) / "get-pip.py"
            with urllib.request.urlopen("https://bootstrap.pypa.io/get-pip.py", timeout=60) as response:
                data = response.read()
            validate_bootstrap(data)
            bootstrap.write_bytes(data)
            subprocess.run([sys.executable, str(bootstrap), "--no-warn-script-location"], check=True)
    # Existing application dependencies are already bundled. Only add OpenVINO.
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-warn-script-location",
                    "openvino==2026.3.0"], check=True)
    print("OpenVINO installation completed.", flush=True)


if __name__ == "__main__":
    try:
        install()
    except Exception:
        traceback.print_exc()
        print("Installation failed. Check internet access, proxy settings, and folder write permission.", flush=True)
        sys.exit(1)

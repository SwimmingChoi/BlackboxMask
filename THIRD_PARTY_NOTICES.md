# Third-party components

This prototype includes unmodified binary distributions, model files and their
available notices. The application's Python files and launcher source are included
so the application can be inspected, changed and rebuilt. No warranty or
commercial redistribution clearance is asserted by this prototype.

| Component | Version / source | Upstream |
|---|---|---|
| CPython Windows embedded | 3.12.10, official amd64 ZIP | https://www.python.org/downloads/release/python-31210/ |
| PySide6 Essentials / Shiboken6 | 6.8.3 Windows wheels | https://code.qt.io/pyside/pyside-setup.git/ |
| Qt | bundled by PySide6 6.8.3 | https://download.qt.io/archive/qt/6.8/6.8.3/ |
| OpenCV headless | 4.11.0.86 Windows wheel | https://github.com/opencv/opencv-python |
| NumPy | 2.2.6 Windows wheel | https://github.com/numpy/numpy/tree/v2.2.6 |
| ONNX Runtime | 1.22.1 Windows wheel | https://github.com/microsoft/onnxruntime/tree/v1.22.1 |
| PyAV | 16.1.0 Windows wheel | https://github.com/PyAV-Org/PyAV/tree/v16.1.0 |
| FFmpeg and codecs | shared libraries bundled in the PyAV wheel | https://github.com/PyAV-Org/pyav-ffmpeg |
| Microsoft VC runtime | msvc-runtime 14.44.35112 wheel; CPython/Qt runtime DLLs | https://pypi.org/project/msvc-runtime/14.44.35112/ |
| Head (face) model | yolox_n_body_head_face_handLR_foot_dist_Nx3xHxW.onnx from resources_n.tar.gz, Apache-2.0, Copyright 2024 Katsuya Hyodo | https://github.com/PINTO0309/PINTO_model_zoo/tree/main/445_YOLOX-Body-Head-Face-HandLR-Foot-Dist |
| License-plate model | yolo-v9-t-640-license-plates-end2end.onnx | https://github.com/ankandrew/open-image-models/releases/tag/assets |

The head model's Apache-2.0 license and the Open Image Models MIT notice are in `licenses/`.
The head model file is included unmodified.
CPython's license is in `runtime/LICENSE.txt`. Each Python distribution's notices
are retained under `runtime/Lib/site-packages/*.dist-info/`, often in `licenses/`.
Dependencies include their own third-party libraries. PySide6/Qt use LGPL/GPL or
commercial terms depending on the component. PyAV is BSD; its bundled FFmpeg and
codec libraries have separate licenses, including LGPL/GPL libraries. Microsoft's
runtime has its own redistribution terms. A wrapper's license does not override
the license of its binary dependencies.

The mutable `main` model URLs were downloaded once for this prototype; exact
downloaded file hashes are recorded in BUILD_MANIFEST.json. Runtime inference
does not perform downloads. Changing a model requires its own format and quality
validation. General plate detection is not proof of Korean dashcam performance.

Transitive Python packages are included from PyPI wheels with their original
metadata: coloredlogs, humanfriendly, flatbuffers, packaging, protobuf, sympy,
mpmath. Exact versions and wheel hashes are listed in BUILD_MANIFEST.json.

These files are provided as an evaluation prototype. Before wider redistribution,
the distributor must evaluate applicable third-party notice, source, relinking,
and other obligations for the chosen binaries. No trademark rights are granted.

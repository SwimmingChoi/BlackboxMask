import sys
import platform
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
print('Platform:',platform.platform())
print('Python:',sys.version)
try:
    import cv2,numpy,av,onnxruntime
    from PySide6 import QtCore
    print('OpenCV:',cv2.__version__,'NumPy:',numpy.__version__)
    print('PyAV:',av.__version__,'ONNX:',onnxruntime.__version__,'Qt:',QtCore.__version__)
    print('Encoder:',av.Codec('libx264','w').name)
    from engine import Detectors
    print('AI inference:',Detectors({'faces':True,'plates':True}).detect(numpy.zeros((96,160,3),numpy.uint8)))
    try:
        import openvino as ov
    except ImportError:
        print('NPU: optional OpenVINO package missing. Run Install_NPU.cmd for Intel NPU support.')
    else:
        core = ov.Core()
        print('OpenVINO:',ov.__version__,'Devices:',core.available_devices)
        for device in core.available_devices:
            print(device,core.get_property(device,'FULL_DEVICE_NAME'))
        # Compile and execute both actual models; discovery alone is insufficient.
        npu = Detectors({'faces':True,'plates':True,'backend':'npu'})
        npu.detect(numpy.zeros((720,1280,3),numpy.uint8))
        print('NPU inference:',npu.status())
    print('PASS: runtime and models loaded. Test the GUI and your videos next.')
except Exception:
    import traceback
    traceback.print_exc()
    sys.exit(1)

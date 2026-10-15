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
            try:driver=core.get_property(device,'DRIVER_VERSION')
            except Exception:driver='-'
            print(device,core.get_property(device,'FULL_DEVICE_NAME'),'· driver',driver)
        # Compile and execute both actual models; discovery alone is insufficient.
        npu = Detectors({'faces':True,'plates':True,'backend':'npu'})
        npu.detect(numpy.zeros((720,1280,3),numpy.uint8))
        print('NPU inference:',npu.status())
        # Per-frame NPU time of each model set on a 1080p frame, after the first compile.
        import time
        from engine import MODEL_SETS, NPU_UNSAFE_MODELS
        frame = numpy.random.default_rng(0).integers(0,255,(1080,1920,3),numpy.uint8)
        for backend in ('npu','gpu'):
            for model in MODEL_SETS:
                if backend == 'npu' and model in NPU_UNSAFE_MODELS:
                    # 시험만 해도 NPU가 멈춰 재부팅 전까지 사라지므로 진단에서도 실행하지 않는다.
                    print(f'NPU {model}: SKIPPED (큰 모델은 NPU 장치 멈춤 위험 - Intel GPU 사용)');continue
                try:
                    started = time.monotonic()
                    detector = Detectors({'faces':True,'plates':True,'backend':backend,'model':model,'verify_npu':False})
                    detector.detect(frame)
                    compiled = time.monotonic()-started
                    started = time.monotonic()
                    for _ in range(5):detector.detect(frame)
                    print(f'{backend.upper()} {model}: {(time.monotonic()-started)/5*1000:.0f} ms/frame (first run incl. compile {compiled:.1f} s)')
                except Exception as error:
                    print(f'{backend.upper()} {model}: FAILED {str(error).splitlines()[0]}')
    print('PASS: runtime and models loaded. Test the GUI and your videos next.')
except Exception:
    import traceback
    traceback.print_exc()
    sys.exit(1)

"""Same-frame ORT CPU / OpenVINO FP32 CPU / NPU comparison."""
import platform
import hashlib
import time
from types import SimpleNamespace
import numpy as np
from detection_checks import compare_boxes

class RecordedSession:
    def __init__(self,session):self.session=session;self.calls=[];self.raw=[]
    def get_inputs(self):return self.session.get_inputs()
    def run(self,outputs,feed):
        started=time.monotonic();result=self.session.run(outputs,feed)
        self.raw.append([x.copy() for x in result])
        self.calls.append({'input_shape':list(next(iter(feed.values())).shape),
            'input_sha256':hashlib.sha256(next(iter(feed.values())).tobytes()).hexdigest(),
            'seconds':time.monotonic()-started,'outputs':[
                {'shape':list(x.shape),'dtype':str(x.dtype),'finite':bool(np.isfinite(x).all()),
                 'min':float(x[np.isfinite(x)].min()) if np.isfinite(x).any() else None,
                 'max':float(x[np.isfinite(x)].max()) if np.isfinite(x).any() else None} for x in result]})
        return result

class OpenVINOCPU:
    def __init__(self,core,data):
        self.core=core;self.model=core.read_model(model=data);self.compiled={}
        self.name=self.model.input(0).get_any_name()
    def get_inputs(self):return [SimpleNamespace(name=self.name)]
    def run(self,outputs,feed):
        tensor=next(iter(feed.values()));shape=tuple(tensor.shape)
        if shape not in self.compiled:
            model=self.model.clone();model.reshape({self.name:list(shape)})
            self.compiled[shape]=self.core.compile_model(model,'CPU',{'INFERENCE_PRECISION_HINT':'f32'})
        compiled=self.compiled[shape];result=compiled([tensor])
        return [result[p].copy() for p in compiled.outputs]

def diagnose_image(image,settings,progress=lambda *a:None,cancel=None):
    import openvino as ov
    import onnxruntime as ort
    from engine import Detectors,model_bytes,MODEL_HASHES,check_cancel
    core=ov.Core()
    raw_runs={}
    report={'platform':platform.platform(),'openvino':ov.__version__,'onnxruntime':ort.__version__,
            'model_hashes':MODEL_HASHES,'image_shape':list(image.shape),'settings':settings,
            'devices':{},'runs':{},'comparisons':{}}
    for device in core.available_devices:
        properties={}
        for key in ('FULL_DEVICE_NAME','DRIVER_VERSION'):
            try:properties[key]=str(core.get_property(device,key))
            except Exception:pass
        report['devices'][device]=properties
    for index,backend in enumerate(('ort_cpu','openvino_cpu','npu')):
        check_cancel(cancel);progress(index*30,f'동일 프레임 비교: {backend}')
        try:
            detector=Detectors(dict(settings,backend='npu' if backend=='npu' else 'cpu',verify_npu=False))
            calls={};recorders={}
            for attr,name in (('face','head_yolox.onnx'),('plate','plate_yolov9.onnx')):
                session=getattr(detector,attr)
                if session is not None:
                    if backend=='openvino_cpu':session=OpenVINOCPU(core,model_bytes(name))
                    recorded=RecordedSession(session);setattr(detector,attr,recorded);calls[attr]=recorded.calls;recorders[attr]=recorded
            started=time.monotonic();boxes=detector.detect(image)
            report['runs'][backend]={'boxes':boxes,'seconds':time.monotonic()-started,'calls':calls}
            raw_runs[backend]=recorders
        except Exception as error:report['runs'][backend]={'error':str(error)}
    for backend in ('openvino_cpu','npu'):
        reference=report['runs']['ort_cpu'];candidate=report['runs'][backend]
        if 'boxes' in reference and 'boxes' in candidate:
            report['comparisons'][backend]={kind:compare_boxes(
                [x for x in reference['boxes'] if x['kind']==kind],
                [x for x in candidate['boxes'] if x['kind']==kind]) for kind in ('face','plate')}
            numeric={}
            for kind,recorder in raw_runs['ort_cpu'].items():
                other=raw_runs[backend][kind];differences=[]
                for index,(a_outputs,b_outputs) in enumerate(zip(recorder.raw,other.raw)):
                    item={'same_input':recorder.calls[index]['input_sha256']==other.calls[index]['input_sha256'],'outputs':[]}
                    for a,b in zip(a_outputs,b_outputs):
                        summary={'same_shape':a.shape==b.shape}
                        if a.shape==b.shape and a.size and np.isfinite(a).all() and np.isfinite(b).all():
                            delta=np.abs(a.astype(np.float64)-b.astype(np.float64))
                            summary.update(max_absolute_difference=float(delta.max()),mean_absolute_difference=float(delta.mean()))
                        item['outputs'].append(summary)
                    differences.append(item)
                numeric[kind]=differences
            report.setdefault('numeric_comparisons',{})[backend]=numeric
    report['note']='CPU 일치율은 정답 기준 탐지 정확도가 아닙니다. 시간은 최초 컴파일을 포함합니다.'
    progress(100,'비교 완료');return report

"""Optional Intel NPU / Intel GPU (OpenVINO) sessions with an explicit, observable fallback."""
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np

# The plate model ends with NonMaxSuppression, whose dynamic output comes back empty on NPU.
# Run the static-shape part up to this node on NPU and repeat the NMS tail on CPU.
NPU_SPLITS = {"plate_yolov9.onnx": "/end2end/Transpose", "plate_yolov9_s.onnx": "/end2end/Transpose"}


def end2end_nms(raw, iou_threshold=.45, score_threshold=.001, limit=100):
    """Same as the model's end2end tail: rows [batch, x1, y1, x2, y2, class, score]."""
    pred = raw[0]
    cx, cy, w, h = pred[:, :4].T
    boxes = np.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2], 1)
    score, cls = pred[:, 4:].max(1), pred[:, 4:].argmax(1)
    shifted = boxes + (cls*640.)[:, None]  # class offset keeps NMS per class, as in the model
    area = (shifted[:, 2]-shifted[:, 0])*(shifted[:, 3]-shifted[:, 1])
    order = np.flatnonzero(score > score_threshold)
    order = order[np.argsort(-score[order], kind="stable")]
    keep = []
    while order.size and len(keep) < limit:
        i, order = order[0], order[1:]
        keep.append(i)
        inter = (np.clip(np.minimum(shifted[i, 2], shifted[order, 2])-np.maximum(shifted[i, 0], shifted[order, 0]), 0, None)
                 * np.clip(np.minimum(shifted[i, 3], shifted[order, 3])-np.maximum(shifted[i, 1], shifted[order, 1]), 0, None))
        union = area[i]+area[order]-inter
        order = order[inter <= iou_threshold*np.where(union > 0, union, 1)]
    keep = np.array(keep, dtype=int)
    return np.concatenate([np.zeros((len(keep), 1)), boxes[keep], cls[keep, None], score[keep, None]], 1).astype(np.float32)


def npu_model(core, name, data):
    """Model to compile for NPU and the CPU post-processing that restores the ONNX output."""
    model = core.read_model(model=data)
    split = NPU_SPLITS.get(name)
    if split is None:
        return model, None
    import openvino as ov
    node = next(op for op in model.get_ordered_ops() if op.get_friendly_name() == split)
    return ov.Model([node.output(0)], model.get_parameters(), name), end2end_nms


LABELS = {"NPU": "Intel NPU", "GPU": "Intel GPU"}


def find_device(core, kind):
    """OpenVINO device id for an Intel NPU or Intel GPU. Non-Intel GPUs (e.g. NVIDIA via OpenCL) are skipped."""
    for d in core.available_devices:
        if kind == "NPU" and (d == "NPU" or d.startswith("NPU.")):
            return d
        if kind == "GPU" and (d == "GPU" or d.startswith("GPU.")):
            try:
                if "intel" in str(core.get_property(d, "FULL_DEVICE_NAME")).lower():
                    return d
            except Exception:
                continue
    hint = "Windows NPU 드라이버를" if kind == "NPU" else "Intel 그래픽 드라이버를"
    raise RuntimeError(f"{LABELS[kind]} 장치가 감지되지 않습니다. {hint} 확인하세요.")


class NPUSession:
    """OpenVINO session on an Intel NPU (default) or Intel GPU.

    On failure it switches to cpu_factory(), which may itself be another NPUSession (auto: NPU → GPU → CPU)."""
    def __init__(self, name, data, cpu_factory, strict=False, kind="NPU"):
        self.name = name
        self.cpu_factory = cpu_factory
        self.strict = strict
        self.kind = kind; self.label = LABELS[kind]
        self.cpu = None
        self.compiled = {}
        self.backend = f"{self.label} (준비 중)"
        self.reason = ""
        try:
            import openvino as ov
            self.core = ov.Core()
            self.device = find_device(self.core, kind)
            self.model, self.postprocess = npu_model(self.core, name, data)
            self.input_name = self.model.input(0).get_any_name()
            cache = Path(os.getenv("LOCALAPPDATA", str(Path.home()))) / "BlackboxMask" / "npu-cache"
            cache.mkdir(parents=True, exist_ok=True)
            self.config = {"CACHE_DIR": str(cache)}
        except Exception as e:
            self._fallback(e)

    def _fallback(self, error):
        if self.cpu is not None and hasattr(self.cpu, "_fallback"):
            # 이미 다음 장치로 넘어간 상태에서 그 장치가 실패하면 그 장치의 대체 경로를 따른다.
            self.cpu._fallback(error); return
        self.reason = f"{self.label}: {error}"
        if self.strict:
            raise RuntimeError(f"{self.name}: {self.label.split()[-1]} 실행 실패: {error}\nInstall_NPU.cmd로 환경을 설치하고 Diagnose.cmd로 확인하세요.") from error
        self.cpu = self.cpu_factory()
        self.backend = "CPU"
        self.compiled.clear()

    def get_inputs(self):
        return self.cpu.get_inputs() if self.cpu else [SimpleNamespace(name=self.input_name)]

    def run(self, outputs, feed):
        if self.cpu:
            return self.cpu.run(outputs, feed)
        tensor = next(iter(feed.values()))
        try:
            shape = tuple(tensor.shape)
            if shape not in self.compiled:
                # Static input specialization supports older Intel NPUs too.
                model = self.model.clone()
                model.reshape({self.input_name: list(shape)})
                compiled = self.core.compile_model(model, self.device, self.config)
                if len(self.compiled) >= 8:
                    self.compiled.clear()
                self.compiled[shape] = compiled
            compiled = self.compiled[shape]
            result = compiled([tensor])
            self.backend = self.label
            # Preserve the ONNX output order consumed by the existing decoders.
            outputs = [result[port].copy() for port in compiled.outputs]
            return [self.postprocess(outputs[0])] if self.postprocess else outputs
        except Exception as e:
            self._fallback(e)
            return self.cpu.run(outputs, {self.cpu.get_inputs()[0].name: tensor})

    def status(self):
        if self.cpu is not None and hasattr(self.cpu, "status"):
            inner = self.cpu.status()
            return {"device": inner["device"], "fallback_reason": " / ".join(x for x in (self.reason, inner["fallback_reason"]) if x)}
        return {"device": self.backend, "fallback_reason": self.reason}

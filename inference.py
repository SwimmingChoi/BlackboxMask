"""Optional Intel NPU sessions with an explicit, observable CPU fallback."""
import os
from pathlib import Path
from types import SimpleNamespace


class NPUSession:
    def __init__(self, name, data, cpu_factory, strict=False):
        self.name = name
        self.cpu_factory = cpu_factory
        self.strict = strict
        self.cpu = None
        self.compiled = {}
        self.backend = "Intel NPU (준비 중)"
        self.reason = ""
        try:
            import openvino as ov
            self.core = ov.Core()
            if not any(d == "NPU" or d.startswith("NPU.") for d in self.core.available_devices):
                raise RuntimeError("Intel NPU 장치가 감지되지 않습니다. Windows NPU 드라이버를 확인하세요.")
            self.model = self.core.read_model(model=data)
            self.input_name = self.model.input(0).get_any_name()
            cache = Path(os.getenv("LOCALAPPDATA", str(Path.home()))) / "BlackboxMask" / "npu-cache"
            cache.mkdir(parents=True, exist_ok=True)
            self.config = {"CACHE_DIR": str(cache)}
        except Exception as e:
            self._fallback(e)

    def _fallback(self, error):
        self.reason = str(error)
        if self.strict:
            raise RuntimeError(f"{self.name}: NPU 실행 실패: {error}\nInstall_NPU.cmd로 환경을 설치하고 Diagnose.cmd로 확인하세요.") from error
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
                compiled = self.core.compile_model(model, "NPU", self.config)
                if len(self.compiled) >= 8:
                    self.compiled.clear()
                self.compiled[shape] = compiled
            compiled = self.compiled[shape]
            result = compiled([tensor])
            self.backend = "Intel NPU"
            # Preserve the ONNX output order consumed by the existing decoders.
            return [result[port].copy() for port in compiled.outputs]
        except Exception as e:
            self._fallback(e)
            return self.cpu.run(outputs, {self.cpu.get_inputs()[0].name: tensor})

    def status(self):
        return {"device": self.backend, "fallback_reason": self.reason}

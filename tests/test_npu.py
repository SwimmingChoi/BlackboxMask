"""Device dispatch tests runnable without video or NPU dependencies."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference import NPUSession


class Array:
    shape = (1, 3, 32, 32)
    def copy(self):
        return self


class Model:
    def input(self, index):
        return SimpleNamespace(get_any_name=lambda: "images")
    def clone(self):
        return self
    def reshape(self, shape):
        self.shape = shape


class NPUTests(unittest.TestCase):
    def fake_core(self, devices=("NPU",), compile_error=False, infer_error=False):
        self.compiles = 0
        self.tensor = Array()
        class Compiled:
            outputs = ("output",)
            def __call__(inner, inputs):
                if infer_error:
                    raise RuntimeError("device lost during inference")
                return {"output": self.tensor}
        def compile_model(model, device, config):
            self.compiles += 1
            self.assertEqual(device, "NPU")
            if compile_error:
                raise RuntimeError("unsupported operator")
            return Compiled()
        return SimpleNamespace(available_devices=devices, read_model=lambda **kw: Model(), compile_model=compile_model)

    def cpu_factory(self):
        return SimpleNamespace(get_inputs=lambda: [SimpleNamespace(name="cpu_input")],
                               run=lambda outputs, feed: list(feed.values()))

    def session(self, core, strict=False):
        with patch.dict(sys.modules, {"openvino": SimpleNamespace(Core=lambda: core)}), patch.object(Path, "mkdir"):
            return NPUSession("test.onnx", b"model", self.cpu_factory, strict)

    def test_npu_dispatch_and_shape_reuse(self):
        session = self.session(self.fake_core(), strict=True)
        self.assertEqual(session.get_inputs()[0].name, "images")
        for _ in range(2):
            self.assertEqual(session.run(None, {"images": self.tensor}), [self.tensor])
        self.assertEqual(self.compiles, 1)
        self.assertEqual(session.status()["device"], "Intel NPU")

    def test_missing_device_auto_falls_back(self):
        session = self.session(self.fake_core(devices=("CPU",)))
        self.assertEqual(session.status()["device"], "CPU")
        self.assertIn("감지", session.status()["fallback_reason"])

    def test_missing_device_strict_stops(self):
        with self.assertRaisesRegex(RuntimeError, "NPU 실행 실패"):
            self.session(self.fake_core(devices=("CPU",)), strict=True)

    def test_compile_failure_auto_falls_back_and_remaps_input(self):
        session = self.session(self.fake_core(compile_error=True))
        self.assertEqual(session.run(None, {"images": self.tensor}), [self.tensor])
        self.assertEqual(session.status()["device"], "CPU")
        self.assertIn("unsupported operator", session.status()["fallback_reason"])
        self.assertEqual(session.get_inputs()[0].name, "cpu_input")

    def test_compile_failure_strict_stops(self):
        session = self.session(self.fake_core(compile_error=True), strict=True)
        with self.assertRaisesRegex(RuntimeError, "unsupported operator"):
            session.run(None, {"images": self.tensor})

    def test_inference_failure_auto_stays_on_cpu(self):
        session = self.session(self.fake_core(infer_error=True))
        for _ in range(2):
            self.assertEqual(session.run(None, {"images": self.tensor}), [self.tensor])
        self.assertEqual(self.compiles, 1)
        self.assertIn("device lost", session.status()["fallback_reason"])

    def test_missing_openvino_auto_and_strict(self):
        with patch.dict(sys.modules, {"openvino": None}):
            session = NPUSession("test.onnx", b"model", self.cpu_factory)
            self.assertEqual(session.status()["device"], "CPU")
            with self.assertRaisesRegex(RuntimeError, "NPU 실행 실패"):
                NPUSession("test.onnx", b"model", self.cpu_factory, strict=True)


if __name__ == "__main__":
    unittest.main()

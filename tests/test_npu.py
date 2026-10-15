"""Device dispatch tests runnable without video or NPU dependencies."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference import NPUSession, end2end_nms


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
    def fake_core(self, devices=("NPU",), compile_error=False, infer_error=False, names=None, fail_on=None):
        self.compiles = 0; self.devices_used = []
        self.tensor = Array()
        class Compiled:
            outputs = ("output",)
            def __call__(inner, inputs):
                if infer_error:
                    raise RuntimeError("device lost during inference")
                return {"output": self.tensor}
        def compile_model(model, device, config):
            self.compiles += 1; self.devices_used.append(device)
            if fail_on is None:
                self.assertEqual(device, "NPU")
            if compile_error or device == fail_on:
                raise RuntimeError("unsupported operator")
            return Compiled()
        return SimpleNamespace(available_devices=devices, read_model=lambda **kw: Model(), compile_model=compile_model,
                               get_property=lambda device, key: (names or {})[device])

    def cpu_factory(self):
        return SimpleNamespace(get_inputs=lambda: [SimpleNamespace(name="cpu_input")],
                               run=lambda outputs, feed: list(feed.values()))

    def session(self, core, strict=False, kind="NPU", fallback=None):
        with patch.dict(sys.modules, {"openvino": SimpleNamespace(Core=lambda: core)}), patch.object(Path, "mkdir"):
            return NPUSession("test.onnx", b"model", fallback or self.cpu_factory, strict, kind)

    def test_gpu_skips_non_intel_device(self):
        names = {"GPU.0": "NVIDIA GeForce RTX 5050 Laptop GPU (dGPU)", "GPU.1": "Intel(R) Arc(TM) 130T GPU (iGPU)"}
        session = self.session(self.fake_core(devices=("CPU", "GPU.0", "GPU.1"), names=names, fail_on="GPU.0"), strict=True, kind="GPU")
        self.assertEqual(session.run(None, {"images": self.tensor}), [self.tensor])
        self.assertEqual(self.devices_used, ["GPU.1"])
        self.assertEqual(session.status()["device"], "Intel GPU")

    def test_gpu_strict_without_intel_gpu_stops(self):
        core = self.fake_core(devices=("CPU", "GPU.0"), names={"GPU.0": "NVIDIA GeForce RTX"})
        with self.assertRaisesRegex(RuntimeError, "GPU 실행 실패"):
            self.session(core, strict=True, kind="GPU")

    def test_auto_npu_failure_moves_to_gpu_then_cpu(self):
        names = {"GPU.0": "Intel(R) Arc(TM) 130T GPU (iGPU)"}
        core = self.fake_core(devices=("CPU", "GPU.0", "NPU"), names=names, fail_on="NPU")
        with patch.dict(sys.modules, {"openvino": SimpleNamespace(Core=lambda: core)}), patch.object(Path, "mkdir"):
            gpu = lambda: NPUSession("test.onnx", b"model", self.cpu_factory, kind="GPU")
            session = NPUSession("test.onnx", b"model", gpu)
            self.assertEqual(session.run(None, {"images": self.tensor}), [self.tensor])
        self.assertEqual(self.devices_used, ["NPU", "GPU.0"])
        self.assertEqual(session.status()["device"], "Intel GPU")
        self.assertIn("NPU", session.status()["fallback_reason"])
        # 탐지 불일치 감시가 대체 장치(GPU)도 끌 수 있어야 한다.
        session._fallback(RuntimeError("mismatch"))
        self.assertEqual(session.status()["device"], "CPU")
        self.assertIn("mismatch", session.status()["fallback_reason"])

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

    def test_plate_nms_tail_matches_end2end_rows(self):
        import numpy as np
        raw = np.zeros((1, 8400, 5), np.float32)
        raw[0, :3] = [[100, 100, 40, 10, .9], [101, 100, 40, 10, .8], [300, 200, 40, 10, .5]]
        rows = end2end_nms(raw)
        # The overlapping lower score box is suppressed; zero score anchors are dropped.
        np.testing.assert_allclose(rows, [[0, 80, 95, 120, 105, 0, .9], [0, 280, 195, 320, 205, 0, .5]], rtol=1e-6)
        self.assertEqual(end2end_nms(np.zeros((1, 8400, 5), np.float32)).shape, (0, 7))


if __name__ == "__main__":
    unittest.main()

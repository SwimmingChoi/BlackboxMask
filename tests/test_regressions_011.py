import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import engine


class ModelPathTests(unittest.TestCase):
    def test_korean_model_path_uses_buffers_for_both_loaders(self):
        ort_factory=engine.ort.InferenceSession
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'사용자 이름'/'마스킹 프로그램'
            shutil.copytree(engine.ROOT/'models',root/'models')
            with patch.object(engine,'ROOT',root), \
                 patch.object(engine.ort,'InferenceSession',wraps=ort_factory) as session:
                d=engine.Detectors({'faces':True,'plates':True})
                self.assertEqual(session.call_count,2)
                for call in session.call_args_list:self.assertIsInstance(call[0][0],bytes)
                self.assertIsInstance(d.detect(np.zeros((96,160,3),np.uint8)),list)

    def test_missing_model_is_explained(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(engine,'ROOT',Path(directory)):
            with self.assertRaisesRegex(ValueError,'AI 모델 파일이 없습니다'):
                engine.model_bytes('head_yolox.onnx')

    def test_corrupt_model_is_explained(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'models').mkdir();(root/'models/head_yolox.onnx').write_bytes(b'truncated')
            with patch.object(engine,'ROOT',root),self.assertRaisesRegex(ValueError,'손상되었거나'):
                engine.model_bytes('head_yolox.onnx')

    def test_display_rotation_and_legacy_coordinates(self):
        pixels=np.arange(18,dtype=np.uint8).reshape(2,3,3)
        frame=SimpleNamespace(rotation=90,to_ndarray=lambda **kwargs:pixels)
        self.assertTrue(np.array_equal(engine.frame_image(frame),np.rot90(pixels)))
        self.assertTrue(np.array_equal(engine.frame_image(frame,0),pixels))
        frame.rotation=-90
        self.assertTrue(np.array_equal(engine.frame_image(frame),np.rot90(pixels,3)))


class ProgressTimeTests(unittest.TestCase):
    def test_hms_and_eta(self):
        self.assertEqual(engine.hms(5),'5초');self.assertEqual(engine.hms(125),'2분 5초');self.assertEqual(engine.hms(3660),'1시간 1분')
        started=engine.time.monotonic()-10  # 10초 동안 100개 중 25개 처리 → 남은 75개 ≈ 30초
        self.assertEqual(engine.eta_text(started,25,100),' · 남은 시간 약 30초')
        self.assertEqual(engine.eta_text(started,0,100),'');self.assertEqual(engine.eta_text(started,25,0),'')


if __name__=='__main__':unittest.main(verbosity=2)

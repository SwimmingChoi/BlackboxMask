import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from detection_checks import compare_boxes

def box(x,kind='face'):
    return {'kind':kind,'box':[x,0,x+20,20],'score':.9}

class ComparisonTests(unittest.TestCase):
    def test_order_and_small_coordinate_difference(self):
        self.assertEqual(compare_boxes([box(0),box(40)],[box(41),box(1)])['agreement'],1.)
    def test_wrong_regions_detected(self):
        result=compare_boxes([box(0),box(40)],[box(100),box(140)])
        self.assertEqual(result['agreement'],0.)
        self.assertEqual(result['missing_reference'],[0,1])
    def test_empty_and_missing(self):
        self.assertEqual(compare_boxes([],[])['agreement'],1.)
        self.assertEqual(compare_boxes([box(0)],[])['agreement'],0.)
    def test_one_box_cannot_match_multiple(self):
        self.assertEqual(compare_boxes([box(0),box(0)],[box(0)])['agreement'],.5)
    def test_different_classes_do_not_match(self):
        self.assertEqual(compare_boxes([box(0)],[box(0,'plate')])['agreement'],0.)

class GuardTests(unittest.TestCase):
    def setUp(self):
        # Exercise actual guard logic even when Linux video packages are absent.
        spec=importlib.util.spec_from_file_location('guard_engine',ROOT/'engine.py')
        self.engine=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,{name:SimpleNamespace() for name in ('av','cv2','numpy','onnxruntime')}):
            spec.loader.exec_module(self.engine)
        self.detector=object.__new__(self.engine.Detectors)
        self.detector.settings={'backend':'auto'}
        self.detector.audit_frames=0;self.detector.audit_log=[]
        self.detector.audit_reference=SimpleNamespace(_detect_impl=lambda image:[box(0)])
        self.detector._detect_impl=lambda image:[box(100)]
        def fallback(error):self.reason=str(error)
        self.detector.face=SimpleNamespace(status=lambda:{'device':'Intel NPU'},_fallback=fallback)
        self.detector.plate=None
    def test_auto_replaces_corrupt_current_frame(self):
        result=self.detector.detect(None)
        self.assertEqual(result,[box(0)])
        self.assertIn('불일치',self.reason)
        self.assertEqual(len(self.detector.audit_log),1)
    def test_strict_rejects_corrupt_result(self):
        def stop(error):raise RuntimeError(str(error))
        self.detector.face._fallback=stop
        with self.assertRaisesRegex(RuntimeError,'불일치'):self.detector.detect(None)
    def test_agreeing_npu_kept(self):
        self.detector._detect_impl=lambda image:[box(1)]
        self.assertEqual(self.detector.detect(None),[box(1)])
        self.assertEqual(self.detector.audit_log,[])

if __name__=='__main__':unittest.main()

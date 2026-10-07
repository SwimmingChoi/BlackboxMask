import copy
import sys
import tempfile
import unittest
from pathlib import Path
from fractions import Fraction

from types import SimpleNamespace

import av
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from engine import (analyze,export_video,sha256,apply_masks,manual_box,effective_boxes,
                    save_project,load_project,Tracker,Cancelled,Detectors,read_preview)


class FakeDetector:
    def __init__(self):self.i=0
    def detect(self,image):
        i=self.i;self.i+=1
        return [{"kind":"plate","box":[20+i,20,65+i,48],"score":.9}]


def sample_video(path,audio=False):
    times=[0,40,80,160,200,280,320,360,400,480]
    with av.open(str(path),'w') as c:
        v=c.add_stream('libx264',rate=25);v.width=160;v.height=96;v.pix_fmt='yuv420p'
        v.time_base=Fraction(1,1000);v.codec_context.time_base=Fraction(1,1000)
        v.codec_context.max_b_frames=0;v.options={'crf':'12'}
        a=c.add_stream('aac',rate=48000) if audio else None
        for i,t in enumerate(times):
            im=np.zeros((96,160,3),np.uint8);im[:]=(35,55,85)
            im[20:48,20+i:65+i]=np.random.default_rng(i).integers(0,256,(28,45,3),dtype=np.uint8)
            f=av.VideoFrame.from_ndarray(im,format='bgr24');f.pts=t;f.time_base=Fraction(1,1000)
            for p in v.encode(f):c.mux(p)
        for p in v.encode():c.mux(p)
        if a:
            f=av.AudioFrame.from_ndarray(np.zeros((1,24000),np.float32),format='flt',layout='mono')
            f.sample_rate=48000;f.pts=0;f.time_base=Fraction(1,48000)
            for p in a.encode(f):c.mux(p)
            for p in a.encode():c.mux(p)
    return times


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.source=self.root/'원본 영상.mp4';sample_video(self.source,audio=True)
        self.project=analyze(self.source,{'faces':False,'plates':False},detector=FakeDetector())
    def tearDown(self):self.temp.cleanup()

    def test_scan_and_tracking(self):
        self.assertEqual(len(self.project['frames']),10)
        self.assertEqual({b['id'] for f in self.project['frames'] for b in f['boxes']},{1})

    def test_short_gap_expires(self):
        t=Tracker();d=FakeDetector().detect(None)
        self.assertFalse(t.update(d,0)[0]['predicted'])
        self.assertTrue(t.update([],.1)[0]['predicted'])
        self.assertEqual(t.update([],.21),[])

    def test_manual_interpolation_and_bounds(self):
        item={'start':1,'end':3,'keys':[{'time':1,'box':[0,0,10,10]},{'time':3,'box':[20,0,30,10]}]}
        self.assertEqual(manual_box(item,2),[10,0,20,10]);self.assertIsNone(manual_box(item,.9))

    def test_mask_changes_only_region(self):
        im=np.random.default_rng(1).integers(0,256,(96,160,3),dtype=np.uint8)
        for style in ['blur','mosaic','solid']:
            out=apply_masks(im,[{'box':[20,20,60,50]}],style,0)
            self.assertTrue(np.array_equal(out[:20],im[:20]))
            self.assertTrue(np.array_equal(out[:,60:],im[:,60:]))
            self.assertFalse(np.array_equal(out[20:50,20:60],im[20:50,20:60]))

    def test_exclusion_and_manual(self):
        self.project['excluded']=[1]
        self.assertEqual(effective_boxes(self.project,0),[])
        self.project['manual']=[{'id':-1,'start':0,'end':1,'keys':[{'time':0,'box':[0,0,10,10]}]}]
        self.assertEqual(effective_boxes(self.project,0)[0]['id'],-1)

    def test_export_vfr_and_audio_and_original(self):
        original=sha256(self.source)
        for audio in [False,True]:
            target=self.root/f'출력_{audio}.mp4'
            export_video(self.project,target,{'style':'solid','watermark':False,'audio':audio})
            with av.open(str(target)) as c:
                self.assertEqual(bool(c.streams.audio),audio)
                output_duration=float(c.streams.video[0].duration*c.streams.video[0].time_base)
                frames=list(c.decode(video=0))
            self.assertEqual(len(frames),10)
            times=[float(f.pts*f.time_base) for f in frames]
            expected=[f['time'] for f in self.project['frames']]
            for a,b in zip(times,expected):self.assertAlmostEqual(a-times[0],b-expected[0],places=5)
            self.assertAlmostEqual(output_duration,self.project['meta']['duration'],places=4)
            self.assertEqual(frames[0].width,160);self.assertEqual(frames[0].height,96)
            roi=frames[0].to_ndarray(format='bgr24')[24:40,26:56]
            self.assertLess(roi.std(),4)
        self.assertEqual(sha256(self.source),original)

    def test_cancel_does_not_replace_file(self):
        out=self.root/'existing.mp4';out.write_bytes(b'previous')
        counter=[False]
        def progress(value,text):
            if value>0:counter[0]=True
        with self.assertRaises(Cancelled):export_video(self.project,out,{},progress,lambda:counter[0])
        self.assertEqual(out.read_bytes(),b'previous');self.assertFalse(list(self.root.glob('.masking-*')))

    def test_protect_original(self):
        with self.assertRaises(ValueError):export_video(self.project,self.source,{})

    def test_project_reload_and_source_change(self):
        p=self.root/'project.json';save_project(self.project,p)
        loaded=load_project(p);self.assertEqual(loaded['sha256'],self.project['sha256'])
        with open(self.source,'ab') as f:f.write(b'changed')
        with self.assertRaises(ValueError):load_project(p)
        with self.assertRaises(ValueError):export_video(self.project,self.root/'out.mp4',{})

    def test_preview_matches_time(self):
        with av.open(str(self.source)) as c:frames=[f.to_ndarray(format='bgr24') for f in c.decode(video=0)]
        for i in [0,3,9]:self.assertTrue(np.array_equal(read_preview(self.project,i),frames[i]))

    def test_head_decode_scale_class_and_nms(self):
        # 2560x1280 원본 → 1280x640 입력. stride 16 격자의 (열 10, 행 5) 칸에 머리 하나
        rows=np.zeros((16800,12),np.float32);at=12800+5*80+10
        rows[at]=[.5,.5,np.log(2),np.log(3),1,0,.9,0,0,0,0,0]
        rows[at+1]=[-.5,.5,np.log(2),np.log(3),1,0,.6,0,0,0,0,0]  # 같은 머리의 중복 후보 → NMS로 제거
        rows[100]=[.5,.5,0,0,1,0,.1,.9,0,0,0,0]  # face 클래스만 높은 칸 → head 점수 미달
        d=Detectors({'faces':False,'plates':False})
        d.face=SimpleNamespace(get_inputs=lambda:[SimpleNamespace(name='images')],run=lambda _,feed:[rows[None]])
        found=d._face(np.zeros((1280,2560,3),np.uint8),5,7)
        self.assertEqual([f['box'] for f in found],[[304+5,128+7,368+5,224+7]])
        self.assertAlmostEqual(found[0]['score'],.9,places=5)

    def test_real_models_execute(self):
        d=Detectors({'faces':True,'plates':True})
        results=d.detect(np.zeros((96,160,3),np.uint8))
        self.assertIsInstance(results,list)


if __name__=='__main__':unittest.main(verbosity=2)

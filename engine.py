"""Offline video masking engine. No uploads or network calls."""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent
VERSION = "0.1.2"
MODEL_HASHES = {
    "head_yolox.onnx": "2680679d07e432cc705a147b3990bd4c6ea240317805043646eabe44b839d8d4",
    "plate_yolov9.onnx": "c3c1026ca7d0585dd88084d68182dd897113712fa734ae1557ca70174440c076",
}


def model_bytes(name):
    """Use Python's Unicode file API; never pass model paths into C++ loaders."""
    path = ROOT / "models" / name
    try:
        data = path.read_bytes()
    except FileNotFoundError as e:
        raise ValueError(f"AI 모델 파일이 없습니다: {name}\n프로그램의 models 폴더에 수정 패키지의 모델 파일을 복사해 주세요.") from e
    except OSError as e:
        raise ValueError(f"AI 모델 파일에 접근할 수 없습니다: {name}\n파일 접근 권한과 압축 해제 상태를 확인해 주세요.") from e
    if hashlib.sha256(data).hexdigest() != MODEL_HASHES[name]:
        raise ValueError(f"AI 모델 파일이 손상되었거나 다른 버전입니다: {name}\n수정 패키지의 models 폴더를 다시 복사해 주세요.")
    return data


def frame_rotation(frame):
    rotation = float(getattr(frame, "rotation", 0) or 0)
    nearest = round(rotation / 90) * 90
    if abs(rotation-nearest) > .1:
        raise ValueError("90도 단위가 아닌 회전 영상은 현재 지원하지 않습니다.")
    return int(nearest) % 360


def frame_image(frame, rotation=None):
    """Apply display rotation before detection, preview, and exported masking."""
    image = frame.to_ndarray(format="bgr24")
    degrees = frame_rotation(frame) if rotation is None else rotation
    return np.ascontiguousarray(np.rot90(image, int(degrees)//90)) if degrees else image


def hms(seconds):
    """초를 '1시간 2분' / '3분 4초' / '5초' 형식으로."""
    m,s = divmod(int(round(seconds)),60); h,m = divmod(m,60)
    return f"{h}시간 {m}분" if h else (f"{m}분 {s}초" if m else f"{s}초")


def eta_text(started, done, total):
    """지금까지의 평균 속도로 남은 시간을 추정. 전체 수를 모르면 빈 문자열."""
    if done <= 0 or total <= done:
        return ""
    return f" · 남은 시간 약 {hms((time.monotonic()-started)/done*(total-done))}"


class Cancelled(Exception):
    pass


def check_cancel(cancel):
    if cancel and cancel():
        raise Cancelled("작업을 취소했습니다. 원본 파일은 변경하지 않았습니다.")


def sha256(path, cancel=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(4 * 1024 * 1024):
            check_cancel(cancel)
            h.update(chunk)
    return h.hexdigest()


def clip_box(box, width, height, margin=0):
    x1, y1, x2, y2 = map(float, box)
    dx, dy = (x2-x1)*margin, (y2-y1)*margin
    return [max(0, int(math.floor(x1-dx))), max(0, int(math.floor(y1-dy))),
            min(width, int(math.ceil(x2+dx))), min(height, int(math.ceil(y2+dy)))]


def iou(a, b):
    inter = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    return inter / max(1, (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1])-inter)


def model_session(name, backend="cpu"):
    if backend not in ("cpu", "auto", "npu"):
        raise ValueError(f"지원하지 않는 추론 장치: {backend}")
    data = model_bytes(name)
    if backend != "cpu":
        from inference import NPUSession
        return NPUSession(name, data, lambda: model_session(name, "cpu"), strict=backend == "npu")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = max(1, min(4, os.cpu_count() or 2))
    # 두 모델이 프레임마다 번갈아 실행되므로, 쉬는 모델의 스레드가 공회전하며 CPU를 잡지 않게 한다.
    opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return ort.InferenceSession(data, sess_options=opts, providers=["CPUExecutionProvider"])


class Detectors:
    def __init__(self, settings):
        self.settings = settings
        cv2.setNumThreads(max(1, min(4, os.cpu_count() or 2)))
        backend = settings.get("backend", "cpu")
        self.face = model_session("head_yolox.onnx", backend) if settings.get("faces", True) else None
        self.plate = model_session("plate_yolov9.onnx", backend) if settings.get("plates", True) else None
        self.grids = {}
        self.audit_reference = None
        self.audit_frames = 0
        self.audit_log = []

    def status(self):
        return {kind: session.status() if hasattr(session, "status") else {"device":"CPU", "fallback_reason":""}
                for kind, session in (("face", self.face), ("plate", self.plate)) if session is not None}

    def _plate(self, image, ox=0, oy=0):
        h, w = image.shape[:2]
        ratio = min(640/w, 640/h)
        nw, nh = round(w*ratio), round(h*ratio)
        dw, dh = (640-nw)/2, (640-nh)/2
        padded = cv2.copyMakeBorder(cv2.resize(image, (nw, nh)), round(dh-.1), round(dh+.1),
                                    round(dw-.1), round(dw+.1), cv2.BORDER_CONSTANT, value=(114,114,114))
        tensor = np.ascontiguousarray(padded[:, :, ::-1].transpose(2,0,1)[None], dtype=np.float32)/255
        rows = self.plate.run(None, {self.plate.get_inputs()[0].name: tensor})[0]
        result = []
        for row in rows:
            if float(row[6]) < self.settings.get("plate_threshold", .25):
                continue
            b = [(float(row[1])-dw)/ratio, (float(row[2])-dh)/ratio,
                 (float(row[3])-dw)/ratio, (float(row[4])-dh)/ratio]
            b = clip_box(b, w, h)
            if b[2] > b[0] and b[3] > b[1]:
                result.append({"kind":"plate", "box":[b[0]+ox,b[1]+oy,b[2]+ox,b[3]+oy], "score":float(row[6])})
        return result

    def _grid(self, height, width):
        """YOLOX 출력 행 순서(stride 8→16→32, 행 우선)에 맞춘 [열, 행, stride]."""
        if (height, width) not in self.grids:
            cells = []
            for stride in (8, 16, 32):
                ys, xs = np.mgrid[:height//stride, :width//stride]
                cells.append(np.stack([xs.ravel(), ys.ravel(), np.full(xs.size, stride)], 1))
            self.grids[(height, width)] = np.concatenate(cells).astype(np.float32)
        return self.grids[(height, width)]

    def _face(self, image, ox=0, oy=0):
        """얼굴이 옆·뒤를 향해도 놓치지 않도록 사람 머리(head) 전체를 얼굴 영역으로 탐지한다."""
        h, w = image.shape[:2]
        scale = min(1., 1280/max(h,w))
        small = cv2.resize(image, (max(32,round(w*scale)), max(32,round(h*scale))))
        sx, sy = small.shape[1]/w, small.shape[0]/h
        padded = cv2.copyMakeBorder(small, 0, -small.shape[0]%32, 0, -small.shape[1]%32,
                                    cv2.BORDER_CONSTANT, value=(114,114,114))
        tensor = np.ascontiguousarray(padded.transpose(2,0,1)[None], dtype=np.float32)
        rows = self.face.run(None, {self.face.get_inputs()[0].name: tensor})[0][0]
        score = rows[:,4]*rows[:,6]  # 물체 점수 × head 클래스 점수 (클래스 순서: body, head, face, ...)
        keep = score >= self.settings.get("face_threshold", .3)
        rows, score, grid = rows[keep], score[keep], self._grid(*padded.shape[:2])[keep]
        cx, cy = (rows[:,0]+grid[:,0])*grid[:,2], (rows[:,1]+grid[:,1])*grid[:,2]
        bw, bh = np.exp(rows[:,2])*grid[:,2], np.exp(rows[:,3])*grid[:,2]
        boxes = np.stack([(cx-bw/2)/sx, (cy-bh/2)/sy, bw/sx, bh/sy], 1)
        result = []
        for i in np.array(cv2.dnn.NMSBoxes(boxes.tolist(), score.tolist(), 0., .4), dtype=int).ravel():
            x,y,bw,bh = map(float,boxes[i])
            b = clip_box([x,y,x+bw,y+bh],w,h)
            if b[2] > b[0] and b[3] > b[1]:
                result.append({"kind":"face", "box":[b[0]+ox,b[1]+oy,b[2]+ox,b[3]+oy], "score":float(score[i])})
        return result

    def _tiled(self, fn, image):
        # ponytail: 겹치는 4분할 탐지 - 정확한 격자 최적화 필요해지면 그때 보완
        h, w = image.shape[:2]
        found = fn(image)
        if self.settings.get("detail", False) and max(w,h)>800:
            tw, th = round(w*.6), round(h*.6)
            for y in (0,h-th):
                for x in (0,w-tw):
                    found = found + fn(image[y:y+th,x:x+tw],x,y)
        return found

    def detect(self, image):
        found = self._detect_impl(image)
        self.audit_frames += 1
        npu_active=any(hasattr(s,'status') and s.status()['device']=='Intel NPU' for s in (self.face,self.plate) if s is not None)
        if npu_active and self.settings.get('verify_npu',True):
            if self.audit_frames <= 3 or self.audit_frames % 60 == 0:
                if self.audit_reference is None:
                    self.audit_reference = Detectors(dict(self.settings,backend='cpu',verify_npu=False))
                reference = self.audit_reference._detect_impl(image)
                from detection_checks import compare_boxes
                for kind,attr in (('face','face'),('plate','plate')):
                    session=getattr(self,attr)
                    if session is None or not hasattr(session,'status') or session.status()['device']!='Intel NPU':continue
                    cpu_boxes=[x for x in reference if x['kind']==kind]
                    comparison=compare_boxes(cpu_boxes,[x for x in found if x['kind']==kind])
                    if comparison['agreement']<.5:
                        reason=f"NPU/CPU 탐지 불일치 ({kind}, 프레임 {self.audit_frames}): CPU {comparison['reference_count']}개 / NPU {comparison['candidate_count']}개 / 일치 {comparison['agreement']:.0%}"
                        self.audit_log.append(dict(comparison,kind=kind,frame=self.audit_frames))
                        session._fallback(RuntimeError(reason))
                        found=[x for x in found if x['kind']!=kind]+cpu_boxes
        return found

    def _detect_impl(self, image):
        found = []
        if self.face is not None:
            found += self._tiled(self._face, image)
        if self.plate is not None:
            found += self._tiled(self._plate, image)
        keep = []
        for d in sorted(found,key=lambda x:x["score"],reverse=True):
            if not any(x["kind"]==d["kind"] and iou(x["box"],d["box"])>.4 for x in keep):
                keep.append(d)
        return keep


class Tracker:
    """Conservative short-gap IoU association; identity is not guaranteed."""
    def __init__(self):
        self.active = {}
        self.next_id = 1

    def update(self, detections, timestamp):
        candidates = []
        for tid,t in self.active.items():
            if timestamp-t["seen"] > .20:
                continue
            for j,d in enumerate(detections):
                if d["kind"] == t["kind"]:
                    score = iou(t["box"],d["box"])
                    if score > .10:
                        candidates.append((score,tid,j))
        matched_ids, matched_d = set(),set()
        result = []
        for _,tid,j in sorted(candidates,reverse=True):
            if tid in matched_ids or j in matched_d:
                continue
            matched_ids.add(tid); matched_d.add(j)
            d = dict(detections[j],id=tid,predicted=False)
            self.active[tid] = dict(d,seen=timestamp)
            result.append(d)
        for j,d in enumerate(detections):
            if j not in matched_d:
                tid = self.next_id; self.next_id += 1
                d = dict(d,id=tid,predicted=False)
                self.active[tid] = dict(d,seen=timestamp)
                matched_ids.add(tid)
                result.append(d)
        for tid,t in list(self.active.items()):
            if timestamp-t["seen"] > .20:
                del self.active[tid]
            elif tid not in matched_ids:
                result.append({k:v for k,v in dict(t,predicted=True).items() if k!="seen"})
        return result


def analyze(path, settings, progress=lambda *a:None, cancel=None, detector=None):
    path = str(Path(path).resolve())
    progress(0,"AI 모델을 준비하는 중")
    detector = detector or Detectors(settings)
    tracker = Tracker()
    frames = []
    with av.open(path) as src:
        if not src.streams.video:
            raise ValueError("영상 스트림이 없습니다.")
        stream = src.streams.video[0]
        rate = stream.average_rate or Fraction(30,1)
        meta = {"width":stream.width,"height":stream.height,"source_width":stream.width,
                "source_height":stream.height,"rate":str(rate),
                "time_base":str(stream.time_base),"has_audio":bool(src.streams.audio)}
        estimated = stream.frames or int((src.duration or 0)/av.time_base*float(rate))
        previous_time = None
        started = time.monotonic()
        for n,frame in enumerate(src.decode(stream)):
            check_cancel(cancel)
            if frame.pts is None:
                raise ValueError("프레임 시각이 없는 영상입니다. 원본 재생기로 표준 MP4로 내보내 주세요.")
            t = float(frame.pts*frame.time_base)
            if previous_time is not None and t <= previous_time:
                raise ValueError("프레임 시각이 중복되거나 역전된 영상입니다. 표준 MP4로 내보내 주세요.")
            previous_time = t
            rotation = frame_rotation(frame)
            if n == 0:
                meta["rotation"] = rotation
                if rotation in (90,270):
                    meta["width"],meta["height"] = stream.height,stream.width
            if rotation != meta["rotation"]:
                raise ValueError("중간에 회전 정보가 바뀌는 영상은 현재 지원하지 않습니다.")
            im = frame_image(frame, rotation)
            if im.shape[:2] != (meta["height"],meta["width"]):
                raise ValueError("중간에 해상도가 바뀌는 영상은 현재 지원하지 않습니다.")
            boxes = tracker.update(detector.detect(im),t)
            frames.append({"pts":frame.pts,"tb":str(frame.time_base),"time":t,"duration":frame.duration,"boxes":boxes})
            progress(min(98,int((n+1)/max(estimated,n+2)*98)),
                     f"분석 중 · {n+1:,}/{max(estimated,n+1):,} 프레임 · {len(boxes)}개 영역{eta_text(started,n+1,estimated)}")
    if not frames:
        raise ValueError("영상에서 읽을 수 있는 프레임이 없습니다.")
    meta["origin"] = frames[0]["time"]
    tail = float(frames[-1]["duration"]*Fraction(frames[-1]["tb"])) if frames[-1]["duration"] else 1/float(rate)
    meta["duration"] = frames[-1]["time"]-meta["origin"]+tail
    progress(99,"원본 파일을 확인하는 중")
    digest = sha256(path,cancel)
    return {"schema":1,"version":VERSION,"source":path,"sha256":digest,"meta":meta,
            "settings":settings,"frames":frames,"manual":[],"excluded":[],
            "inference":detector.status() if hasattr(detector,"status") else {},
            "inference_checks":getattr(detector,'audit_log',[])}


def manual_box(item, t):
    if not item["start"] <= t <= item["end"]:
        return None
    keys = sorted(item["keys"],key=lambda k:k["time"])
    if not keys:
        return None
    if t <= keys[0]["time"]:
        return keys[0]["box"]
    for a,b in zip(keys,keys[1:]):
        if a["time"] <= t <= b["time"]:
            f = (t-a["time"])/max(1e-9,b["time"]-a["time"])
            return [x+(y-x)*f for x,y in zip(a["box"],b["box"])]
    return keys[-1]["box"]


def effective_boxes(project,index):
    f = project["frames"][index]
    result = [b for b in f["boxes"] if b["id"] not in project["excluded"]]
    t = f["time"]-project["meta"]["origin"]
    for item in project["manual"]:
        box = manual_box(item,t)
        if box is not None:
            result.append({"kind":"manual","box":box,"id":item["id"],"score":1.,"predicted":False})
    return result


def apply_masks(image, boxes, style="blur", margin=.18):
    out = image.copy()
    h,w = image.shape[:2]
    for item in boxes:
        x1,y1,x2,y2 = clip_box(item["box"],w,h,margin)
        if x2<=x1 or y2<=y1:
            continue
        roi = out[y1:y2,x1:x2]
        if style=="solid":
            roi[:] = (24,24,24)
        elif style=="mosaic":
            tiny = cv2.resize(roi,(max(1,(x2-x1)//24),max(1,(y2-y1)//24)),interpolation=cv2.INTER_AREA)
            roi[:] = cv2.resize(tiny,(x2-x1,y2-y1),interpolation=cv2.INTER_NEAREST)
        else:
            # Scale blur with region size; strongest privacy uses solid fill.
            sigma = max(9., min(x2-x1,y2-y1)*.35)
            roi[:] = cv2.GaussianBlur(roi,(0,0),sigmaX=sigma,sigmaY=sigma)
    return out


def read_preview(project,index):
    target = project["frames"][index]
    with av.open(project["source"]) as src:
        stream = src.streams.video[0]
        target_time = Fraction(target["pts"])*Fraction(target["tb"])
        src.seek(int(target_time/stream.time_base),stream=stream,backward=True)
        for frame in src.decode(stream):
            if frame.pts is not None and frame.pts*frame.time_base >= target_time:
                return frame_image(frame, project["meta"].get("rotation", 0))
    raise ValueError("미리보기 프레임을 읽지 못했습니다.")


def save_project(project,path):
    path = Path(path)
    fd,tmp = tempfile.mkstemp(dir=path.parent,suffix=".tmp")
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as f:
            json.dump(project,f,ensure_ascii=False,separators=(",",":"))
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def load_project(path, source=None):
    p = json.loads(Path(path).read_text(encoding="utf-8"))
    if p.get("schema")!=1 or not p.get("frames"):
        raise ValueError("지원하지 않는 프로젝트 파일입니다.")
    if source: p["source"]=str(Path(source).resolve())
    if not Path(p["source"]).is_file():
        raise ValueError("원본 영상 경로가 바뀌었습니다. 원본 영상을 원래 위치에 두세요.")
    if sha256(p["source"])!=p["sha256"]:
        raise ValueError("프로젝트와 원본 영상이 일치하지 않습니다. 다시 분석해 주세요.")
    return p


def export_video(project,destination,options,progress=lambda *a:None,cancel=None):
    destination = Path(destination).resolve()
    if destination == Path(project["source"]).resolve():
        raise ValueError("원본 파일에는 덮어쓸 수 없습니다.")
    if destination.exists() and sha256(destination) == project["sha256"]:
        raise ValueError("원본 파일과 동일한 내용의 경로에는 저장할 수 없습니다.")
    progress(0,"원본 파일을 확인하는 중")
    if sha256(project["source"],cancel)!=project["sha256"]:
        raise ValueError("분석 이후 원본 영상이 변경되었습니다. 다시 분석해 주세요.")
    fd,tmp = tempfile.mkstemp(prefix=".masking-",suffix=".mp4",dir=destination.parent)
    os.close(fd)
    count=0
    try:
        with av.open(project["source"]) as src, av.open(tmp,"w",format="mp4",options={"movflags":"+faststart"}) as dst:
            vs=src.streams.video[0]
            # H.264 with frame timestamps, not a constant-rate OpenCV VideoWriter.
            out=dst.add_stream("libx264",rate=Fraction(project["meta"]["rate"]))
            out.width,out.height=project["meta"]["width"],project["meta"]["height"]
            if out.width%2 or out.height%2:
                raise ValueError("현재 MP4 출력은 가로·세로가 짝수인 영상을 지원합니다.")
            out.pix_fmt="yuv420p"
            out.time_base=Fraction(project["meta"]["time_base"])
            out.codec_context.time_base=out.time_base
            out.codec_context.max_b_frames=0
            out.options={"crf":"18","preset":"fast"}
            durations={Fraction(f["pts"])*Fraction(f["tb"]):Fraction(f.get("duration",0))*Fraction(f["tb"])
                       for f in project["frames"]}
            def mux_video(packet):
                if packet.pts is not None and packet.time_base:
                    duration=durations.get(Fraction(packet.pts)*packet.time_base)
                    if duration:
                        packet.duration=max(1,round(duration/packet.time_base))
                dst.mux(packet)
            audio=None; audio_out=None
            if options.get("audio") and src.streams.audio:
                audio=src.streams.audio[0]
                audio_out=dst.add_stream_from_template(audio)
                audio_out.metadata.clear()
            streams=[vs]+([audio] if audio else [])
            started=time.monotonic()
            for packet in src.demux(streams):
                check_cancel(cancel)
                if audio is not None and packet.stream.index==audio.index:
                    if packet.dts is not None:
                        packet.stream=audio_out
                        dst.mux(packet)
                    continue
                for frame in packet.decode():
                    check_cancel(cancel)
                    if count>=len(project["frames"]):
                        raise ValueError("프레임 수가 분석 결과와 다릅니다.")
                    expected=project["frames"][count]
                    if frame.pts is None or Fraction(frame.pts)*frame.time_base != Fraction(expected["pts"])*Fraction(expected["tb"]):
                        raise ValueError("프레임 시각이 분석 결과와 다릅니다.")
                    masked=apply_masks(frame_image(frame,project["meta"].get("rotation",0)),effective_boxes(project,count),
                                       options.get("style","blur"),options.get("margin",.18))
                    if options.get("watermark",True):
                        scale=max(.45,min(out.width,out.height)/1100)
                        cv2.putText(masked,"MASKED COPY",(12, max(20,out.height-14)),cv2.FONT_HERSHEY_SIMPLEX,
                                    scale,(0,0,0),max(3,round(scale*5)),cv2.LINE_AA)
                        cv2.putText(masked,"MASKED COPY",(12,max(20,out.height-14)),cv2.FONT_HERSHEY_SIMPLEX,
                                    scale,(255,255,255),max(1,round(scale*2)),cv2.LINE_AA)
                    encoded=av.VideoFrame.from_ndarray(masked,format="bgr24")
                    encoded.pts=frame.pts; encoded.time_base=frame.time_base
                    encoded.duration=frame.duration
                    for op in out.encode(encoded): mux_video(op)
                    count+=1
                    progress(int(count/len(project["frames"])*98),
                             f"저장 중 · {count:,}/{len(project['frames']):,} 프레임{eta_text(started,count,len(project['frames']))}")
            if count!=len(project["frames"]):
                raise ValueError("영상이 분석 결과보다 일찍 끝났습니다. 출력 파일을 저장하지 않았습니다.")
            for op in out.encode(): mux_video(op)
        check_cancel(cancel)
        os.replace(tmp,destination)
        report={"app":VERSION,"source_sha256":project["sha256"],"output_sha256":sha256(destination),
                "frames":count,"options":options,"inference":project.get("inference",{}),"inference_checks":project.get("inference_checks",[]),"excluded_track_ids":project["excluded"],
                "manual_regions":len(project["manual"]),"reviewed":bool(options.get("reviewed")),
                "note":"자동 탐지 결과는 무누락을 보장하지 않습니다. 음성·화면 내 문자 검토는 별도입니다."}
        log=destination.with_suffix(".mask-report.json")
        try:
            log.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
        except OSError:
            return str(destination)+"\n(영상 저장 완료, 처리 기록 파일 저장 실패)"
        return str(destination)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

from __future__ import annotations

import copy
import json
import sys
import threading
import time
import traceback
from pathlib import Path

import av
import cv2
from PySide6.QtCore import Qt, Signal, QThread, QTimer, QRectF, QPointF
from PySide6.QtGui import QImage, QPainter, QColor, QPen, QFont, QPalette
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QMessageBox, QListWidget, QListWidgetItem,
    QCheckBox, QComboBox, QDoubleSpinBox, QSlider, QProgressBar, QGroupBox, QFormLayout,
    QSplitter, QAbstractItemView, QScrollArea, QStyle)

from engine import (VERSION, analyze, apply_masks, effective_boxes, read_preview, frame_image,
                    save_project, load_project, export_video, Cancelled, hms)

STYLE = """
/* 한화 디자인 토큰(hw-translator.html) 기반 — navy #1A2B4A / orange #F37321 / 라이트 서피스 */
QMainWindow, QWidget {background:#F7F9FC;color:#1A2B4A;font-size:13px;}
QLabel, QCheckBox {background:transparent;}
QWidget#nav {background:#1A2B4A;}
QLabel#mark {background:#F37321;color:#FFFFFF;font-weight:800;font-size:15px;border-radius:8px;min-width:32px;max-width:32px;min-height:32px;max-height:32px;qproperty-alignment:AlignCenter;}
QLabel#brand {color:#FFFFFF;font-size:18px;font-weight:700;}
QLabel#navmuted {color:#CBD5E0;}
QLabel#muted {color:#64748B;}
QLabel#badge {background:#FFF3EB;color:#C75E14;border-radius:13px;padding:5px 12px;font-size:11px;font-weight:700;}
QPushButton {background:#FFFFFF;color:#1A2B4A;border:1px solid #E2E8F0;border-radius:12px;padding:9px 14px;}
QPushButton:hover {background:#F7F9FC;border-color:#CBD5E0;}
QPushButton:pressed {background:#EEF2F7;}
QPushButton:disabled {color:#94A3B8;background:#F7F9FC;border-color:#EEF2F7;}
QPushButton#primary {background:qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 #F37321,stop:1 #E06A1B);color:#FFFFFF;border:0;font-weight:700;padding:10px 18px;border-radius:14px;}
QPushButton#primary:hover {background:#E06A1B;}
QPushButton#primary:pressed {background:#C75E14;}
QPushButton#primary:disabled {background:#FDEEDE;color:#94A3B8;}
QListWidget {background:#FFFFFF;border:1px solid #E2E8F0;border-radius:12px;padding:4px;}
QListWidget::item {padding:7px;border-radius:6px;}
QListWidget::item:hover {background:#F7F9FC;}
QListWidget::item:selected {background:#FFF3EB;color:#C75E14;}
QGroupBox {background:#FFFFFF;border:1px solid #E2E8F0;border-radius:16px;margin-top:14px;padding:18px 12px 10px;}
QGroupBox::title {subcontrol-origin:margin;left:14px;padding:0 4px;color:#1A2B4A;font-weight:700;}
QComboBox,QDoubleSpinBox {background:#FFFFFF;border:1px solid #E2E8F0;border-radius:8px;padding:5px 8px;}
QComboBox:hover,QDoubleSpinBox:hover {border-color:#CBD5E0;}
QComboBox:focus,QDoubleSpinBox:focus {border-color:#F37321;}
QComboBox QAbstractItemView {background:#FFFFFF;border:1px solid #E2E8F0;selection-background-color:#FFF3EB;selection-color:#C75E14;}
QCheckBox {spacing:7px;padding:3px;}
QCheckBox::indicator {width:16px;height:16px;border:1px solid #CBD5E0;border-radius:4px;background:#FFFFFF;}
QCheckBox::indicator:hover {border-color:#F37321;}
QCheckBox::indicator:checked {background:#F37321;border-color:#F37321;}
QCheckBox::indicator:disabled {background:#F7F9FC;border-color:#E2E8F0;}
QCheckBox::indicator:checked:disabled {background:#FDEEDE;border-color:#FDEEDE;}
QProgressBar {background:#EEF2F7;border:0;border-radius:5px;height:10px;text-align:center;color:#1A2B4A;}
QProgressBar::chunk {background:#F37321;border-radius:5px;}
QSlider::groove:horizontal {height:6px;background:#E2E8F0;border-radius:3px;}
QSlider::sub-page:horizontal {background:#F37321;border-radius:3px;}
QSlider::handle:horizontal {background:#FFFFFF;border:2px solid #F37321;width:14px;margin:-6px 0;border-radius:8px;}
QSplitter::handle {background:#E2E8F0;}
QScrollArea {background:transparent;}
QScrollBar:vertical {background:transparent;width:8px;margin:0;}
QScrollBar::handle:vertical {background:#CBD5E0;border-radius:4px;min-height:24px;}
QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical {height:0;}
QToolTip {background:#1A2B4A;color:#FFFFFF;border:0;padding:6px 8px;}
"""


class Canvas(QWidget):
    drawn = Signal(list)
    def __init__(self, editable=False):
        super().__init__()
        self.setMinimumSize(280,240)
        self.image=None; self.boxes=[]; self.rect=QRectF(); self.start=None; self.end=None
        self.editable=editable
        self.setMouseTracking(True)

    def set_image(self,image,boxes=None):
        if image is None: self.image=None
        else:
            rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            self.image=QImage(rgb.data,rgb.shape[1],rgb.shape[0],rgb.strides[0],QImage.Format_RGB888).copy()
        self.boxes=boxes or []; self.update()

    def paintEvent(self,event):
        p=QPainter(self); p.fillRect(self.rect_adjusted(),QColor("#1A2B4A"))
        if self.image is None:
            p.setPen(QColor("#94A3B8"));p.drawText(self.rect_adjusted(),Qt.AlignCenter,"영상을 추가하면 미리보기가 표시됩니다")
            return
        scale=min(self.width()/self.image.width(),self.height()/self.image.height())
        w,h=self.image.width()*scale,self.image.height()*scale
        self.rect=QRectF((self.width()-w)/2,(self.height()-h)/2,w,h)
        p.drawImage(self.rect,self.image)
        for b in self.boxes:
            x1,y1,x2,y2=b["box"]
            color="#FBBF24" if b.get("predicted") else ("#60A5FA" if b["kind"]=="manual" else "#F37321")
            p.setPen(QPen(QColor(color),2))
            r=QRectF(self.rect.x()+x1*scale,self.rect.y()+y1*scale,(x2-x1)*scale,(y2-y1)*scale)
            p.drawRect(r);p.drawText(r.topLeft()+QPointF(3,15),f"#{b['id']}")
        if self.start is not None and self.end is not None:
            p.setPen(QPen(QColor("#93C5FD"),2,Qt.DashLine))
            p.drawRect(QRectF(self.start,self.end).normalized())

    def rect_adjusted(self):
        return QRectF(0,0,self.width(),self.height())

    def mousePressEvent(self,event):
        if self.editable and self.isEnabled() and self.image is not None and event.button()==Qt.LeftButton and self.rect.contains(event.position()):
            self.start=event.position();self.end=self.start

    def mouseMoveEvent(self,event):
        if self.start is not None:
            self.end=QPointF(max(self.rect.left(),min(self.rect.right(),event.position().x())),
                            max(self.rect.top(),min(self.rect.bottom(),event.position().y())))
            self.update()

    def mouseReleaseEvent(self,event):
        if self.start is None:return
        r=QRectF(self.start,self.end).normalized().intersected(self.rect)
        if r.width()>4 and r.height()>4:
            sx=self.image.width()/self.rect.width();sy=self.image.height()/self.rect.height()
            self.drawn.emit([(r.left()-self.rect.left())*sx,(r.top()-self.rect.top())*sy,
                             (r.right()-self.rect.left())*sx,(r.bottom()-self.rect.top())*sy])
        self.start=None;self.end=None;self.update()


class SeekSlider(QSlider):
    """클릭한 위치로 바로 이동하는 탐색 슬라이더. 기본 QSlider는 홈 클릭 시 페이지 단위로만 움직인다."""
    def mousePressEvent(self,event):
        if event.button()==Qt.LeftButton and self.maximum()>self.minimum():
            self.setValue(QStyle.sliderValueFromPosition(self.minimum(),self.maximum(),int(event.position().x()),self.width()))
        super().mousePressEvent(event)


class Worker(QThread):
    progress=Signal(int,str)
    result=Signal(object)
    failed=Signal(str)
    def __init__(self,fn):
        super().__init__();self.fn=fn;self.cancel=threading.Event()
    def run(self):
        try:self.result.emit(self.fn(self.progress.emit,self.cancel.is_set))
        except Cancelled as e:self.failed.emit(str(e))
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(str(e))


def button(text,callback,primary=False):
    b=QPushButton(text);b.clicked.connect(callback)
    if primary:b.setObjectName("primary")
    return b


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Blackbox Mask · {VERSION} 미리보기")
        self.resize(1460,900)
        self.project=None;self.raw=None;self.index=0;self.worker=None;self.busy=False;self.dirty=False
        self.queue=[];self.projects={};self.active_path=None;self.batch_results=[]
        self.timer=QTimer(self);self.timer.timeout.connect(self.advance)
        root=QWidget();self.setCentralWidget(root);outer=QVBoxLayout(root);outer.setContentsMargins(0,0,0,0);outer.setSpacing(0)
        nav=QWidget();nav.setObjectName("nav");nav.setFixedHeight(64);top=QHBoxLayout(nav);top.setContentsMargins(22,0,22,0);top.setSpacing(12)
        mark=QLabel("B");mark.setObjectName("mark");top.addWidget(mark)
        brand=QLabel("BLACKBOX MASK");brand.setObjectName("brand");top.addWidget(brand)
        subtitle=QLabel(f"얼굴 · 번호판 마스킹  /  시제품 {VERSION}");subtitle.setObjectName("navmuted");top.addWidget(subtitle);top.addStretch()
        badge=QLabel("●  로컬 처리");badge.setObjectName("badge");badge.setFixedHeight(26);top.addWidget(badge);outer.addWidget(nav)
        body=QWidget();outer.addWidget(body,1);layout=QVBoxLayout(body);layout.setContentsMargins(22,16,22,16);layout.setSpacing(12)
        self.toolbar=QWidget();bar=QHBoxLayout(self.toolbar);bar.setContentsMargins(0,0,0,0)
        bar.addWidget(button("＋ 영상 추가",self.add_files));bar.addWidget(button("프로젝트 열기",self.open_project))
        self.save_btn=button("프로젝트 저장",self.save_current);bar.addWidget(self.save_btn);bar.addStretch()
        self.analyze_btn=button("1. 선택 영상 분석",self.run_analysis,True);bar.addWidget(self.analyze_btn)
        self.batch_btn=button("전체 자동 처리",self.run_batch);bar.addWidget(self.batch_btn);layout.addWidget(self.toolbar)
        split=QSplitter(Qt.Horizontal);layout.addWidget(split,1)
        left=QWidget();lv=QVBoxLayout(left);lv.setContentsMargins(0,0,12,0)
        lv.addWidget(QLabel("영상 목록"));self.files=QListWidget();self.files.setMaximumHeight(140);self.files.currentRowChanged.connect(self.select_file);lv.addWidget(self.files)
        views=QHBoxLayout()
        self.original=Canvas(True);self.original.drawn.connect(self.add_box);self.output=Canvas()
        for title,canvas in [("원본 · 드래그하여 수동 영역 추가",self.original),("처리 미리보기",self.output)]:
            box=QVBoxLayout();label=QLabel(title);label.setObjectName("muted");box.addWidget(label);box.addWidget(canvas,1);views.addLayout(box,1)
        lv.addLayout(views,1)
        play=QHBoxLayout();self.prev=button("◀",lambda:self.goto(self.index-1));play.addWidget(self.prev)
        self.play=button("재생",self.toggle_play);play.addWidget(self.play);self.next=button("▶",lambda:self.goto(self.index+1));play.addWidget(self.next)
        self.slider=SeekSlider(Qt.Horizontal);self.slider.valueChanged.connect(self.goto);play.addWidget(self.slider,1)
        self.slider.sliderPressed.connect(self.scrub_start);self.slider.sliderReleased.connect(self.scrub_end)
        self.time_label=QLabel("분석 후 프레임 이동 가능");play.addWidget(self.time_label);lv.addLayout(play)
        hint=QLabel("자동 탐지는 누락될 수 있습니다. 제출 전에 전체 영상을 확인하고 필요한 영역을 추가하세요.")
        hint.setWordWrap(True);hint.setObjectName("muted");lv.addWidget(hint);split.addWidget(left)
        self.sidebar=QWidget();rv=QVBoxLayout(self.sidebar);rv.setContentsMargins(12,0,0,0);rv.setSpacing(10)
        detect=QGroupBox("자동 탐지 설정");dv=QVBoxLayout(detect)
        self.backend=QComboBox();self.backend.addItems(["자동 · Intel NPU 우선 / CPU 대체", "CPU", "Intel NPU 전용 · 실패 시 중단"])
        dv.addWidget(self.backend)
        self.faces=QCheckBox("얼굴");self.faces.setChecked(True);self.plates=QCheckBox("차량 번호판");self.plates.setChecked(True)
        dh=QHBoxLayout();dh.addWidget(self.faces);dh.addWidget(self.plates);dv.addLayout(dh)
        self.detail=QCheckBox("작은 얼굴·번호판 추가 탐지 (처리시간 증가)");dv.addWidget(self.detail)
        tip=QLabel("NPU는 AI 탐지에 사용 · 첫 분석은 모델 컴파일로 지연될 수 있음");tip.setObjectName("muted");tip.setWordWrap(True);dv.addWidget(tip);rv.addWidget(detect)
        mask=QGroupBox("마스킹 설정");form=QFormLayout(mask)
        self.style=QComboBox();self.style.addItems(["블러","모자이크","불투명 가리기"]);self.style.currentIndexChanged.connect(self.options_changed);form.addRow("처리 방식",self.style)
        self.margin=QDoubleSpinBox();self.margin.setRange(0,100);self.margin.setValue(18);self.margin.setSuffix(" %");self.margin.valueChanged.connect(self.options_changed);form.addRow("영역 여유 폭",self.margin);rv.addWidget(mask)
        manual=QGroupBox("수동 영역 · 시간 구간");mv=QVBoxLayout(manual);times=QFormLayout()
        self.start=QDoubleSpinBox();self.end=QDoubleSpinBox()
        for spin in (self.start,self.end):spin.setRange(0,999999);spin.setDecimals(3);spin.setSuffix(" 초")
        times.addRow("시작",self.start);times.addRow("끝",self.end);mv.addLayout(times)
        self.edit_key=QCheckBox("선택한 수동 영역에 현재 위치 추가");mv.addWidget(self.edit_key)
        hint=QLabel("다른 시점에서 같은 영역의 위치를 추가하면 사이 구간을 직선으로 연결합니다.")
        hint.setWordWrap(True);hint.setObjectName("muted");mv.addWidget(hint);rv.addWidget(manual)
        rv.addWidget(QLabel("현재 프레임의 대상 / 수동 영역"));self.tracks=QListWidget();self.tracks.setMinimumHeight(115);rv.addWidget(self.tracks,1)
        self.remove_btn=button("선택 대상 제외 / 복원 · 수동 영역 삭제",self.remove_region);rv.addWidget(self.remove_btn)
        self.audio=QCheckBox("음성 유지 (음성은 마스킹되지 않음)");rv.addWidget(self.audio)
        self.watermark=QCheckBox("MASKED COPY 워터마크");self.watermark.setChecked(True);rv.addWidget(self.watermark)
        self.reviewed=QCheckBox("전체 구간 검토 완료");rv.addWidget(self.reviewed)
        self.export_btn=button("2. 마스킹 MP4 저장",self.run_export,True)
        scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setWidget(self.sidebar);scroll.setMinimumWidth(340)
        scroll.setFrameShape(QScrollArea.NoFrame)
        right=QWidget();right_layout=QVBoxLayout(right);right_layout.setContentsMargins(0,0,0,0)
        right_layout.addWidget(scroll,1);right_layout.addWidget(self.export_btn);split.addWidget(right)
        split.setSizes([1050,365]);self.sidebar.setMinimumWidth(310)
        bottom=QHBoxLayout();self.status=QLabel("영상을 추가해 시작하세요.");bottom.addWidget(self.status,1)
        self.progress=QProgressBar();self.progress.setFixedWidth(230);self.progress.setRange(0,100);bottom.addWidget(self.progress)
        self.cancel_btn=button("취소",self.cancel_work);self.cancel_btn.setEnabled(False);bottom.addWidget(self.cancel_btn);layout.addLayout(bottom)
        self.update_enabled()

    def update_enabled(self):
        ready=self.project is not None and not self.busy
        self.toolbar.setEnabled(not self.busy);self.files.setEnabled(not self.busy);self.sidebar.setEnabled(not self.busy)
        self.analyze_btn.setEnabled(bool(self.active_path) and not self.busy)
        self.batch_btn.setEnabled(bool(self.queue) and not self.busy)
        for w in (self.save_btn,self.export_btn,self.remove_btn,self.original,self.slider,self.prev,self.next,self.play,
                  self.start,self.end,self.edit_key):w.setEnabled(ready)
        self.cancel_btn.setEnabled(self.busy)

    def set_progress(self,n,text):self.progress.setValue(n);self.status.setText(text)

    def work(self,fn,done):
        if self.busy:return
        self.timer.stop();self.play.setText("재생");self.busy=True;self.update_enabled()
        worker=Worker(fn);self.worker=worker;started=time.monotonic()
        worker.progress.connect(self.set_progress)
        worker.result.connect(done)
        worker.failed.connect(self.error)
        def finished():
            # result/failed 처리 뒤에 호출되므로 완료·실패 메시지 뒤에 총 소요 시간을 덧붙인다.
            self.status.setText(f"{self.status.text()} · 소요 시간 {hms(time.monotonic()-started)}")
            self.busy=False;self.update_enabled();worker.deleteLater()
            self.worker=None
        worker.finished.connect(finished);worker.start()

    def error(self,text):
        self.status.setText("작업이 완료되지 않았습니다.")
        QMessageBox.warning(self,"확인",text)

    def cancel_work(self):
        if self.worker:self.worker.cancel.set();self.status.setText("취소 중…")

    def add_files(self):
        paths,_=QFileDialog.getOpenFileNames(self,"영상 선택","","영상 (*.mp4 *.avi *.mov *.mkv *.ts *.mts);;모든 파일 (*)")
        # analyze()가 project["source"]를 resolve 경로로 저장하므로 목록 경로도 같은 형식으로 맞춘다.
        # Windows에서는 대화상자 경로(C:/…)와 resolve 경로(C:\…)가 달라 projects 조회가 실패해 마스킹이 사라졌다.
        paths=[str(Path(p).resolve()) for p in paths]
        for p in paths:
            if p not in self.queue:
                self.queue.append(p);item=QListWidgetItem(Path(p).name);item.setToolTip(p);self.files.addItem(item)
        if paths:self.files.setCurrentRow(self.queue.index(paths[0]))
        self.update_enabled()

    def select_file(self,row):
        if row<0 or row>=len(self.queue) or self.busy:return
        if self.project:self.projects[self.project["source"]]=self.project
        self.timer.stop();self.play.setText("재생");self.active_path=self.queue[row]
        self.project=self.projects.get(self.active_path);self.index=0;self.raw=None;self.tracks.clear();self.reviewed.setChecked(False)
        try:
            if self.project:self.install_project(self.project)
            else:
                with av.open(self.active_path) as src:
                    frame=next(src.decode(video=0));self.raw=frame_image(frame)
                self.original.set_image(self.raw);self.output.set_image(self.raw)
                self.status.setText("선택 영상 분석을 누르세요.");self.time_label.setText("분석 후 프레임 이동 가능")
        except Exception as e:self.error(str(e))
        self.update_enabled()

    def settings(self):return {"faces":self.faces.isChecked(),"plates":self.plates.isChecked(),"detail":self.detail.isChecked(),"backend":["auto","cpu","npu"][self.backend.currentIndex()]}
    def options(self):return {"style":["blur","mosaic","solid"][self.style.currentIndex()],"margin":self.margin.value()/100,
                              "audio":self.audio.isChecked(),"watermark":self.watermark.isChecked(),"reviewed":self.reviewed.isChecked()}

    def run_analysis(self):
        if self.project and QMessageBox.question(self,"다시 분석","현재 영상의 수동 보정과 제외 설정을 초기화하고 다시 분석할까요?")!=QMessageBox.Yes:return
        if not self.faces.isChecked() and not self.plates.isChecked():
            self.status.setText("자동 탐지 없이 수동 편집용으로 프레임을 읽습니다.")
        p=self.active_path;settings=self.settings()
        self.work(lambda progress,cancel:analyze(p,settings,progress,cancel),self.install_project)

    def install_project(self,project):
        self.project=project;self.active_path=project["source"];self.projects[self.active_path]=project
        self.index=0;self.start.setValue(0);self.end.setValue(project["meta"]["duration"])
        self.slider.blockSignals(True);self.slider.setRange(0,len(project["frames"])-1);self.slider.setValue(0);self.slider.blockSignals(False)
        self.reviewed.setChecked(False);self.progress.setValue(100);self.goto(0,force=True)
        tracks={b['id'] for f in project['frames'] for b in f['boxes']}
        self.status.setText(f"분석 완료 · {len(project['frames']):,} 프레임 · 추적 대상 {len(tracks)}개 · 누락 영역을 검토하세요.")
        devices=project.get("inference",{})
        if devices:
            self.status.setText(self.status.text()+" · "+" / ".join(f"{'머리' if k=='face' else '번호판'}: {v['device']}" for k,v in devices.items()))
            self.status.setToolTip("\n".join(f"{k}: {v.get('fallback_reason','')}" for k,v in devices.items() if v.get('fallback_reason')))
        self.dirty=True;self.update_enabled()

    def goto(self,index,force=False):
        if not self.project or (self.busy and not force):return
        index=max(0,min(len(self.project["frames"])-1,int(index)))
        self.index=index
        try:self.raw=read_preview(self.project,index)
        except Exception as e:self.timer.stop();self.error(str(e));return
        self.slider.blockSignals(True);self.slider.setValue(index);self.slider.blockSignals(False)
        t=self.project["frames"][index]["time"]-self.project["meta"]["origin"]
        self.time_label.setText(f"{t:.3f}초 · {index+1:,}/{len(self.project['frames']):,}")
        self.render();self.update_tracks()

    def render(self):
        if self.raw is None or not self.project:return
        boxes=effective_boxes(self.project,self.index)
        self.original.set_image(self.raw,boxes)
        opts=self.options();self.output.set_image(apply_masks(self.raw,boxes,opts["style"],opts["margin"]))

    def options_changed(self,*args):
        self.reviewed.setChecked(False);self.dirty=True;self.render()

    def toggle_play(self):
        if self.timer.isActive():self.timer.stop();self.play.setText("재생")
        elif self.project:
            self.play.setText("일시정지");self.timer.start(50)

    def scrub_start(self):
        # 재생 중 드래그하면 타이머의 goto가 슬라이더 값을 덮어쓰고 프레임 디코딩이 이벤트 루프를 막아 조작이 안 된다.
        self.resume=self.timer.isActive();self.timer.stop()

    def scrub_end(self):
        if self.resume and self.project:self.timer.start(50)

    def advance(self):
        if not self.project or self.index>=len(self.project["frames"])-1:self.timer.stop();self.play.setText("재생");return
        old=self.index;self.goto(old+1)
        interval=1000*(self.project["frames"][self.index]["time"]-self.project["frames"][old]["time"])
        self.timer.setInterval(max(10,min(1000,int(interval))))

    def update_tracks(self):
        selected=self.tracks.currentItem();old=selected.data(Qt.UserRole) if selected else None
        self.tracks.clear()
        entries=[]
        for b in self.project["frames"][self.index]["boxes"]:
            state="제외" if b["id"] in self.project["excluded"] else ("추정 위치" if b.get("predicted") else "마스킹")
            entries.append((b["id"],f"#{b['id']} {'얼굴' if b['kind']=='face' else '번호판'} · {state}"))
        for m in self.project["manual"]:entries.append((m["id"],f"#{m['id']} 수동 · {m['start']:.1f}–{m['end']:.1f}초 · 위치 {len(m['keys'])}개"))
        for tid,text in entries:
            item=QListWidgetItem(text);item.setData(Qt.UserRole,tid);self.tracks.addItem(item)
            if tid==old:self.tracks.setCurrentItem(item)

    def add_box(self,box):
        if not self.project or self.busy:return
        start,end=self.start.value(),self.end.value()
        if end<start:self.error("끝 시각은 시작 시각보다 뒤여야 합니다.");return
        t=self.project["frames"][self.index]["time"]-self.project["meta"]["origin"]
        if not start<=t<=end:self.error("현재 프레임이 적용 시간 구간 안에 있어야 합니다.");return
        selected=self.tracks.currentItem();tid=selected.data(Qt.UserRole) if selected else None
        item=None
        if self.edit_key.isChecked():
            item=next((m for m in self.project["manual"] if m["id"]==tid),None)
            if item is None:self.error("목록에서 수동 영역을 먼저 선택하세요.");return
        if item is None:
            tid=min([0]+[m["id"] for m in self.project["manual"]])-1
            item={"id":tid,"start":start,"end":end,"keys":[]};self.project["manual"].append(item)
        item["start"],item["end"]=start,end
        item["keys"]=[k for k in item["keys"] if abs(k["time"]-t)>1e-6]+[{"time":t,"box":box}]
        self.dirty=True;self.reviewed.setChecked(False);self.render();self.update_tracks()
        for n in range(self.tracks.count()):
            if self.tracks.item(n).data(Qt.UserRole)==tid:self.tracks.setCurrentRow(n)

    def remove_region(self):
        item=self.tracks.currentItem()
        if not self.project or not item:return
        tid=item.data(Qt.UserRole)
        if tid<0:self.project["manual"]=[m for m in self.project["manual"] if m["id"]!=tid]
        elif tid in self.project["excluded"]:self.project["excluded"].remove(tid)
        else:
            if QMessageBox.question(self,"대상 제외",f"#{tid}의 전체 추적 구간에서 마스킹을 제외할까요?\n추적 대상이 바뀐 구간도 반드시 확인하세요.")!=QMessageBox.Yes:return
            self.project["excluded"].append(tid)
        self.dirty=True;self.reviewed.setChecked(False);self.render();self.update_tracks()

    def save_current(self):
        if not self.project:return
        p,_=QFileDialog.getSaveFileName(self,"프로젝트 저장",str(Path(self.project["source"]).with_suffix(".maskproj.json")),"프로젝트 (*.json)")
        if p:
            try:
                self.project["export_options"]=self.options();save_project(self.project,p);self.dirty=False
                self.status.setText("프로젝트 저장 완료 · 원본 영상도 함께 보관하세요.")
            except Exception as e:self.error(str(e))

    def open_project(self):
        p,_=QFileDialog.getOpenFileName(self,"프로젝트 열기","","프로젝트 (*.json)")
        if not p:return
        def done(project):
            path=project["source"]
            if path not in self.queue:self.queue.append(path);self.files.addItem(Path(path).name)
            self.files.blockSignals(True);self.files.setCurrentRow(self.queue.index(path));self.files.blockSignals(False)
            self.install_project(project)
            opts=project.get("export_options",{})
            self.style.setCurrentIndex(["blur","mosaic","solid"].index(opts.get("style","blur")))
            self.margin.setValue(opts.get("margin",.18)*100)
            self.audio.setChecked(opts.get("audio",False));self.watermark.setChecked(opts.get("watermark",True))
        self.work(lambda progress,cancel:load_project(p),done)

    def run_export(self):
        if not self.project:return
        if not self.reviewed.isChecked():
            if QMessageBox.question(self,"검토 전 출력","전체 구간 검토가 완료되지 않았습니다. 검토용 파일로 저장할까요?")!=QMessageBox.Yes:return
        p,_=QFileDialog.getSaveFileName(self,"MP4 저장",str(Path(self.project["source"]).with_name(Path(self.project["source"]).stem+"_masked.mp4")),"MP4 (*.mp4)")
        if not p:return
        if not p.lower().endswith(".mp4"):p+=".mp4"
        project=copy.deepcopy(self.project);options=self.options()
        self.work(lambda progress,cancel:export_video(project,p,options,progress,cancel),self.export_done)

    def export_done(self,path):
        self.progress.setValue(100);self.status.setText("저장 완료 · 원본 파일은 변경하지 않았습니다.")
        QMessageBox.information(self,"저장 완료",str(path))

    def run_batch(self):
        directory=QFileDialog.getExistingDirectory(self,"일괄 처리 결과 폴더")
        if not directory:return
        if QMessageBox.question(self,"전체 자동 처리",f"{len(self.queue)}개 영상을 새로 자동 분석하여 저장합니다.\n일괄 처리에는 현재 수동 보정·제외 설정이 적용되지 않습니다.\n생성 파일은 검토 전 결과입니다. 계속할까요?")!=QMessageBox.Yes:return
        paths=list(self.queue);settings=self.settings();options=self.options();options["reviewed"]=False
        def run(progress,cancel):
            results=[]
            for n,path in enumerate(paths):
                if cancel():raise Cancelled("일괄 처리를 취소했습니다. 이미 완료된 파일은 유지됩니다.")
                try:
                    def report(value,text):progress(int((n+value/100)/len(paths)*100),f"[{n+1}/{len(paths)}] {Path(path).name} · {text}")
                    project=analyze(path,settings,report,cancel)
                    dest=Path(directory)/(Path(path).stem+"_masked.mp4");suffix=1
                    while dest.exists() or dest.with_suffix(".maskproj.json").exists():
                        dest=Path(directory)/(Path(path).stem+f"_masked_{suffix}.mp4");suffix+=1
                    save_project(project,dest.with_suffix(".maskproj.json"))
                    export_video(project,dest,options,report,cancel)
                    results.append(f"완료: {dest.name}")
                except Cancelled:raise
                except Exception as e:results.append(f"실패: {Path(path).name} — {e}")
            return "\n".join(results)
        self.work(run,self.export_done)

    def closeEvent(self,event):
        if self.busy:
            self.cancel_work();event.ignore();return
        if self.dirty and self.projects:
            if QMessageBox.question(self,"종료","저장하지 않은 편집 내용이 있을 수 있습니다. 종료할까요?")!=QMessageBox.Yes:event.ignore();return
        event.accept()


def main():
    app=QApplication(sys.argv);app.setStyle("Fusion");app.setFont(QFont("Malgun Gothic",10))
    pal=app.palette()
    for role,color in [(QPalette.Window,"#F7F9FC"),(QPalette.Base,"#FFFFFF"),(QPalette.AlternateBase,"#F7F9FC"),(QPalette.Button,"#FFFFFF"),
                       (QPalette.Text,"#1A2B4A"),(QPalette.WindowText,"#1A2B4A"),(QPalette.ButtonText,"#1A2B4A"),
                       (QPalette.Highlight,"#F37321"),(QPalette.HighlightedText,"#FFFFFF")]:pal.setColor(role,QColor(color))
    app.setPalette(pal);app.setStyleSheet(STYLE)
    window=Window();window.show();return app.exec()


if __name__=="__main__":sys.exit(main())

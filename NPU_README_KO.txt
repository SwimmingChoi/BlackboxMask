2026-10-13: Intel GPU(내장 그래픽) 추론 추가
- 일부 PC에서 고정밀/최고 정밀 머리 모델이 NPU 추론 중 ZE_RESULT_ERROR_DEVICE_LOST(장치 멈춤)로
  실패했습니다. 기본 모델은 같은 NPU에서 정상입니다. 큰 모델은 Intel GPU를 권장합니다.
- 장치 선택에 [Intel GPU 전용 · 실패 시 중단]을 추가했습니다. OpenVINO GPU 플러그인을 사용하며
  Intel GPU만 사용합니다(NVIDIA 등 다른 GPU가 목록에 보여도 사용하지 않음).
- 고정밀·최고 정밀 모델은 NPU에 올리지 않습니다. 한 번 멈추면 재부팅 전까지 NPU가 사라져
  기본 모델도 "Intel NPU 장치가 감지되지 않습니다"로 실패했기 때문입니다.
  자동 모드는 이 모델을 Intel GPU → CPU로 처리하고, NPU 전용 모드는 분석 시작 전에 중단합니다.
  Diagnose.cmd도 이 조합의 NPU 시험을 건너뜁니다.
- 기본 모델의 자동 모드는 NPU → Intel GPU → CPU 순으로 대체합니다. 대체 사유는 상태 표시 툴팁에 장치별로 남습니다.
- CPU 비교 감시(최초 3프레임, 이후 60프레임마다)는 Intel GPU에도 동일하게 적용됩니다.
- Diagnose.cmd가 장치별 드라이버 버전과 NPU/GPU 모델 조합별 처리 시간 또는 실패를 출력합니다.
- 실제 Intel GPU 속도/호환성은 개발 환경에서 확인하지 못했습니다. 목표 PC에서 Diagnose.cmd로 확인하세요.

2026-10-12: 탐지 모델 선택 추가 (자동 탐지 설정 > 탐지 모델)
- 기본 · 빠름: 머리 YOLOX-Nano + 번호판 YOLOv9-t 640 (기존과 동일)
- 고정밀: 머리 YOLOX-S + 번호판 YOLOv9-s 608 (CPU 기준 약 7배 연산)
- 최고 정밀: 머리 YOLOX-M + 번호판 YOLOv9-s 608 (CPU 기준 약 18배 연산, Intel GPU 권장)
공개 COCO 계열 기준 머리 AP: Nano 46.2 / S 52.6 / M 54.7, 작은 물체 AP: 21.8 / 29.5 / 34.3.
번호판 재현율: t-640 0.896 / s-608 0.917. 한국 블랙박스 영상에서의 정확도는 아직 검증하지 않았습니다.
Diagnose.cmd가 모델 조합별 NPU 처리 시간(1080p 프레임 1장 기준)을 출력합니다.
NPU 실패로 CPU 대체되면 고정밀 모델은 매우 느려질 수 있습니다.
번호판 모델은 NPU에서 NMS 이전까지만 실행하고 NMS는 CPU에서 동일 규칙으로 처리합니다.

2026-10-12: 동봉 runtime에 OpenVINO 2026.3.0(Intel NPU 플러그인 포함)을 미리 설치했습니다.
대상 PC에서 Python/pip 설치나 인터넷 연결 없이 사용할 수 있습니다. Install_NPU.cmd 실행은 필요 없으며,
Diagnose.cmd로 NPU 인식/컴파일을 확인한 뒤 BlackboxMask.exe를 실행하면 됩니다.

Intel NPU 지원 추가
==================
탐지 결과 비교: 영상에서 문제가 보이는 프레임을 선택하고 [CPU / NPU 비교]를 누릅니다.
같은 프레임을 ONNX CPU, OpenVINO FP32 CPU, Intel NPU에 넣어 비교 JSON을 저장합니다.
이미지/영상은 업로드하지 않으며 보고서에는 박스, 입력/출력 형태, 값 범위, 장치 정보가 들어갑니다.
첫 컴파일 시간이 포함되므로 비교 시간은 정상 처리속도 측정값이 아닙니다.
자동 분석은 최초 3프레임과 이후 60프레임마다 NPU 탐지를 CPU와 비교합니다.
IoU 0.5 이상인 일대일 박스 일치율이 50% 미만이면 자동 모드는 해당 모델을 CPU로 전환하고
현재 프레임도 CPU 결과로 교체합니다. NPU 전용 모드는 중단합니다.
표본 비교는 전체 정확도를 보장하지 않으며 CPU 탐지가 정답이라는 의미도 아닙니다.
검사 때문에 시작과 주기적 프레임의 처리시간/메모리가 늘어납니다.

대상: Intel AI Boost NPU + Windows NPU 드라이버. AMD/Qualcomm NPU는 지원하지 않습니다.

기존 휴대용 폴더에는 OpenVINO가 없으므로 Install_NPU.cmd를 한 번 실행합니다.
별도 Python 설치는 필요하지 않습니다. 동봉 runtime/python.exe를 사용합니다.
pip가 없으면 공식 bootstrap.pypa.io에서 설치 도구를 내려받아 동봉 runtime에 설치합니다.
설치 때만 인터넷이 필요하며, 기존 runtime에 OpenVINO와 필요한 의존성을 설치합니다.
설치 전 폴더를 복사해 두면 이전 환경으로 복구할 수 있습니다.

회사망에서 다운로드가 차단되는 경우: 승인된 인터넷 PC에서 Python/pip를 준비하고
Download_NPU_Wheels.cmd를 실행합니다. 만들어진 npu-wheels 폴더 전체를 승인된 경로로
대상 PC의 BlackboxMask 폴더에 복사한 뒤 Install_NPU.cmd를 실행합니다.
pip wheel과 OpenVINO 및 모든 의존성의 Windows x64 Python 3.12 wheel이 필요합니다.
npu-wheels가 있으면 외부 다운로드 없이 설치하며 누락 파일이 있어도 온라인으로 전환하지 않습니다.
대상 PC에는 별도 Python/pip가 없어도 됩니다. pip는 로컬 wheel에서 직접 실행합니다.
조직이 지정한 내부 패키지 저장소나 담당자의 승인을 받은 패키지를 사용하세요.
설치 후 기존 BlackboxMask.exe를 실행하면 수정된 소스를 사용합니다.
재빌드는 build/build_windows.cmd로 가능하며 OpenVINO도 포함합니다.

자동 탐지 설정에서 장치를 선택합니다.
- 자동: Intel NPU 시도, 모델별 로딩/컴파일/추론 실패 시 Intel GPU, 그것도 실패하면 CPU 전환.
- CPU: 이전 ONNX Runtime 처리.
- Intel NPU 전용: NPU 실패 시 중단. CPU로 조용히 전환하지 않습니다.
- Intel GPU 전용: Intel GPU 실패 시 중단.
분석 완료 상태에 머리/번호판별 실제 사용 장치가 표시됩니다.
자동 모드 CPU 전환 사유는 상태 표시의 툴팁과 프로젝트의 inference 항목에 기록합니다.
출력 처리 기록에도 inference 항목을 저장합니다.

기존 탐지 전처리와 원본 좌표를 유지합니다. 각 입력 크기를 정적 크기로 컴파일하고
사용자 로컬 폴더의 npu-cache에 캐시합니다. 첫 분석과 새 영상 크기에서는 지연이 있습니다.
번호판 ONNX 모델의 동적 출력/지원하지 않는 연산 등으로 NPU 컴파일이 실패할 수 있습니다.
Diagnose.cmd는 실제 두 모델의 NPU 컴파일/추론까지 확인합니다.
프로젝트를 다시 분석해야 새 장치가 사용되며, 기존 프로젝트 출력만으로는 NPU를 쓰지 않습니다.
영상 디코딩, 추적, 마스킹, MP4 인코딩은 계속 CPU에서 처리합니다.

현재 개발 환경은 WSL이며 Windows 프로세스 실행과 NPU 장치 접근이 불가합니다.
따라서 실제 NPU 속도/두 모델의 하드웨어 호환성은 목표 Windows PC에서 확인해야 합니다.
README_KO.txt의 CPU 전용 설명은 원래 0.1.2 배포본 기준입니다.
이번 소스 변경 이후 BUILD_MANIFEST.json의 변경 파일 해시는 이전 배포본 기준입니다.

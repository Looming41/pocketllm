# PocketLLM 종합 안내 (밤사이 작업 결과, 2026-10-27)

## 한 줄 요약
**"메모리가 모자란 노트북에서도 30B~100B급 MoE 모델을 USB에서 스트리밍해 오프라인으로 돌리고, USB를 더 꽂으면 더 빨라지는 보안 기관용 로컬 AI 키트"**. 핵심 엔진은 실험실에서 기본 llama.cpp 대비 **6.6배(드라이브 1개)에서 12.9배(드라이브 2개)** 빨랐습니다.

## 1. 무엇을 만들었나
| 구성 | 내용 | 검증 수준 |
|---|---|---|
| **전문가 스트리밍 엔진** (`patches/`, `docs/ENGINE_KO.md`) | llama.cpp MoE 커널에 훅을 추가해, 라우터가 고른 전문가만 즉시 동시에 읽음. 이웃 전문가 낭비 제거 | 에뮬레이션 실험 ✅, 출력 동일성 ✅, 실제 모델 ⏳ |
| **멀티 USB 스트라이핑** (`pocketllm stripe`, `make_usb.sh --stripe`, `drives.py`) | 전문가 텐서를 드라이브별로 나눠 담고, 어느 컴퓨터에서든 꽂힌 드라이브를 찾아 자동 조립 | 동일 출력 ✅, 2드라이브 1.8배 ✅(에뮬레이션) |
| **멀티 기기 클러스터** (`pocketllm node`, `cluster`) | llama.cpp RPC로 여러 기기의 메모리를 합침(노드 메모리 비례 분할) | 로컬 2노드 동일 출력 ✅ |
| **폐쇄망 모드** (`docs/SECURITY_KO.md`) | 정책 파일, 패킷 없는 네트워크 감지, 서명 매니페스트, 세션 API 키, Origin 검사, 메타데이터 감사 로그, 암호화 상태 | 변조·추가 파일·서명 위조 탐지 테스트 ✅ |
| **Meta 2구역 설계** (`docs/META_KO.md`) | 준비 구역: Muse Spark로 fleet-plan, 평가셋, 채점 → 현장: Meta 오픈 웨이트만, 네트워크 0 | 모의 API 테스트 ✅, 실제 API ⏳ |
| **열·전력 프로파일** (`pocketllm thermal`) | 팬리스 기기용 cool/balanced/max, macOS 열 압력 감지 | Linux만 실행 ✅, Mac ⏳ |
| **플래너, 튜너, UI** (기존 + 확장) | 다운로드 전 판정, `expert_stream` 전략, 엔진 환경 변수 자동 설정, 배지 | 테스트 30개 이상 ✅, Playwright E2E ✅ |
| **조사** (`docs/LANDSCAPE_KO.md`) | GitHub 프로젝트 약 50개를 클론해 활동성과 라이선스를 검증하고 원리를 7가지로 분류 | ✅ |

## 2. "USB가 효율을 높인다"는 말이 물리적으로 맞는 범위
- USB 저장 장치는 **연산(TOPS)을 더하지 않습니다.** 발열도 노트북 몫입니다.
- 더해 주는 것은 **용량과 읽기 대역폭**입니다. MoE 모델이 RAM보다 크면 속도는 "필요한 전문가를 얼마나 빨리 읽느냐"로 정해집니다. PocketLLM 엔진은 그 읽기를 **필요한 만큼만, 동시에, 여러 드라이브에서** 하게 만듭니다.
- 맥북 에어의 발열 문제는 **모델 선택**으로 줄입니다. Qwen3-30B-A3B(활성 3.3B)는 토큰당 연산이 7B dense의 절반 수준입니다. 그 위에 열 프로파일로 스레드를 제한합니다.
- 연산까지 늘리려면 연산 장치가 있는 노드를 연결합니다(클러스터 모드). "USB 컴퓨트 스틱"은 하드웨어 로드맵(`docs/ROADMAP_HW_KO.md`)에 있습니다.

## 3. 내일 아침 할 일 (맥북 에어 기준)
```bash
git clone <저장소> pocketllm && cd pocketllm
xcode-select --install                       # 컴파일러가 없을 때만
scripts/build_llamacpp.sh                    # 고정 llama.cpp + 엔진 패치 + Metal + RPC
scripts/check_laptop.sh                      # 사양 보고서 → data/laptop_report.txt (보내 주세요)

# 실제 MoE 모델로 엔진을 검증(예: Qwen3-30B-A3B Q4_K_M, 약 19GB)
./bin/macos-arm64/llama-server -hf unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M --port 9999   # 다운로드만 하고 Ctrl-C
python3 -m pocketllm tune <받은 gguf 경로>    # llama.cpp 기본 vs PocketLLM 엔진 실측
```
- 다음 세 가지를 비교해 주세요: ① llama.cpp 기본 ② `POCKET_MOE=1 POCKET_MOE_NORA=1` ③ ②에 `POCKET_MOE_IO_THREADS=8`을 더한 것(macOS에서는 ③이 나을 수 있음)
- USB가 2개 있으면 `scripts/make_usb.sh --stripe /Volumes/A /Volumes/B -- model.gguf`로 만든 뒤 같은 측정을 반복합니다.

## 4. 아직 안 된 것 / 확인 필요
- **실제 모델 실행 결과 없음.** 이 클라우드 환경에서는 Hugging Face가 차단돼 무작위 가중치 모델로만 실험했습니다.
- **Mac에서 엔진 미실행.** 코드는 macOS 경로(`F_RDADVISE`, `pread` 스레드)를 포함하지만 Mac에서 컴파일하거나 실행해 보지 않았습니다.
- Metal과 스트리밍을 함께 쓰는 하이브리드(attention은 GPU, 전문가는 스트리밍)는 미검증입니다. 그래서 지금은 CPU 경로로 고정했습니다.
- Meta Model API 실제 호출, 해커톤 공식 규정 확인(dev.meta.ai가 차단됨)
- Windows용 엔진 없음(추론은 동작)

## 5. 실험 수치 출처
`docs/RESULTS.md`(R1~R7), 원자료 `docs/results/*.json`, 표 재생성은 `python3 scripts/lab_report.py docs/results/*.json`

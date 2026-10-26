# 대형 모델(100B~200B) 실행 실험 계획

원칙: 제품 데모(P0)와 **분리된 트랙**으로 진행합니다. 각 실험에는 "중단 조건"을 두어, 하드웨어가 지원하지 않으면 빨리 증명하고 끝냅니다. 측정하지 않은 수치는 쓰지 않습니다.

## 통제 변수 (모든 실험 공통)
- llama.cpp 커밋 `caa8191`(`bin/*/VERSION`에 기록), 백엔드(Metal/CUDA/Vulkan/CPU)
- 프롬프트: `llama-bench` 기본 합성 프롬프트(`-p 128 -n 64 -r 3`)와 고정 대화 프롬프트 1개(`docs/prompts/fixed.txt`)
- 컨텍스트 4096, 배치 기본값, 전원 연결, 다른 앱 종료, 같은 실내 온도
- 기록: `data/tuning/*.json`(tuner), `data/usb_vs_ssd.json`, `data/laptop_report.txt`

## E1. 사양 확인과 판정 (Day 1, 10분)
```
scripts/build_llamacpp.sh && scripts/check_laptop.sh
```
**산출물:** RAM, VRAM, 대역폭, 저장 장치 속도, 모델별 판정. 이 결과로 아래 실험의 분기를 결정합니다.

| 노트북 메모리 (RAM + VRAM) | Scout 실험 경로 |
|---|---|
| < 24GB | E3과 E4만(페이징 실측으로 "불가"를 증명). 실제 데모는 8B 이하 |
| 24~48GB | Scout IQ1_S/IQ2_XXS(21~28GB 추정)로 E4 진행 |
| 48~80GB | Scout Q2_K/Q3_K_M, MoE 분할 `-ncmoe` 탐색 |
| ≥ 80GB 또는 Mac 96GB 이상 | Scout Q4_K_M, gpt-oss-120b MXFP4 |

## E2. 소형 Meta 모델 기준선 (Day 1~2, P0)
- Llama 3.2 3B Q4_K_M, Llama 3.1 8B Q4_K_M(가능하면 Muse Glimmer Q4 GGUF도)
- `python -m pocketllm tune models/<m>.gguf`로 기본 설정과 튜닝 설정 비교
- **측정:** 로딩 시간, TTFT, 생성 tok/s, 프롬프트 tok/s, 최대 RSS/VRAM, 오프라인 동작 여부

## E3. USB와 내부 SSD (Day 2~3)
```
scripts/usb_vs_ssd.sh /media/POCKETLLM/PocketLLM/models/<8B>.gguf ~/pocketllm-ssd
```
- 콜드 읽기 GB/s, `-lm none` 로딩 시간, `-lm mmap` 첫 토큰과 생성 속도
- 엄격한 콜드 측정: 실행 사이에 USB를 뽑았다 다시 꽂습니다(OS 캐시 제거)
- **가설:** RAM에 들어가는 모델은 로딩 시간만 다르고, 생성 속도는 같다

## E4. 페이징 한계 실측 (Day 3~5, 중단 조건 있음)
RAM보다 큰 모델을 일부러 mmap으로 실행해, 플래너의 페이징 모델(속도 ≈ 저장 장치 속도 ÷ 토큰당 페이징 바이트)을 검증합니다.
- 모델: RAM의 약 1.3배 크기 GGUF 1개(예: 16GB 노트북 → Llama 3.1 8B Q8 + 20B급, 또는 Scout IQ1_S)
- `llama-bench -lm mmap -n 16 -r 1`(오래 걸리므로 생성 토큰 수를 줄임)
- **중단 조건:** 첫 토큰이 120초 넘게 걸리거나 시스템이 응답하지 않으면 중단하고 그 사실을 기록합니다. 이것 자체가 "불가"의 증거입니다.
- **산출물:** 플래너 추정치와 실측의 비교표 → 플래너 보정

## E5. Llama 4 Scout 실행 (Day 4~6, 메모리가 허용할 때만)
1. 메모리에 맞는 가장 높은 양자화를 선택합니다(E1 판정 기준).
2. `python -m pocketllm inspect`로 GGUF 헤더의 실제 수치(전문가 바이트, KV/토큰)를 확인하고 `catalog.py` 추정치를 보정합니다.
3. GPU가 있으면 `-ncmoe` 스윕: 플래너 값 −4/−2/0/+2/+4 → 생성 tok/s와 VRAM 최대치
4. `ik_llama.cpp`(NVIDIA인 경우)와 같은 설정으로 비교합니다(선택).
5. 품질 확인: 고정 질문 20개(한국어 10, 영어 10)에 대해 8B와 Scout의 답변을 나란히 비교합니다(정성 평가, 블라인드).
6. (선택) `llama-perplexity`로 wikitext 일부 PPL을 측정해 양자화 간 비교합니다.

## E6. 기본 설정 vs 자동 최적화 (Day 6, P1)
- 같은 모델과 프롬프트에서 `llama.cpp defaults`와 PocketLLM 튜닝 결과를 비교합니다(이미 `tuner.py`에 구현).
- 개선이 잡음 범위 안이면 "개선 없음"으로 정직하게 보고합니다.

## E7. 오프라인 데모 검증 (Day 8)
- Wi-Fi를 끈 상태에서 모델 로드, 채팅, 튜닝을 해 봅니다. UI 배지가 `network: offline`인지, Meta 도우미가 오프라인 규칙으로 자동 전환되는지 확인합니다.

## 결과 보고 형식
`docs/RESULTS.md`에 "환경 E1(노트북)" 절을 추가하고, 실험마다 설정 → 측정값 → 해석 → 실패 시 원인 순으로 기록합니다.

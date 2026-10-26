# PocketLLM 조사 보고서 (2026-10-27 기준)

표기 규칙
- ✅ **직접 확인**: 저장소를 클론해 소스·커밋·README를 확인했거나 직접 실행해 본 내용
- 🔎 **2차 출처**: 웹 검색 결과(기사·블로그)로만 확인한 내용. 공식 페이지 재확인 필요
- ❌ **확인 불가**: 이 작업 환경의 네트워크 정책 때문에 접속하지 못한 출처

> 작업 환경 제약: 클라우드 빌드 컨테이너에서 `dev.meta.ai`, `ai.meta.com`, `huggingface.co`, `ollama.com`, 제조사 사이트(tiiny.ai, asus.com, nvidia.com, mobilint.com)가 **프록시에서 차단**됐습니다. GitHub 공개 저장소 `git clone`과 웹 검색만 가능했습니다. 그래서 해커톤 공식 규정과 제조사 스펙은 **2차 출처**로만 정리했고, 제출 전에 반드시 직접 확인해야 합니다.

---

## 1. 해커톤 공식 조건 검증 결과

| 항목 | 내용 | 확인 수준 |
|---|---|---|
| 공식 페이지 | https://dev.meta.ai/events/global-hackathon | ❌ 접속 차단 |
| 제출 마감 | **2026-11-04 21:00 ET** (= 11-05 UTC). 다수 출처가 일치 | 🔎 |
| 심사 / 발표 | 11-04~08 심사, 11-09 수상 발표 (출처 1곳) | 🔎 |
| 트랙 | **Indie Developer**(개인·학생·최대 4인 팀): 창의성, 기술적 야심, 실제 작동 여부 / **Startup**(실제 제품을 만드는 팀): 기술적 실행, 프로덕션 내구성 | 🔎 |
| 테마 | Build for Your World / Push the Frontier | 🔎 |
| 심사 비중 30/25/20/25 | **검증 실패.** 출처마다 다름: "impact·Meta 도구 활용·기술 깊이·데모"(비중 없음), "아이디어·기술 동등 가중" 등 | 🔎 상충 |
| 사용 가능한 Meta 기술 | Meta Model API의 Muse Spark 시리즈, Muse Image, SAM 3.1, Meta 오픈 웨이트 모델, Muse Code | 🔎 |
| 상금 | 총 $1M. 배분은 출처마다 다름 | 🔎 상충 |
| 크레딧 | $150 크레딧 언급(1개 출처), API 가입 시 $20 크레딧(별도 출처) | 🔎 |

**해야 할 일:** 공식 페이지에서 ① 마감 시각, ② 심사 기준과 비중, ③ "Meta 기술 사용"의 필수 여부와 인정 범위(오픈 웨이트 모델의 로컬 실행도 인정되는지), ④ 데모 영상 길이와 제출 양식, ⑤ 본인 트랙을 직접 확인해 이 표를 갱신하세요.

### Meta Model API (🔎, 공식 문서 페이지는 ❌)
- Base URL `https://api.meta.ai/v1`, `Authorization: Bearer $MODEL_API_KEY`
- OpenAI 호환 Chat Completions, Responses, Messages 엔드포인트
- 모델 ID: 문서에 `muse-spark-1.1/1.2/1.3`이 보임. **`/v1/models`로 실제 목록을 확인**하세요. PocketLLM은 `POCKETLLM_META_MODEL` 환경 변수로 바꿀 수 있습니다.
- Muse Spark의 미국 public preview 언급이 있음. **한국에서 API 접근이 되는지 확인 필요**

### Meta 오픈 웨이트 모델과 로컬 실행
| 모델 | 규모 | 라이선스 | llama.cpp 지원 | 비고 |
|---|---|---|---|---|
| Llama 4 Scout | 총 109B / 활성 17B, 전문가 16개 | Llama 4 Community | ✅ `llama4` 아키텍처가 소스에 있음 | 100B급 목표 모델. Q4_K_M ≈ 60.9 GiB 🔎 |
| Muse Glimmer | 30B (dense라는 보고 1건), 4-bit에서 20GB 미만 | **Apache 2.0** 🔎 | ✅ `muse-glimmer` 아키텍처가 소스에 있음 | **로컬 실행 가능한 최신 Meta 모델.** GGUF 배포 여부는 HF에서 확인 필요 |
| Muse Spark Open | 816B / 활성 42B | Muse Model License 🔎 | ✅ `muse-spark` 아키텍처가 소스에 있음 | 2-bit도 250GB 이상 🔎. 노트북에서는 불가 |
| Llama 3.1 8B / 3.2 3B | dense | Llama Community | ✅ | UX 데모용 기본 모델 |

→ **권장:** "Meta 기술 활용"을 두 축으로 구성합니다. ① Meta Model API(Muse Spark)를 설정 도우미로 사용(클라우드, 선택 기능) ② Meta 오픈 웨이트(Llama 4 Scout, Muse Glimmer, Llama 3.x)를 로컬 추론 대상으로 사용.

---

## 2. 오픈소스 저장소 조사 (✅ 2026-10-26 `git clone`으로 직접 확인)

| 저장소 | 마지막 커밋 | 7/1 이후 커밋 수 | 라이선스 | 판단 |
|---|---|---|---|---|
| [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | 2026-10-26 | 매우 활발 | MIT | **기본 엔진으로 채택** |
| [SJTU-IPADS/PowerInfer](https://github.com/SJTU-IPADS/PowerInfer) | 2026-05-11 | 0 | MIT | 아이디어만 참고 |
| [ollama/ollama](https://github.com/ollama/ollama) | 2026-10-25 | 446 | MIT | 전체 통합 대신 llama.cpp 직접 통합 |
| [turboderp-org/exllamav2](https://github.com/turboderp-org/exllamav2) | 2026-03-04 | 0 | MIT | 사실상 중단됨 → v3로 대체 |
| [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) | 2026-10-25 | 1092 | MIT | NVIDIA CUDA 전용. 범용 USB 제품에는 부적합 |
| [ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm) | 2026-10-24 | 151 | MIT | Apple Silicon 전용 2차 백엔드 후보 |
| [mlc-ai/mlc-llm](https://github.com/mlc-ai/mlc-llm) | 2026-10-07 | 19 | Apache-2.0 | 모델별 컴파일 필요. 해커톤 기간에는 제외 |
| [ikawrakow/ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) | 2026-10-26 | 451 | MIT | CPU·MoE 하이브리드 최적화 포크. 벤치 비교 후보 |
| [kvcache-ai/ktransformers](https://github.com/kvcache-ai/ktransformers) | 2026-10-19 | 64 | Apache-2.0 | CPU-GPU 이기종 MoE 연구. 서버급 지향 |
| [Mozilla-Ocho/llamafile](https://github.com/Mozilla-Ocho/llamafile) | 2026-10-08 | 28 | Apache-2.0 | **단일 실행 파일 배포의 참고 모델** |
| [mostlygeek/llama-swap](https://github.com/mostlygeek/llama-swap) | 2026-10-25 | 151 | (LICENSE 파일명 미확인) | 모델 교체 프록시. 참고용 |

### 2.1 llama.cpp: 직접 빌드·실행해 확인
- **아키텍처**(`src/llama-arch.cpp`): `llama4`, `gpt-oss`, `qwen3moe`, `deepseek2`, `glm4moe`, `muse-glimmer`, `muse-spark` 등이 있습니다.
- **백엔드**(`ggml/src`): CPU, CUDA, HIP(ROCm), Metal, Vulkan, SYCL, OpenCL, CANN, MUSA, WebGPU, Hexagon, OpenVINO, RPC 등
- **MoE 하이브리드 옵션**: `-cmoe`(전문가 전체를 CPU에), `-ncmoe N`(앞쪽 N개 레이어의 전문가를 CPU에), `-ot`(텐서 패턴별 배치), `--moe-cache-mib`(CPU 쪽 전문가의 GPU 캐시)
- **자동 메모리 맞춤**: `--fit`이 **기본으로 켜져 있음**(`common.h`: `fit_params = true`). 지정하지 않은 파라미터를 장치의 여유 메모리에 맞춰 줄입니다. → "자동 메모리 맞춤"만으로는 차별점이 될 수 없습니다. PocketLLM은 **로드·다운로드 전에 판정하고, 설명하고, 실측으로 검증하는 것**을 차별점으로 삼습니다.
- **플래그 변경을 실제로 겪음**: `--no-mmap`이 없어지고 `-lm/--load-mode auto|none|mmap|mlock|mmap+mlock|dio`로 바뀌었고, `-fa`는 `on|off|auto` 값을 받습니다. → **USB 제품은 llama.cpp 커밋을 고정해야 합니다**(`scripts/build_llamacpp.sh`, 커밋 `caa8191`).
- **메모리 부족 시 동작**: 기본 mmap 모드에서는 RAM보다 큰 모델도 로드됩니다. 이때 OS가 저장 장치에서 페이지를 다시 읽으므로 속도는 저장 장치 읽기 속도에 묶입니다. USB를 뽑으면 런타임이 죽습니다(mmap 페이지 폴트).
- **USB 파일 구조**: `bin/<os>-<arch>/`(공유 라이브러리 `libggml*.so`, `libllama*.so`가 실행 파일과 같은 폴더에 있어야 함), `models/*.gguf`(분할 GGUF는 `-00001-of-0000N` 모두 필요). 드라이브는 **exFAT**(FAT32는 4GB 제한)이어야 합니다.

### 2.2 PowerInfer
- 최근 활동: 2026-05 이후 커밋 없음. README에 **"Tiiny AI Pocket Lab을 출시했다(GPT-OSS-120B int4, 20 tok/s)"**는 공지가 있고, `Tiiny-AI/PowerInfer`로 클론하라는 안내도 있습니다. → **PowerInfer 팀이 Tiiny AI의 기술 기반**입니다. 경쟁사 분석에서 중요한 사실입니다.
- 핵심 아이디어: 자주 활성화되는 hot neuron은 GPU에, 나머지 cold neuron은 CPU에 두는 방식. ReLU 계열의 희소화 모델(ProSparse, TurboSparse, Falcon-ReLU)을 전제로 합니다(README에서 확인). 일반 Llama 4나 gpt-oss에 바로 적용하기는 어렵습니다.
- **판단:** 그대로 채택하지 않습니다. 대신 MoE 모델에서는 "전문가 단위 hot/cold" 배치가 같은 아이디어입니다. llama.cpp의 `-ncmoe`와 `--moe-cache-mib`로 근사할 수 있고, 장기적으로는 **실제 라우팅 통계를 기반으로 전문가를 GPU에 배치**하는 기능이 PocketLLM의 연구 과제 후보입니다.

### 2.3 Ollama
- 모델 디렉터리는 `OLLAMA_MODELS` 환경 변수로 바꿀 수 있고, 호스트는 `OLLAMA_HOST`로 지정합니다(`envconfig/config.go`에서 확인). → USB를 모델 저장소로 쓰는 것 자체는 가능합니다.
- 다만 서비스 설치가 필요하고, 자체 모델 레지스트리와 블롭 구조를 쓰며, 런처 설정 노출이 제한적입니다. **설치 없이 USB에서 바로 실행하고, 설정을 명시적으로 탐색**하려는 PocketLLM의 목표에는 llama.cpp 직접 통합이 맞습니다.
- 참고: 한 2차 출처는 DGX Spark에서 같은 모델로 Ollama 약 11.7 tok/s, 튜닝한 llama.cpp 약 38.5 tok/s를 보고했습니다 🔎. "설정이 성능을 좌우한다"는 근거 중 하나지만 독립 검증은 필요합니다.

### 2.4 ExLlamaV2 → V3
- v2는 2026-03 이후 커밋이 없고, 후속인 v3가 활발합니다. v3는 PyTorch와 CUDA 12.4 이상이 필요하고 Windows에서는 triton-windows가 필요합니다(README에서 확인).
- **판단:** NVIDIA GPU 전용이고 설치가 무거워 "아무 노트북에 꽂아 쓰는" 제품에는 맞지 않습니다. 장기적으로 NVIDIA 전용 고성능 모드 후보로만 둡니다.

### 2.5 그 밖의 프로젝트
- **MLX (mlx-lm)**: Apple Silicon 전용이고 활발합니다. Mac 사용자 비중이 크면 2차 백엔드로 검토합니다. 해커톤 기간에는 llama.cpp Metal로 충분합니다.
- **llamafile**: "단일 파일로 여러 OS에서 설치 없이 실행"하는 방식입니다. **USB 휴대성의 참고 모델**입니다. Windows는 4GB 넘는 실행 파일을 못 돌리므로 가중치를 외부 GGUF로 분리해야 한다고 README에 적혀 있습니다. → PocketLLM 런처를 단일 바이너리로 만들 때의 설계 근거로 씁니다.
- **ik_llama.cpp**: CPU와 하이브리드 MoE 성능을 주장하는 포크입니다. CUDA(Turing 이상)와 CPU 외 백엔드는 지원하지 않는다고 명시합니다. 노트북이 NVIDIA라면 Scout 실험에서 비교군으로 씁니다.
- **ktransformers**: CPU-GPU 이기종 MoE 추론과 미세 조정 연구입니다. 최근 뉴스가 DeepSeek-V4-Flash, GLM-5.3-flash 등 서버·워크스테이션 규모입니다. 아이디어(전문가를 CPU에 두는 레이아웃 최적화)만 참고합니다.

---

## 3. 경쟁 제품 비교 (🔎 모두 2차 출처. 공식 사이트는 ❌)

| 항목 | Tiiny AI Pocket Lab | ASUS UGen300 | Mobilint MLD-R1 | NVIDIA DGX Spark | **PocketLLM (이번 프로토타입)** |
|---|---|---|---|---|---|
| 형태 | 포켓형 **독립 컴퓨터** | USB 가속기 스틱 | USB 가속기 동글 | 데스크톱 미니 PC | USB 드라이브 + 호스트 소프트웨어 |
| 연결 | 미확인 | USB 3.1 Gen2 (10Gbps) | USB 5Gbps | 네트워크·데스크톱 단독 사용 | USB(파일 저장소) |
| 메모리 | 80GB LPDDR5X + 1TB SSD | 8GB LPDDR4 | 4/8GB LPDDR4X | 128GB 통합 메모리(약 273GB/s) | **호스트 RAM/VRAM** |
| 연산 칩 | 미확인(190 TOPS, 65W라는 포럼 글) | Hailo-10H, INT4 40 TOPS | REGULUS SoC, INT8 10 TOPS | GB10 | 호스트 CPU/GPU |
| 전력 | 65W(미확인) | 약 2.5W | 약 3W | 미확인 | 호스트 |
| 실행 모델 규모 | GPT-OSS-120B int4(제조사 주장) | 공개된 LLM 수치 없음(8GB 한계) | 1~3B SLM(제조사) | 120B급 | 호스트 사양에 따라 다름(판정 제공) |
| 속도 | 20 tok/s(제조사 주장, **독립 벤치 없음**) | 미공개 | 미공개 | gpt-oss-120b 약 35~60 tok/s(출처마다 다름) | 실측해 표시 |
| 가격 | $1,299~$1,399(킥스타터) | $299.99(단일 출처) | 미확인 | $3,999 출시 → $4,699~$6,950 보도(상충) | 소프트웨어 + USB 가격 |
| OS 호환 | 미확인 | Win/Linux/Android | Win/Linux(x86/Arm) | DGX OS(Linux) | Win/macOS/Linux(설계상. 검증은 Linux만) |
| 오프라인 | 예(주장) | 예 | 예 | 예 | 예(설정 도우미의 Meta API만 온라인) |
| 출시 상태 | 배송 예정 2026-08이었으나 실제 배송 여부 미확인 | 판매 중으로 보임 | 2026-08-07 상용 출시 | 판매 중 | 프로토타입 |

### 3.1 비교에서 나온 핵심 결론
1. **USB 가속기(UGen300, MLD-R1)의 메모리는 4~8GB**뿐이라 1~8B급 모델만 다룹니다. 100B급을 실제로 돌리는 제품(Tiiny, DGX Spark)은 **64~128GB 고대역폭 메모리를 갖춘 별도 컴퓨터**입니다. "USB 스틱 형태로 100B"를 하려면 가속기가 아니라 **작은 컴퓨터**를 만들어야 합니다.
2. 장치 안에 가중치가 들어 있으면 **USB 링크로는 토큰과 프롬프트만 오가므로** USB 대역폭은 병목이 아닙니다. 반대로 가중치를 USB로 계속 전송하는 구조(=PocketLLM 프로토타입의 페이징 모드)는 USB 대역폭에 묶입니다. 10Gbps 이론치 1.25GB/s로 Scout Q4의 토큰당 활성 가중치(약 10GB)를 읽으면 **약 0.1 tok/s**입니다.
3. 차별화할 수 있는 영역은 다음과 같습니다.
   - **하드웨어에 종속되지 않는 소프트웨어 계층**: 기존 장치는 각자 자기 하드웨어 전용 SDK를 씁니다. PocketLLM은 "어느 컴퓨터, 어느 가속 장치든" 판정하고, 배치하고, 실측하는 계층입니다.
   - **정직한 실행 가능성 판정**: 제조사 주장(20 tok/s)과 실측 사이의 간극이 시장의 불신 요인입니다. PocketLLM은 다운로드 전에 메모리 내역, 병목, 속도 상한을 보여 주고, 로드 후에는 실측값을 보여 줍니다.
   - **장기적으로 장치 + 호스트 하이브리드**: 장치 메모리에는 전문가를 두고 호스트 GPU에는 attention을 두는 분할은 기존 제품에 없는 방향입니다. `docs/ROADMAP_HW_KO.md` 참고.

---

## 4. 11가지 핵심 질문에 대한 현재 답

1. **사양 없이 100B 실행 가능성을 빨리 평가하는 방법:** `docs/feasibility_matrix.md`(사양 등급별 플래너 추정)를 먼저 봅니다. 노트북에서 `scripts/check_laptop.sh`를 한 번 실행하면 실제 판정이 나옵니다(1분 이내).
2. **RAM/VRAM별 최대 모델:** 4-bit 기준 대략 `(가용 RAM + VRAM − 약 4GB) ÷ 0.61 GB/B`입니다. 예를 들어 16GB 노트북은 약 20B(실사용 여유를 두면 8~14B가 현실적), 32GB는 약 40B(dense 기준), 64GB + 16GB GPU는 Scout Q4(67GB)가 MoE 분할로 겨우 들어갑니다. 상세는 표를 참고하세요.
3. **Scout를 llama.cpp에서 실행할 수 있나:** 아키텍처 지원은 ✅입니다. 실제 실행은 RAM과 VRAM 합이 대략 70GB 이상(Q4)이거나 약 30GB 이상(IQ2_XXS)이어야 실용적입니다.
4. **4/3/2-bit 트레이드오프:** llama.cpp 표에서 Llama-3-8B 기준 PPL 증가는 Q4_K_M +0.18, Q3_K_M +0.66, Q2_K +3.52입니다. 메모리는 bpw에 비례합니다(4.9 / 4.0 / 3.2). 큰 모델에서는 열화가 덜하다는 통념이 있지만 **Scout에서 직접 측정해야 합니다**(테스트 계획 E5).
5. **CPU 오프로딩 시 메모리와 속도:** 생성 속도 상한 ≈ 메모리 대역폭 ÷ 토큰당 활성 바이트입니다. Scout Q4는 토큰당 약 10.4GB를 읽으므로 60GB/s RAM에서 ≤5.8 tok/s, 페이징이 끼면 저장 장치 속도에 묶여 1 tok/s 미만입니다.
6. **PowerInfer 실용성:** 낮습니다. 유지보수가 멈췄고 희소화 모델을 전제로 합니다. 아이디어만 MoE 전문가 배치에 적용합니다.
7. **USB와 내부 SSD 비교:** 모델이 RAM에 다 들어가면 차이는 **로딩 시간뿐**입니다(`-lm none`으로 RAM에 올리면 이후 동일). 들어가지 않으면 속도가 저장 장치 읽기 속도에 비례합니다. USB 3.2 Gen2 SSD는 약 1GB/s, NVMe 내장 SSD는 3~7GB/s이므로 수 배 차이가 납니다. **`scripts/usb_vs_ssd.sh`로 실측해야 합니다.**
8. **가장 명확한 차별점:** "꽂으면 이 컴퓨터에서 무엇이 돌아가는지 다운로드 전에 정직하게 알려 주고, 실측으로 최적 설정을 찾아 바로 쓰게 해 주는 것"입니다. 하드웨어 경쟁(TOPS, 메모리 크기)은 하지 않습니다.
9. **Meta 기술:** Meta Model API(Muse Spark)는 설정 도우미로, Meta 오픈 웨이트(Llama 4 Scout, Muse Glimmer, Llama 3.x)는 로컬 실행 대상으로 씁니다. 필수 여부는 공식 규정을 확인하세요.
10. **10일 MVP:** 하드웨어 탐지, 판정표, 실제 소형 Meta 모델 로컬 채팅, 실측 튜닝, 오프라인 데모, Meta API 도우미입니다. 코드는 대부분 이미 동작합니다.
11. **상용화의 핵심 가치:** 단기에는 **소프트웨어 경험**입니다(하드웨어에 종속되지 않음). 장기 하드웨어는 "연산 성능"보다 **메모리 용량 × 대역폭 ÷ 가격**이 핵심 지표입니다. USB 링크 속도는 중요하지 않습니다(가중치가 장치에 있으면).

## 출처 (검색으로 확인)
- 해커톤: [Cerebral Valley 이벤트](https://cerebralvalley.ai/e/meta-ai-developer-hackathon), [vibecoderslife](https://vibecoderslife.com/post/meta-ai-developer-hackathon-2026), [reconscribe](https://reconscribe.com/meta-global-ai-developer-hackathon-2026/), [resourify](https://resourify.com/resources/meta-api)
- Meta Model API: [dev.meta.ai/docs](https://dev.meta.ai/docs), [Muse Spark 블로그](https://developer.meta.com/ai/resources/blog/build-with-muse-spark/)
- Muse Glimmer: [Artificial Analysis](https://artificialanalysis.ai/articles/muse-glimmer), [VKTR](https://www.vktr.com/ai-platforms/meta-releases-muse-glimmer/), [The Register](https://www.theregister.com/a/5285666)
- Muse Spark Open: [Unsloth 문서](https://unsloth.ai/docs/models/muse-spark-open)
- Llama 4 Scout GGUF: [Unsloth 가이드](https://unsloth.ai/docs/models/tutorials/llama-4-how-to-run-and-fine-tune), [markaicode 벤치](https://markaicode.com/benchmarks/llamacpp-llama-33-h100-throughput-benchmark/)
- Tiiny: [Yahoo Finance](https://finance.yahoo.com/news/tiiny-ai-launches-pocket-size-083700323.html), [cechtechreviews](https://cechtechreviews.com/reviews/tiiny-ai-pocket-lab-review/), [LTT 포럼](https://linustechtips.com/topic/1630089-tiiny-ai-pocket-lab-a-65w-190tops-80gb-poket-pc/)
- ASUS UGen300: [ASUS](https://www.asus.com/motherboards-components/ai-accelerator/ugen/ugen300-usb-8g/), [electronics-lab](https://www.electronics-lab.com/asus-ugen300-%CE%B2rings-hailo-10h-%CE%B5dge-ai-%CE%B1cceleration-%CE%BFver-usb-c/), [rswebsols(가격)](https://www.rswebsols.com/news/299-99-asus-ugen300-usb-ai-accelerator-features-40-tops-hailo-10h-chip-and-8gb-lpddr4/)
- Mobilint MLD-R1: [letsdatascience](https://letsdatascience.com/news/mobilint-launches-mld-r1-usb-edge-ai-accelerator-477ab5e3), [보안뉴스](https://m.boannews.com/html/detail.html?idx=145060)
- DGX Spark: [computingforgeeks](https://computingforgeeks.com/dgx-spark-vs-strix-halo-vs-mac-studio/), [openclawdc](https://openclawdc.com/blog/is-the-dgx-spark-worth-it/), [implicator](https://www.implicator.ai/nvidia-ships-dgx-spark-at-4k-five-months-late-filling-a-prosumer-ai-gap-with-a-bandwidth-trade-off/)

# 대형 모델 로컬 실행 프로젝트 전수 조사와 원리 정리 (2026-10-27)

**검증 방법:** 아래 저장소는 모두 2026-10-26~27에 `git clone`(메타데이터만)으로 존재와 최근 커밋, 라이선스를 직접 확인했습니다. "7/1 이후"는 2026-07-01 이후의 커밋 수입니다. 성능 수치는 **각 프로젝트의 자체 주장**이며 독립 검증이 아닙니다(✳ 표시).

prima.cpp(`Lizonghang/prima.cpp`)는 논문(ICLR 2026)은 확인됐지만, 저장소가 현재 공개 상태가 아닙니다(클론할 때 인증 요구). 그래서 아이디어만 반영했습니다.

---

## A. 원리별 지도: "어떻게 수십~수백 B를 작은 기기에서 돌리나"

메모리가 부족할 때 쓸 수 있는 방법은 결국 일곱 가지로 수렴합니다.

| # | 원리 | 대표 프로젝트 | 핵심 메커니즘 | PocketLLM 반영 |
|---|---|---|---|---|
| 1 | **저비트 양자화** | llama.cpp(k/i-quants), BitNet(1.58bit), T-MAC(LUT 커널), GPTQModel, AQLM, llm-awq | 가중치 바이트 수 자체를 줄임. 생성 속도는 읽는 바이트에 비례 | 플래너가 양자화별 크기와 속도 상한을 계산하고 추천 |
| 2 | **MoE 희소 활성화** | 모든 MoE 모델(Llama 4 Scout, gpt-oss, Qwen3-30B-A3B) | 토큰마다 전체의 3~15%만 계산 | **MoE 우선 카탈로그**(같은 품질에서 열이 적음) |
| 3 | **전문가 오프로딩·캐시** | ktransformers, ik_llama.cpp, fastllm, MoE-Infinity, Fiddler, mixtral-offloading, llama.cpp `-ncmoe` | 자주 쓰는 전문가는 빠른 메모리에, 나머지는 느린 계층에. LRU·빈도 캐시 | `-ncmoe` 자동 계산 + **hot expert pinning** |
| 4 | **SSD/플래시 스트리밍** | TinyGiant, AirLLM, Strata, (논문) LLM in a Flash, PowerInfer-2, EDGEO, SSD-LLaMA | 메모리에 안 들어가는 가중치를 저장 장치에서 필요할 때만 읽음 | **ggml-pocket 엔진**: 전문가 단위 비동기 선읽기 + 다음 레이어 예측 |
| 5 | **활성화 희소성(hot/cold 뉴런)** | PowerInfer / Tiiny, PowerInfer-2 | 자주 켜지는 뉴런은 GPU, 나머지는 CPU | MoE 전문가 단위로 일반화(pinning) |
| 6 | **여러 기기로 분산** | exo, distributed-llama, prima.cpp, cake, petals, llama.cpp RPC, mlx distributed | 레이어(파이프라인) 또는 텐서 단위로 여러 기기의 메모리를 합침 | **멀티 드라이브 스트라이핑 + 멀티 노드(RPC) 클러스터** |
| 7 | **디코딩 가속** | EAGLE, Medusa, LookaheadDecoding, prompt-lookup-decoding, llama.cpp `--spec-*` | 작은 초안 모델이나 n-gram으로 여러 토큰을 미리 추측한 뒤 한 번에 검증 | 플래너 확장 예정(초안 모델 자동 페어링) |

**중요한 결론:** 일반 노트북에서 30B 이상을 "쓸 만한 속도"로 돌린 사례는 거의 모두 **② MoE + ③/④ 전문가 스트리밍** 조합입니다. 예를 들어 TinyGiant는 M1 16GB에서 Qwen3-30B-A3B를 llama.cpp CPU 대비 약 2.5배 빠르게 돌렸다고 보고합니다✳. 그래서 PocketLLM의 핵심 엔진도 이 조합으로 만들었습니다.

---

## B. 프로젝트별 정리

### B1. 단일 기기, 메모리 계층 활용 (스트리밍과 오프로딩)
| 프로젝트 | 마지막 커밋 / 7월 이후 | 라이선스 | 원리 | 주장✳ | 평가 |
|---|---|---|---|---|---|
| [jerryjokesalot/tinygiant](https://github.com/jerryjokesalot/tinygiant) | 10-24 / 33 | MIT | 전문가를 연속된 SSD 영역으로 재배치하고, 짧은 실제 텍스트로 보정해 자주 쓰는 전문가를 mlock | M1 16GB, Qwen3-30B-A3B Q4_K_M: llama.cpp CPU 1.06 → 2.68 tok/s | **PocketLLM과 가장 가까움.** 같은 아이디어를 llama.cpp 안에서(전체 커널 재사용) 구현 |
| [lyogavin/airllm](https://github.com/lyogavin/airllm) | 10-26 / 137 | Apache-2.0 | 레이어를 하나씩 디스크에서 읽어 GPU에서 계산한 뒤 버림. 최근에는 전문가 단위 스트리밍 | 70B를 4GB GPU에서, 거대 MoE를 수 GB VRAM에서 | 메모리는 최소지만 응답에 분 단위가 걸린다는 리뷰가 있음. 대화형보다 배치 처리용 |
| [Niko1221/strata](https://github.com/Niko1221/strata) | 10-25 / 4196 | MIT | 메모리에 안 들어가는 부분은 SSD에서 읽는 데스크톱 앱 | 125B MoE를 12GB GPU + 64GB RAM에서 | 매우 활발함. UX 경쟁 상대 |
| [kvcache-ai/ktransformers](https://github.com/kvcache-ai/ktransformers) | 10-19 / 64 | Apache-2.0 | CPU(AMX/AVX512)에 전문가, GPU에 attention. 이기종 MoE | 서버급 DeepSeek를 단일 GPU + 대용량 RAM에서 | 워크스테이션용 |
| [ikawrakow/ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) | 10-26 / 451 | MIT | llama.cpp 포크. CPU·하이브리드 MoE 커널 최적화 | — | CPU와 CUDA만 완전 지원 |
| [ztxz16/fastllm](https://github.com/ztxz16/fastllm) | 10-27 / 1371 | Apache-2.0 | C++ 엔진. GPU, CPU, NUMA, **디스크 혼합 추론**, `--auto`로 전문가 캐시 자동 계획 | — | 자동 계획 개념이 PocketLLM 플래너와 같은 방향 |
| [EfficientMoE/MoE-Infinity](https://github.com/EfficientMoE/MoE-Infinity) | 10-26 / 162 | Apache-2.0 | 활성화 인식 전문가 캐시(GPU ← RAM ← SSD) | — | HF 기반, GPU 중심 |
| [giannisanni/pulsar](https://github.com/giannisanni/pulsar) | 09-01 | MIT | Rust + CUDA. NVMe 전문가 스트리밍, LFU 호스트 캐시, io_uring QD32, 라우터 lookahead, **전문가 Q2_K + 나머지 Q8_0 양자화 레시피**, 실행 간 `.warm` 인기 전문가 기록, 분할 GGUF를 여러 SSD에 심볼릭 링크로 분산 | 2×16GB GPU에서 Qwen3-235B 5.3 tok/s, 첫 실행 hotlist로 19.2 → 25.1 tok/s | 캐시·lookahead는 엔진 v2로 실측(느린 USB에서는 손해). **양자화 레시피는 `pocketllm requant`로 가져와 ×2.3~3.3 (R9)**. warm 기록은 `POCKET_MOE_WARM_MIB`. 다중 SSD 분산은 `pocketllm stripe`와 같은 방식 |
| [SharpAI/SwiftLM](https://github.com/SharpAI/SwiftLM) | 10-27 | MIT | Swift + MLX. 애플 실리콘 `--stream-experts`(동시 pread QD24, 프로젝션 묶음 읽기, 런타임 top-k), TurboQuant KV 압축, MTP 추측 디코딩 | 126GB 모델 SSD 스트리밍: 40K 문맥에서 KV 압축으로 0.32 → 4.16 tok/s | **맥 백엔드로 연결**. top-k는 `POCKET_MOE_TOPK`(R8). KV 압축 교훈은 플래너의 자동 q8_0 KV로 반영 |
| [jhammant/expertflow](https://github.com/jhammant/expertflow) | 03-29 | README에 MIT 배지(LICENSE 파일 없음) | Rust 스케줄러 설계. 온도(최근성×빈도) 기반 핫 전문가 유지와 `MADV_DONTNEED` 축출, 라우터 lookahead, 핫·콜드 전문가 혼합 정밀도, KV 크기에 따른 전문가 예산 | README의 속도는 **예상치**("projected targets")이고 실측은 미공개 | 아이디어만 참고(혼합 정밀도 → requant, KV 인식 예산 → 플래너) |
| [techjarves/USB-Uncensored-LLM](https://github.com/techjarves/USB-Uncensored-LLM) | — | 라이선스 없음 | USB에 휴대용 런타임 + 모델을 담아 꽂아서 실행 | — | 코드는 가져오지 않고 아이디어만: 드라이브에 휴대용 Python(`scripts/fetch_python.py`, python-build-standalone, SHA-256 검증) |
| [efeslab/fiddler](https://github.com/efeslab/fiddler) | 2024-04 / 0 | Apache-2.0 | 전문가 가중치를 옮기지 않고 CPU에서 계산(활성화 값만 이동) | Mixtral 8x7B 비양자화 24GB GPU에서 3 tok/s 이상 | 원리는 llama.cpp `-ncmoe`가 이미 흡수 |
| [dvmazur/mixtral-offloading](https://github.com/dvmazur/mixtral-offloading) | 2024-01 / 0 | MIT | 전문가 LRU 캐시 + 혼합 양자화(HQQ) + 투기적 전문가 로딩 | — | 투기적 로딩 → PocketLLM의 다음 레이어 예측 |
| [Tiiny-AI/PowerInfer](https://github.com/Tiiny-AI/PowerInfer) (=SJTU-IPADS) | 2026-05 / 0 | MIT | hot/cold 뉴런 분할. 희소화 모델이 필요 | Tiiny Pocket Lab: GPT-OSS-120B 20 tok/s | 아이디어만 반영 |
| [FMInference/FlexGen](https://github.com/FMInference/FlexGen) | 2024-10 / 0 | Apache-2.0 | GPU, CPU, 디스크 사이의 최적 오프로딩 스케줄. 대량 배치 처리량 위주 | — | 대화형에는 맞지 않음 |
| [huggingface/accelerate](https://github.com/huggingface/accelerate), [deepspeedai/DeepSpeed](https://github.com/deepspeedai/DeepSpeed) | 활발 | Apache-2.0 | `device_map="auto"` 디스크 오프로드, ZeRO-Inference | — | PyTorch 의존. USB 휴대 실행에는 무거움 |

### B2. 여러 기기로 분산 ("USB 여러 개"의 연산판)
| 프로젝트 | 마지막 / 7월 이후 | 라이선스 | 원리 | 비고 |
|---|---|---|---|---|
| [exo-explore/exo](https://github.com/exo-explore/exo) | 10-13 / 92 | Apache-2.0 | 자동 기기 탐색, 토폴로지를 고려한 자동 분할, MLX 텐서 병렬, Thunderbolt RDMA | Mac 클러스터 중심. 2대에서 1.8배, 4대에서 3.2배(텐서 병렬)✳ |
| [b4rtaz/distributed-llama](https://github.com/b4rtaz/distributed-llama) | 10-25 / 4 | MIT | 이더넷 텐서 병렬. "기기가 많을수록 빠르다" | Llama 3.3 70B를 Mac mini 4대에서✳ |
| prima.cpp (논문, 저장소 비공개) | — | — | piped-ring 병렬 + 디스크 선읽기 + 이기종 스케줄러(Halda) | 70B를 가정용 기기 4대에서 674 ms/token✳ |
| [evilsocket/cake](https://github.com/evilsocket/cake) | 2026-04 / 0 | FAIR(비상업) | iOS·Android·Mac·PC 이기종 샤딩 | **라이선스가 상업 제품에 부적합** |
| [bigscience-workshop/petals](https://github.com/bigscience-workshop/petals) | 2024-08 / 0 | MIT | 인터넷 P2P(BitTorrent식) 분산 | **폐쇄망 요구와 정반대**(외부 노드에 데이터 노출) |
| llama.cpp RPC(`ggml-rpc`) | 활발 | MIT | 원격 백엔드에 레이어 할당 | PocketLLM 클러스터 모드의 기반 |

### B3. 디코딩 가속
| 프로젝트 | 상태 | 원리 |
|---|---|---|
| [SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE) | 2026-02 / 0 | 특징 수준 초안 헤드로 약 3배✳(13B) |
| [FasterDecoding/Medusa](https://github.com/FasterDecoding/Medusa) | 2024 / 0 | 다중 디코딩 헤드 |
| [hao-ai-lab/LookaheadDecoding](https://github.com/hao-ai-lab/LookaheadDecoding) | 2025 / 0 | 초안 모델 없이 n-gram 병렬 추측 |
| [apoorvumang/prompt-lookup-decoding](https://github.com/apoorvumang/prompt-lookup-decoding) | 2024 / 0 | 프롬프트 안의 n-gram 재사용(요약과 RAG에 효과적) |

### B4. 저비트와 커널
| 프로젝트 | 상태 | 원리 |
|---|---|---|
| [microsoft/BitNet](https://github.com/microsoft/BitNet) | 07-27 / 14 | 1.58-bit 전용 커널(I2_S). 전용으로 학습한 모델이 필요 |
| [microsoft/T-MAC](https://github.com/microsoft/T-MAC) | 2025-06 / 0 | 저비트 행렬곱을 룩업 테이블로 처리(CPU, NPU) |
| [ModelCloud/GPTQModel](https://github.com/ModelCloud/GPTQModel), [Vahe1994/AQLM](https://github.com/Vahe1994/AQLM), [mit-han-lab/llm-awq](https://github.com/mit-han-lab/llm-awq) | 활발 / 정지 / 정지 | GPU용 2~4bit 양자화 |

### B5. 범용 런타임과 앱 (UX 비교 대상)
| 프로젝트 | 마지막 / 7월 이후 | 라이선스 | 비고 |
|---|---|---|---|
| [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) / [ggml](https://github.com/ggml-org/ggml) | 매우 활발 | MIT | **PocketLLM의 기반** |
| [ollama/ollama](https://github.com/ollama/ollama) | 10-25 / 446 | MIT | 설치형 |
| [LostRuins/koboldcpp](https://github.com/LostRuins/koboldcpp) | 10-18 / 2154 | AGPL-3.0 | 단일 실행 파일 |
| [mudler/LocalAI](https://github.com/mudler/LocalAI) | 10-26 / 1878 | MIT | OpenAI 호환 로컬 서버 |
| [janhq/jan](https://github.com/janhq/jan) | 10-26 / 1150 | 자체 라이선스 | 데스크톱 앱 |
| [nomic-ai/gpt4all](https://github.com/nomic-ai/gpt4all) | 2025-05 / 0 | — | 정지 |
| [Mozilla-Ocho/llamafile](https://github.com/Mozilla-Ocho/llamafile) | 10-08 / 28 | Apache-2.0 | **단일 파일 다중 OS** → USB 런처 설계의 근거 |
| [EricLBuehler/mistral.rs](https://github.com/EricLBuehler/mistral.rs), [huggingface/candle](https://github.com/huggingface/candle) | 활발 | MIT / MIT+Apache | Rust 엔진 |
| [ml-explore/mlx](https://github.com/ml-explore/mlx), [lmstudio-ai/mlx-engine](https://github.com/lmstudio-ai/mlx-engine) | 활발 | MIT | Apple 전용 |
| [gpustack/gpustack](https://github.com/gpustack/gpustack), [xorbitsai/inference](https://github.com/xorbitsai/inference) | 활발 | Apache-2.0 | 사내 GPU 클러스터 관리 |
| [vllm-project/vllm](https://github.com/vllm-project/vllm), [sgl-project/sglang](https://github.com/sgl-project/sglang), [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM), [kvcache-ai/Mooncake](https://github.com/kvcache-ai/Mooncake) | 매우 활발 | Apache 등 | 데이터센터 서빙용. 노트북 대상 아님 |
| [NexaAI/nexa-sdk](https://github.com/NexaAI/nexa-sdk) | 10-26 / 862 | BSD-3 | Qualcomm 전용 |
| [tinygrad/tinygrad](https://github.com/tinygrad/tinygrad), [zml/zml](https://github.com/zml/zml) | 활발 | MIT, Apache | 범용 프레임워크 |

---

## C. 원리를 "USB 하나에 총합"하면 무엇이 가능한가

### C1. USB 저장 장치가 실제로 더해 주는 것 (물리적으로 정직하게)
- 연산 능력(TOPS)은 **더하지 않습니다.** 열과 전력도 노트북의 몫입니다.
- 더해 주는 것은 두 가지입니다.
  1. **용량:** 내장 SSD를 차지하지 않고 수십~수백 GB 모델을 들고 다닙니다.
  2. **읽기 대역폭:** 모델이 RAM보다 클 때 생성 속도는 "가중치를 저장 장치에서 다시 읽는 속도"에 묶입니다. USB4 NVMe 외장 SSD는 약 3GB/s이고, **드라이브를 여러 개 꽂으면 대역폭이 더해집니다**(모델을 드라이브별로 분할 + 동시 선읽기). 이것이 "USB를 여러 개 꽂을수록 좋아진다"의 정직한 메커니즘입니다.
- 그리고 소프트웨어(ggml-pocket 엔진, 플래너, 튜너)가 그 대역폭을 실제로 끌어씁니다. 기본 llama.cpp mmap은 컴퓨트 스레드가 4~16KB 페이지 폴트를 순차로 처리해서 고속 저장 장치를 놀립니다.

### C2. 맥북 에어(팬리스)가 7B dense에서 뜨거워지는 문제
- 발열은 **토큰당 연산량 × 토큰 수**에 비례합니다. Qwen 7B dense는 토큰마다 7B 전체를 계산합니다.
- **MoE로 바꾸면** Qwen3-30B-A3B(활성 3B), gpt-oss-20b(활성 3.6B), Llama 4 Scout(활성 17B)는 토큰당 연산이 활성 파라미터에 비례합니다. 그래서 **30B급 MoE가 7B dense보다 토큰당 연산이 적습니다**(3B 대 7B). 단, 전체 가중치는 저장 장치에 있어야 하므로 위의 스트리밍 엔진이 필요합니다.
- 추가로 PocketLLM은 **열 프로파일**(cool / balanced / max)을 둡니다. 스레드 수를 제한하고(성능 코어만 쓰거나 일부만 사용), macOS 열 압력을 감지하면 낮은 프로파일로 내려갑니다.

### C3. "USB 여러 개 = 연산 여러 개"로 가려면
연산이 있는 장치(SBC, 미니 PC, 다른 노트북)를 USB-C 네트워킹이나 LAN으로 연결합니다. 그러면 llama.cpp RPC나 exo, distributed-llama식으로 **메모리를 합칠 수 있습니다.** PocketLLM 클러스터 모드는 llama.cpp RPC 기반이고, 노드별 메모리에 비례해 레이어를 나눕니다. 이것이 장기 하드웨어 로드맵(H1)과 같은 구조입니다.

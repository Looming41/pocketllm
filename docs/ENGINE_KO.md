# PocketLLM 전문가 스트리밍 엔진 (ggml-pocket)

## 1. 문제
MoE 모델(Llama 4 Scout, Qwen3-30B-A3B, gpt-oss 등)은 토큰마다 일부 전문가만 씁니다. 그래서 **모델이 RAM보다 커도** "선택된 전문가만 저장 장치에서 읽으면" 돌릴 수 있어야 합니다. 그런데 기본 llama.cpp(mmap)에서는 그렇게 되지 않습니다.

1. **동기 페이지 폴트:** 계산 스레드가 가중치를 만지는 순간 4~16KB 단위로 순차 대기합니다. 그래서 고속 SSD나 여러 드라이브의 병렬성을 쓰지 못합니다.
2. **읽기 증폭:** Linux에서 llama.cpp는 파일에 `POSIX_FADV_SEQUENTIAL`을 주고, 커널은 폴트 주변을 크게 미리 읽습니다(read-around). MoE에서는 그 주변이 **쓰지 않을 이웃 전문가**입니다. 이 실험에서는 필요한 양의 약 **3배**를 읽었습니다(아래 표의 GB 열).

## 2. 해결: 라우터가 전문가를 고른 바로 그 순간 읽는다
llama.cpp CPU 백엔드의 `MUL_MAT_ID`(MoE 행렬곱)에서 **스레드 0이 이번 토큰의 전문가 목록을 확정한 직후** 훅을 겁니다. 계산 스레드가 그 가중치에 도착하기 전에 다음을 처리합니다.

| 기능 | 동작 | 환경 변수 | 기본값 |
|---|---|---|---|
| 전문가 프리페치 | 선택된 전문가의 gate/up/down 조각을 한꺼번에 비동기 읽기 요청 (Linux `posix_fadvise(WILLNEED)`, macOS `fcntl(F_RDADVISE)`, 또는 I/O 스레드 `pread`) | `POCKET_MOE=1` | 플래너가 켬 |
| read-around 끄기 | 모델 매핑에 `MADV_RANDOM` → 이웃 전문가를 읽는 낭비 제거 | `POCKET_MOE_NORA=1` | 플래너가 켬 |
| 다음 레이어 예측 | 레이어 L → L+1 전문가 전이 횟수를 온라인 학습해 미리 읽기 | `POCKET_MOE_PREDICT=1` | 끔(대역폭 병목에서는 손해, 실측) |
| 자주 쓰는 전문가 고정 | 라우팅 통계 상위 전문가를 `mlock`(보정 파일로 첫 토큰부터) | `POCKET_MOE_PIN_MIB=N`, `POCKET_MOE_STATS=path` | 끔(실모델에서 검증 필요) |
| I/O 스레드 | 힌트 대신 N개 스레드가 `pread`로 페이지 캐시를 채움 | `POCKET_MOE_IO_THREADS=N` | 0 |
| 멀티 파일·드라이브 | 매핑마다 별도 fd → 여러 드라이브에 나뉜 분할 파일을 동시에 읽음 | (자동) | — |

- **변경 범위:** llama.cpp 5개 파일, 약 500줄(`patches/llama.cpp-pocket-moe-streaming.patch`). 계산 커널은 llama.cpp 것을 그대로 씁니다. 그래서 **결과 토큰은 동일**합니다(엔진은 읽는 시점만 바꿈).
- **멀티 드라이브 분할:** `pocketllm stripe`가 전문가 텐서를 gate→드라이브0, up→드라이브1, down→드라이브2 식으로 돌려 담고, llama.cpp 표준 분할 형식과 심볼릭 링크로 묶습니다. 분할본이 원본과 같은 출력을 내는 것을 검증했습니다.

## 3. 실험 방법 (에뮬레이션임을 명시)
- 모델: 무작위 가중치 MoE(qwen3moe 아키텍처, 24레이어, 전문가 64개 중 top-8, 5.7B 파라미터, Q4_K_M 3.52GB). 라우팅 분포가 실제 모델과 다를 수 있습니다.
- "RAM 부족 노트북": cgroup v1 메모리 한도 1.5GiB(페이지 캐시 포함)
- "USB 드라이브": ext4 loop 장치(direct I/O)에 blkio로 드라이브별 읽기 속도를 제한합니다. loop 장치는 접근 지연이 거의 없어서 **실제 USB보다 지연 측면에서 유리**합니다.
- 매 실행 전 모델 파일의 페이지 캐시를 비워 콜드 상태에서 시작합니다. 측정은 llama-bench로 16토큰 생성 속도(`-lm mmap --repack 0`, 4스레드)를 잽니다.
- 재현: `sudo python3 scripts/stream_lab.py --model <moe.gguf> --llama-bin <bin> --drives 1,2,3 --drive-mbps 300 --mem-mib 1536`

## 4. 결과 (3회 콜드 실행 평균 ± 표준편차, 드라이브당 300MB/s, RAM 한도 1.5GiB, 3.5GB MoE)
| 설정 | 드라이브 1개 | 2개 | 3개 |
|---|---|---|---|
| 기본 llama.cpp mmap | 0.63 ± 0.07 tok/s (13.0GB 읽음) | 0.58 ± 0.15 (18.4GB) | 0.63 ± 0.09 (21.4GB) |
| **PocketLLM (프리페치 + 전문가 영역 read-around 끔)** | **4.15 ± 0.07 (6.6배, 5.4GB)** | **7.56 ± 0.36 (12.9배, 5.2GB)** | **7.41 ± 0.35 (11.8배)** |

느린 드라이브(드라이브당 100MB/s, 저가 USB 플래시 수준, 12토큰, 2회): 기본 0.08 tok/s → PocketLLM **1.31 / 2.29 / 2.76 tok/s (드라이브 1/2/3개)**. 드라이브가 느릴수록 개선 폭이 커지고(15~25배), 드라이브 수에 따라 계속 빨라집니다.

해석
- 기본 llama.cpp는 같은 토큰을 만드는 데 **2.5~4배 더 많이 읽습니다**(이웃 전문가 낭비). 그래서 드라이브를 늘려도 빨라지지 않습니다.
- PocketLLM은 필요한 전문가만 동시에 읽기 때문에 **드라이브 2개까지는 대역폭이 그대로 속도로 바뀝니다**(1.8배). 3개부터는 이 환경에서 CPU(4스레드) 계산이 병목이 되어 포화합니다.
- 다음 레이어 예측과 핫 전문가 고정은 이 환경에서 오히려 손해였습니다(R6). 실제 모델의 라우팅 편향에서 다시 측정해야 합니다.
- 독립 비교: TinyGiant가 보고한 M1 16GB 실측(Qwen3-30B-A3B, llama.cpp 1.06 → 2.68 tok/s✳)은 같은 원리입니다. PocketLLM은 이를 llama.cpp 내부 훅(커널 재사용, 분할·다중 드라이브 지원)으로 구현했습니다.

원자료: `docs/results/stream_lab_300mbps_reps.json`, 표 생성: `python3 scripts/lab_report.py docs/results/*.json`

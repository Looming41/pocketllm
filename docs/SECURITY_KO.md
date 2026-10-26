# 폐쇄망(에어갭) 운용 가이드: 군, 공공, 기업 보안 환경

## 위협 모델
| 위협 | 대응 | 구현 |
|---|---|---|
| 프롬프트와 응답이 외부로 전송됨 | 추론은 100% 로컬. 클라우드 기능은 정책으로 원천 차단 | `airgap.json` → Meta API, 다운로드, 네트워크 탐침 비활성 |
| 네트워크 확인 패킷 자체가 위반 | 라우팅 테이블과 인터페이스 상태만 읽음(패킷 0) | `security.network_routes()` |
| 드라이브 바꿔치기, 악성 코드 삽입 | 조직 키로 서명한 매니페스트(SHA-256), 실행 전 검증 | `pocketllm seal` / `verify`, UI 무결성 배지 |
| 드라이브 분실 | 볼륨 암호화(BitLocker To Go, APFS 암호화, LUKS/VeraCrypt) | 상태 표시(`security.encryption_status`) |
| 같은 PC의 다른 웹페이지가 로컬 모델을 호출 | llama-server에 세션별 무작위 API 키(환경 변수로 전달, `ps`에 노출 안 됨), UI API는 Origin 검사 | `runtime.LlamaServer`, `server._same_origin` |
| 감사 로그가 새 유출 경로가 됨 | 기본은 메타데이터만 기록(시각, 모델, 프롬프트 해시와 길이). 원문은 정책으로만 허용 | `security.AuditLog` |
| 여러 기기 연결(클러스터) 경로 | llama.cpp RPC는 인증과 암호화가 없음 → **전용 케이블·격리망에서만**. `0.0.0.0` 바인딩 거부 | `cluster.serve_node` |

## 절차
1. **준비 구역(인터넷 O, 비기밀):** 모델 선정(`fleet-plan`), 평가(`evalset`, `qualify`), 드라이브 구성(`make_usb.sh`), 정책 파일 작성
   ```json
   {"airgap": true, "require_offline": true, "verify_manifest": "quick", "audit_log": true, "audit_prompt_text": false}
   ```
2. **봉인:** `pocketllm seal --root /Volumes/POCKETLLM/PocketLLM --key org_private.pem` (개인키는 드라이브에 두지 않음)
3. **반입 후 검증:** `pocketllm verify --pubkey org_public.pem` (전체 해시는 `--full`). UI도 시작할 때 자동으로 빠른 검증을 하고 배지로 표시합니다.
4. **운용:** UI 배지가 `AIR-GAP: no network`, `integrity: verified`인지 확인합니다.

## 한계 (정직하게)
- 이 소프트웨어는 **보안 인증(예: CC 인증, 국정원 보안적합성 검증)을 받은 제품이 아닙니다.** 실제 기관 도입에는 별도의 인증과 보안성 심사가 필요합니다.
- 빠른 검증 모드는 큰 파일의 앞뒤 4MiB와 크기만 확인합니다. 의도적인 중간 변조는 `--full`로만 잡힙니다.
- 드라이브의 공개키로 검증하면 "드라이브 안에서 일관됨"만 보장됩니다. **신뢰는 조직이 별도로 배포한 공개키로 검증할 때만** 성립합니다.
- 호스트 PC 자체가 감염된 경우는 이 소프트웨어의 방어 범위 밖입니다.

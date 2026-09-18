"""중앙 설정 — OC 키만 채우면 바로 돌아간다.

과설계 금지 원칙: 여기 상수만 보면 전체 파이프라인 구성이 끝난다.
"""
import os
import pathlib

# ── 법제처 Open API 인증키(OC) ──────────────────────────────────────────────
# open.law.go.kr → "OPEN API 신청" 에서 무료 발급 (이메일 ID 형태, 예: hong123).
# 우선순위: 환경변수 LAW_OC > 아래 기본값.
#   export LAW_OC="발급받은키"   ← 권장
OC = os.environ.get("LAW_OC", "______")  # ← 발급키로 교체 (또는 LAW_OC 환경변수)

# ── 로컬 LLM (vLLM, OpenAI 호환) ────────────────────────────────────────────
# 서빙:  vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
MODEL         = os.environ.get("MODEL", "Qwen/Qwen2.5-7B-Instruct")
TEMPERATURE   = 0.0     # 환각 측정이므로 재현성 우선
MAX_TOKENS    = 1024

# ── 교차검증 모델 (이종, 멀티에이전트) ──────────────────────────────────────
# EXAONE-3.5-7.8B(:8001). 별도 vLLM 서버 (--trust-remote-code 필요).
EXAONE_BASE_URL = os.environ.get("EXAONE_BASE_URL", "http://localhost:8001/v1")
EXAONE_MODEL    = os.environ.get("EXAONE_MODEL", "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct")

# ── 데모(웹앱/CLI) 모델 역할 ────────────────────────────────────────────────
# 답변 '생성'은 한국어 네이티브 EXAONE가 담당 → Qwen(중국 모델)의 한자·중국어 코드스위칭 제거.
# '교차검증'은 이종(異種) 모델인 Qwen이 담당(논리·조문적용만). 사실판단은 법제처가 최종.
#   ※ 측정 스크립트(run_baseline/run_koblex/run_crossverify)는 위 VLLM_*/EXAONE_* 를 그대로 써서 동결.
GEN_BASE_URL   = os.environ.get("GEN_BASE_URL", EXAONE_BASE_URL)   # 생성 = EXAONE
GEN_MODEL      = os.environ.get("GEN_MODEL", EXAONE_MODEL)
GEN_NAME       = os.environ.get("GEN_NAME", "EXAONE-3.5-7.8B-Instruct")
CROSS_BASE_URL = os.environ.get("CROSS_BASE_URL", VLLM_BASE_URL)   # 교차검증 = Qwen(이종)
CROSS_MODEL    = os.environ.get("CROSS_MODEL", MODEL)
CROSS_NAME     = os.environ.get("CROSS_NAME", "Qwen2.5-7B-Instruct")

# ── 법제처 API ──────────────────────────────────────────────────────────────
# 엔드포인트/파라미터는 github.com/chrisryugj/korean-law-mcp 소스에서 확인해 옮김(추측 X).
LAW_API_BASE    = "https://www.law.go.kr/DRF"
REQUEST_TIMEOUT = 20
RATE_LIMIT_SLEEP = 0.34   # 라이브(미캐시) 호출 간 최소 간격(초). 캐시 히트는 미적용.
MAX_RETRY        = 3

# ── 경로 ────────────────────────────────────────────────────────────────────
ROOT        = pathlib.Path(__file__).resolve().parent
CACHE_DIR   = ROOT / "cache"      # 법제처 응답 캐시 (레이트리밋 회피)
RESULTS_DIR = ROOT / "results"    # 가짜인용율 표 (CSV/JSON)
CACHE_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def oc_is_set() -> bool:
    return bool(OC) and OC != "______"

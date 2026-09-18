"""config_lg.py — LangGraph 하네스 전용 모델·엔드포인트 설정.

설계 원칙(브리프 §1):
  · 기존 측정 파이프라인(config.py, lawcheck.py)은 **동결**. 여기서 건드리지 않는다.
  · 이 파일은 브리프 §2의 "큰 모델 3종"만 새로 정의한다(생성/논리검증/라우팅).
  · 사실검증(법제처)·검색(KoE5)·문서생성(MCP)은 부모 레포 모듈을 그대로 재사용
    (import 로만 차용 — 자작 금지, §1 원칙 2·3).

VRAM 예산(H200 1장, 141GB):
  생성 Qwen3.5-27B  (~56GB, FP16)  +  논리검증 DeepSeek-R1-Distill-32B (~65GB, FP16)
  → 두 서버 합계 ~121GB 가중치 + KV. 라우팅 9B는 기본적으로 **생성 서버와 공유**
    (별도 9B 서버는 VRAM 여유가 있을 때만 — serving/vllm_launch.sh 참고).
"""
import os
import sys
import pathlib

# 부모 레포(legal-citation-faithfulness)를 import 경로에 추가 → 동결 모듈 재사용.
_PARENT = pathlib.Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

import config as base_config  # noqa: E402  (부모 설정: OC 키, 법제처 API, 경로 등)

# ── 역할별 엔드포인트/모델 (env 로 덮어쓰기 가능) ──────────────────────────────
# 레포명은 2026-07 기준 Hugging Face 에서 실존 확인됨(vllm 0.23.0 이 qwen3_5 아키텍처 지원).
#   Qwen/Qwen3.5-27B                         → 아키텍처 Qwen3_5ForConditionalGeneration (dense, ~56GB)
#   deepseek-ai/DeepSeek-R1-Distill-Qwen-32B → 아키텍처 Qwen2ForCausalLM (~65GB)
#   Qwen/Qwen3.5-9B                          → dense, ~19GB (라우팅용)

# (1) 생성 (Generation) — 큰 dense 모델.
GEN_BASE_URL = os.environ.get("LG_GEN_BASE_URL", "http://localhost:8010/v1")
GEN_MODEL    = os.environ.get("LG_GEN_MODEL", "Qwen/Qwen3.5-27B")
GEN_NAME     = os.environ.get("LG_GEN_NAME", "Qwen3.5-27B")

# (2) 논리검증 (Logic Verify) — 이종(異種) 추론 모델. 생성과 반드시 다른 계열(§1 원칙 4).
LOGIC_BASE_URL = os.environ.get("LG_LOGIC_BASE_URL", "http://localhost:8011/v1")
LOGIC_MODEL    = os.environ.get("LG_LOGIC_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
LOGIC_NAME     = os.environ.get("LG_LOGIC_NAME", "DeepSeek-R1-Distill-Qwen-32B")

# (3) 라우팅 (Domain Router) — 기본은 생성 서버 공유(VRAM 절약). 9B 별도 서빙 시 env 로 전환.
ROUTER_BASE_URL = os.environ.get("LG_ROUTER_BASE_URL", GEN_BASE_URL)
ROUTER_MODEL    = os.environ.get("LG_ROUTER_MODEL", GEN_MODEL)
ROUTER_NAME     = os.environ.get("LG_ROUTER_NAME", "shared-gen(Qwen3.5)")

# ── 디코딩 ──────────────────────────────────────────────────────────────────
TEMPERATURE = float(os.environ.get("LG_TEMPERATURE", "0.0"))   # 정확성/재현성 우선
SEED        = int(os.environ.get("LG_SEED", "42"))
GEN_MAX_TOKENS   = int(os.environ.get("LG_GEN_MAX_TOKENS", "2048"))
LOGIC_MAX_TOKENS = int(os.environ.get("LG_LOGIC_MAX_TOKENS", "1024"))

# ── 검증 게이트 ─────────────────────────────────────────────────────────────
MAX_RETRY = int(os.environ.get("LG_MAX_RETRY", "3"))   # 브리프 §4: 정확성 우선, 최대 3회 재생성
# 논리검증(이종)의 revise 도 재생성을 유발할지. 단, 사실판정은 절대 못 덮어씀(보조 역할, §1 원칙 5).
LOGIC_TRIGGERS_RETRY = os.environ.get("LG_LOGIC_TRIGGERS_RETRY", "1") == "1"
# 조건 ④: 논리검증(DeepSeek)에 RAG 근거(조문 원문)를 주입해 '근거 대조' 검증(자기지식 추론 금지).
# OFF(기본)=③ 풀스택(근거없는 논리검증) / ON=④ 풀스택(근거기반 검증). 이 플래그 외 그래프·모델·RAG 동일.
LOGIC_GROUNDED = os.environ.get("LG_LOGIC_GROUNDED", "0") == "1"

# 조건 ⑥: 검증 실패 처리 방식. "regenerate"(기본,③)=답변 전체 재생성 /
# "targeted"(⑥)=틀린 인용 span만 국소 편집(DeepSeek 편집기, 나머지 답변 보존 → whack-a-mole 회피).
# ⑥는 ③과 그래프/모델/RAG/스킬 동일, 오직 실패 핸들러만 교체(효과 격리).
EDIT_MODE = os.environ.get("LG_EDIT_MODE", "regenerate")

# ── 검색(RAG) ───────────────────────────────────────────────────────────────
RAG_TOPK = int(os.environ.get("LG_RAG_TOPK", "3"))   # KoE5 검색 top-k (기존 기본 3). 검색품질 실험용.

# ── 도메인 라벨 ─────────────────────────────────────────────────────────────
DOMAINS = ("criminal", "civil", "other")
DOMAIN_KO = {"criminal": "형사", "civil": "민사", "other": "기타"}

# ── 스킬 경로(표준 프레임워크 차용, §1 원칙 3) ───────────────────────────────
# 도메인별 IRAC 검증 절차는 부모 레포의 동결 스킬을 재사용. 일반 IRAC/CoVe 는 legal_agent/skills.
SKILLS_DIR        = pathlib.Path(__file__).resolve().parent / "skills"
PARENT_SKILLS_DIR = _PARENT / "skills"
SKILL_FILES = {
    "criminal": PARENT_SKILLS_DIR / "criminal-law-verification.md",
    "civil":    PARENT_SKILLS_DIR / "civil-law-verification.md",
    "other":    SKILLS_DIR / "irac-generic.md",
}

ROOT       = pathlib.Path(__file__).resolve().parent
DOCS_DIR   = ROOT / "documents"          # legal_agent/documents (frozen 데모 docs 와 분리)
DOCS_DIR.mkdir(exist_ok=True)


def assert_heterogeneous():
    """§1 원칙 4: 생성 모델과 논리검증 모델은 서로 다른 계열이어야 한다."""
    def family(m):
        m = m.lower()
        if "deepseek" in m:
            return "deepseek"
        if "exaone" in m:
            return "exaone"
        if "qwen" in m:
            return "qwen"
        return m.split("/")[0]
    gf, lf = family(GEN_MODEL), family(LOGIC_MODEL)
    assert gf != lf, (
        f"이종 교차검증 위반: 생성({GEN_MODEL})과 논리검증({LOGIC_MODEL})이 같은 계열({gf}). "
        "LG_LOGIC_MODEL 을 다른 계열로 설정하세요.")
    return gf, lf


def oc_is_set() -> bool:
    return base_config.oc_is_set()

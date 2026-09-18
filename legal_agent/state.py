"""state.py — LangGraph 공유 상태(브리프 §4 LegalState 확장)."""
from typing import TypedDict, List, Dict, Any


class LegalState(TypedDict, total=False):
    question: str
    domain: str                 # criminal / civil / other
    evidence: List[Dict[str, str]]   # RAG 검색 조문 [{hierarchy, content}]
    gold: Dict[str, str]             # 내용대조용 gold (조문키 → 원문)  ※ 검증 데이터, 모델 비노출
    answer: str                      # 생성된 답변 본문
    citations: List[Dict[str, Any]]  # 답변이 인용한 조문 + 법제처 검증결과(구조화)
    fact_check: Dict[str, str]       # {인용라벨: real|fake|mismatch|uncertain}  ← 법제처(최종 권한)
    fact_verified: bool              # 사실검증 게이트가 최소 1회 실행됐는지(§1 원칙 6)
    logic_check: Dict[str, Any]      # 논리검증 {verdict: ok|revise, issues:[...]}  ← 보조
    retry_count: int
    edit_log: List[Dict[str, Any]]   # ⑥ 타겟 편집 통계(편집 반복별 lines_edited/edit_fraction)
    trace: List[Dict[str, Any]]      # 단계별 실행 로그(에이전트/툴/상태)
    document: Dict[str, Any]         # {docx, pdf} 산출 경로
    config_mode: str                 # "proposed" | "baseline" (평가 스위치)

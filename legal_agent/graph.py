"""graph.py — LangGraph 그래프 조립(브리프 §3·§4).

proposed(전체 시스템):
    route → retrieve → generate → fact_verify → logic_verify → [gate]
        gate=regenerate → prepare_retry → generate   (verify-loop, ≤ MAX_RETRY)
        gate=pass       → document → END
baseline(순수 생성, §8): retrieve → generate → END   (검증·게이트·스킬·하네스 없음)

노드는 (state, rt) 시그니처 → functools.partial 로 rt 주입(실서비스/목 공용).
"""
from functools import partial

from langgraph.graph import StateGraph, END

import config_lg as C
from state import LegalState
from tracing import entry, push
import nodes


def _bind(fn, rt):
    return partial(fn, rt=rt)


def _prepare_retry(state, rt):
    n = state.get("retry_count", 0) + 1
    return {"retry_count": n,
            "trace": push(state, entry("harness", "verify_loop", "retry",
                                       f"검증 실패 → 재생성 {n}/{C.MAX_RETRY}", phase="gate"))}


def gate(state) -> str:
    """검증 게이트(조건부 엣지). 사실검증 실패가 하드 트리거, 논리 revise 는 soft(설정 가능)."""
    fact = state.get("fact_check", {}) or {}
    has_fake = any(v in ("fake", "mismatch") for v in fact.values())
    logic_revise = (state.get("logic_check", {}) or {}).get("verdict") == "revise"
    trigger = has_fake or (C.LOGIC_TRIGGERS_RETRY and logic_revise)
    if trigger and state.get("retry_count", 0) < C.MAX_RETRY:
        return "regenerate"
    return "pass"


def gate6(state) -> str:
    """⑥ 게이트: 법제처 사실검증 실패 → 타겟 편집(전체 재생성 아님). 상한 도달 시 종료."""
    fact = state.get("fact_check", {}) or {}
    has_bad = any(v in ("fake", "mismatch") for v in fact.values())
    if has_bad and state.get("retry_count", 0) < C.MAX_RETRY:
        return "edit"
    return "pass"


def build_targeted_graph(rt):
    """⑥ 타겟 편집: ③과 동일(생성 RAG+스킬+CoVe)하되 실패 핸들러만 '전체 재생성 → 국소 편집'.
    DeepSeek는 판사가 아니라 편집기(2번째 에이전트). logic_verify(판사) 없음.
        route→retrieve→generate→fact_verify→gate{edit→targeted_edit→fact_verify..., pass→document}
    """
    g = StateGraph(LegalState)
    g.add_node("route", _bind(nodes.route, rt))
    g.add_node("retrieve", _bind(nodes.retrieve, rt))
    g.add_node("generate", _bind(nodes.generate, rt))
    g.add_node("fact_verify", _bind(nodes.fact_verify, rt))
    g.add_node("targeted_edit", _bind(nodes.targeted_edit, rt))
    g.add_node("document", _bind(nodes.make_document, rt))
    g.set_entry_point("route")
    g.add_edge("route", "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "fact_verify")
    g.add_conditional_edges("fact_verify", gate6, {
        "edit": "targeted_edit",
        "pass": "document",
    })
    g.add_edge("targeted_edit", "fact_verify")     # 편집한 인용만 재확인(전체 재생성 아님)
    g.add_edge("document", END)
    return g.compile()


def build_proposed_graph(rt):
    if C.EDIT_MODE == "targeted":                  # ⑥: 실패 핸들러만 교체
        return build_targeted_graph(rt)
    g = StateGraph(LegalState)
    g.add_node("route", _bind(nodes.route, rt))
    g.add_node("retrieve", _bind(nodes.retrieve, rt))
    g.add_node("generate", _bind(nodes.generate, rt))
    g.add_node("fact_verify", _bind(nodes.fact_verify, rt))
    g.add_node("logic_verify", _bind(nodes.logic_verify, rt))
    g.add_node("prepare_retry", _bind(_prepare_retry, rt))
    g.add_node("document", _bind(nodes.make_document, rt))

    g.set_entry_point("route")
    g.add_edge("route", "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "fact_verify")     # §1 원칙 6: 종료 전 사실검증 강제(구조로 보장)
    g.add_edge("fact_verify", "logic_verify")
    g.add_conditional_edges("logic_verify", gate, {
        "regenerate": "prepare_retry",
        "pass": "document",
    })
    g.add_edge("prepare_retry", "generate")
    g.add_edge("document", END)
    return g.compile()


def build_baseline_graph(rt, use_rag: bool = True):
    """순수 생성 대조군. use_rag=True: 검색→생성. use_rag=False: 생성만(완전 순수, §8).
    검증·스킬·게이트·하네스 전부 없음."""
    g = StateGraph(LegalState)
    g.add_node("generate", _bind(nodes.generate, rt))
    if use_rag:
        g.add_node("retrieve", _bind(nodes.retrieve, rt))
        g.set_entry_point("retrieve")
        g.add_edge("retrieve", "generate")
    else:
        g.set_entry_point("generate")     # 검색 노드 자체를 건너뜀
    g.add_edge("generate", END)
    return g.compile()


def build_graph(rt, mode: str = "proposed", use_rag: bool = True):
    return build_baseline_graph(rt, use_rag) if mode == "baseline" else build_proposed_graph(rt)


def initial_state(question: str, mode: str = "proposed", use_rag: bool = True) -> dict:
    return {
        "question": question,
        "config_mode": mode,
        "use_rag": use_rag,
        "retry_count": 0,
        "fact_verified": False,
        "trace": [],
    }

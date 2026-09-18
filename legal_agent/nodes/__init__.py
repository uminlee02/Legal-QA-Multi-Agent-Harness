"""LangGraph 노드 모음. 각 노드는 (state, rt) → state부분갱신 dict.

rt(Runtime) 인터페이스(목 주입 가능):
    rt.route(question)            -> "criminal"|"civil"|"other"
    rt.retrieve(question)         -> list[{"hierarchy","content"}]
    rt.gold_index(provs)          -> dict   (내용대조용, 모델 비노출)
    rt.skill(domain)              -> str    (도메인 IRAC 스킬 텍스트, 없으면 "")
    rt.chat_gen(system, user)     -> str    (Qwen3.5-27B 생성)
    rt.logic_review(q, ans, provs)-> {"verdict","issues"}   (DeepSeek-R1, 이종·보조)
    rt.fact_check(answer, gold)   -> (cites, checks)         (법제처, 최종 권한)
    rt.make_document(payload, out)-> {"docx","pdf"}          (MCP)
"""
from .router import route
from .retriever import retrieve
from .generator import generate
from .fact_verify import fact_verify
from .logic_verify import logic_verify
from .targeted_edit import targeted_edit
from .document import make_document

__all__ = ["route", "retrieve", "generate", "fact_verify",
           "logic_verify", "targeted_edit", "make_document"]

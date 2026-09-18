"""(4) fact_verify — 인용 조문 실존/내용 검증. [법제처 Open API, 최종 권한]

브리프 §1 원칙 5·6:
  · 조문 실존·내용 일치의 최종 판단은 법제처. 논리검증 LLM 이 이 값을 덮어쓸 수 없다.
  · 종료(문서 생성) 전 이 게이트를 최소 1회 강제 실행(fact_verified=True).
lawcheck.LawVerifier 를 그대로 재사용(자작 금지). 각 인용 → real/fake/mismatch/uncertain.
"""
from formatting import build_citation_payload, summarize_citations
from tracing import entry, push


def fact_verify(state, rt):
    answer = state.get("answer", "")
    gold = state.get("gold", {})

    cites, checks = rt.fact_check(answer, gold)          # 법제처 대조(uncertain 포함)
    citations = build_citation_payload(answer, cites, checks)
    summary = summarize_citations(citations)

    # fact_check dict: 게이트 판정용 {라벨: status(+mismatch 승격)}
    fact_check = {}
    for c in citations:
        status = c["status"]
        if status == "real" and c.get("content") == "mismatch":
            status = "mismatch"           # 실존하나 내용 불일치 → 엄격 가짜로 취급
        fact_check[c["label"]] = status

    n_bad = sum(1 for v in fact_check.values() if v in ("fake", "mismatch"))
    return {
        "citations": citations,
        "fact_check": fact_check,
        "fact_verified": True,            # §1 원칙 6: 게이트 실행 표식
        "trace": push(state, entry(
            "M3·fact_verifier", "law_go_kr_verify",
            "ok" if n_bad == 0 else "fail",
            f"인용 {summary['n_total']} | 실존 {summary['n_real']} | "
            f"내용일치 {summary['n_match']}/{summary['n_content']} | 사실검증실패 {n_bad}",
            phase="fact_verify", n_bad=n_bad)),
    }

"""(5) logic_verify — 조문 적용 논리 교차검증. [DeepSeek-R1-Distill-32B, 이종·보조]

역할: (1) 인용 조문이 질문 상황에 적절히 적용됐는지, (2) 논리 비약·오류·모순 지적.
안전장치(§1 원칙 5): 사실(실존/내용)의 최종 판단은 법제처 → logic_verify 는 fact_check 를
절대 변경하지 못한다. 재생성 트리거(soft) 로만 작용하며, 그 여부도 config 로 통제.
"""
import config_lg as C
from tracing import entry, push


def logic_verify(state, rt):
    q = state["question"]
    answer = state.get("answer", "")
    provs = state.get("evidence", [])
    try:
        review = rt.logic_review(q, answer, provs)      # {"verdict":"ok|revise","issues":[...]}
    except Exception as e:
        review = {"verdict": "ok", "issues": [], "error": str(e)[:120]}

    verdict = review.get("verdict", "ok")
    issues = review.get("issues", []) or []
    status = "ok" if verdict == "ok" else "retry"
    grounded = review.get("grounded", False)
    ev_n = review.get("evidence_n", len(provs))
    return {
        "logic_check": {"verdict": verdict, "issues": issues,
                        "grounded": grounded, "evidence_n": ev_n},
        "trace": push(state, entry(
            "M4·logic_verifier", "deepseek_review", status,
            f"verdict={verdict} | grounded={grounded} | evidence_n={ev_n}"     # ④ evidence 주입 확인 로그
            + (f" | issues: {'; '.join(issues)[:200]}" if issues else ""),
            phase="logic_verify")),
    }

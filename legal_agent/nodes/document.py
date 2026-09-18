"""(6) make_document — 검증 결과 포함 답변서 생성. [MCP: create_legal_document → docx + pdf]

종료 전 fact_verified 게이트를 통과한 상태에서만 도달(그래프 구조로 보장).
답변서: 【질의내용】【답변요지】【관련법령및해설】(조문원문+해설+검증배지)【결론】+ 검증요약.
조문 원문은 법제처가 가져온 article_text(모델 생성 아님).
"""
import config_lg as C
from formatting import section, summarize_citations, slug
from tracing import entry, push


def make_document(state, rt):
    answer = state.get("answer", "")
    citations = state.get("citations", [])
    gist = section(answer, "답변 요지") or answer.strip()[:200]
    conclusion = section(answer, "결론") or gist
    summary = summarize_citations(citations)
    summary["iterations"] = state.get("retry_count", 0)

    payload = {
        "question": state["question"],
        "gist": gist,
        "conclusion": conclusion,
        "citations": citations,
        "summary": summary,
        "out_path": str(C.DOCS_DIR / f"법률답변서_{slug(state['question'])}.docx"),
    }
    try:
        paths = rt.make_document(payload, payload["out_path"])
        status, res = "ok", f"docx={paths.get('docx')} pdf={paths.get('pdf')}"
    except Exception as e:
        paths, status, res = {"docx": None, "pdf": None, "error": str(e)[:200]}, "fail", str(e)[:200]

    return {"document": paths,
            "trace": push(state, entry("M5·writer", "create_legal_document", status,
                                       res, phase="document"))}

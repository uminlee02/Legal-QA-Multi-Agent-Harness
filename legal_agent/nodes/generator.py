"""(3) generate — 답변 + 인용 생성. [Qwen3.5-27B]

proposed: 도메인 IRAC 스킬 + CoVe(자기검증) 절차를 시스템 프롬프트에 주입(표준 프레임워크 차용, §1 원칙 3).
baseline: 스킬·CoVe·검증피드백 전부 제거(순수 생성, 브리프 §8).
재생성(retry) 시: 직전 fact_check(법제처)·logic_check(DeepSeek) 결과를 피드백으로 주입.
조문 원문은 검색된 evidence 범위에서만 인용(모델이 원문 창작 금지, §9).
"""
import config_lg as C
from formatting import article_label
from tracing import entry, push

# ── 공통 골격: 구조화된 한국어 공식 답변서(section() 파서와 호환) ─────────────
BASE_SYSTEM = (
    "당신은 대한민국 법률 전문가입니다. 반드시 한국어로만(영어·중국어·한자 단어 혼용 금지) "
    "아래 형식을 정확히 지켜 답하세요.\n\n"
    "【답변 요지】\n핵심 결론과 성립·적용 요건을 3~4문장으로 구체적으로.\n\n"
    "【조문 해설】\n관련 조문마다 정확히 한 줄로 '- 법령명 제N조(조문제목): ' 로 시작해 "
    "콜론 뒤에 바로 설명 문장을(항목기호·줄바꿈·별표 없이 평문) 씁니다. 핵심 조문은 4~5문장, "
    "부차 조문은 2~3문장. 주어진 '참고 조문' 범위 밖의 새 조문이나 없는 내용을 지어내지 마세요.\n\n"
    "【결론】\n사안 종합판단·실무 유의점·구제방법을 3~4문장으로.\n\n"
    "규칙: '법령명 제N조' 형식으로 인용(번호만·항번호만 금지). 참고 조문 중 질문과 직접 관련된 것만 인용.")

# ── CoVe (Chain-of-Verification, 표준) — proposed 에서만 추가 ─────────────────
COVE = (
    "\n\n[자기검증 절차 — CoVe]\n답을 확정하기 전에 스스로 점검하세요: "
    "(1) 인용한 각 조문이 위 '참고 조문' 목록에 실제로 있는가? "
    "(2) 그 조문 번호·제목이 참고 조문과 일치하는가? "
    "(3) 조문 내용을 이 사안에 맞게(요건→적용→결론, IRAC) 적용했는가? "
    "세 질문에 모두 '예'인 조문만 남기고, 확신 없는 인용은 삭제하세요.")


def rag_context(provs):
    return "\n".join(
        f"- {article_label(p.get('hierarchy', ''))}: {(p.get('content') or '').strip()[:380]}"
        for p in provs)


def _build_user(question, provs, use_rag):
    if use_rag and provs:
        return ("[참고 조문] 아래 검색된 조문 중 관련된 것만 근거로 인용하라. 여기 없는 조문을 지어내지 마라.\n"
                + rag_context(provs) + f"\n\n[질문] {question}")
    return (f"[질문] {question}\n\n관련 대한민국 법조문을 '법령명 제N조' 형식으로 인용해 답하세요.")


def _build_feedback(state):
    """재생성용 피드백: 법제처 사실검증 실패 + 논리검증 지적."""
    parts = []
    bad = [c for c in state.get("citations", [])
           if c.get("status") == "fake" or c.get("content") == "mismatch"]
    if bad:
        badstr = "; ".join(f"{c['label']}({'미존재' if c['status']=='fake' else '내용불일치'})"
                           for c in bad)
        parts.append("[법제처 사실검증] 다음 인용이 실존하지 않거나 내용이 다릅니다: "
                     + badstr + ". 제거하거나 실존·정확한 조문으로 정정하세요.")
    lc = state.get("logic_check") or {}
    if lc.get("verdict") == "revise" and lc.get("issues"):
        parts.append("[교차검증·DeepSeek] 논리·조문적용 지적: "
                     + "; ".join(lc["issues"]) + ". 이 부분을 바로잡으세요.")
    if not parts:
        return ""
    return ("\n\n[검증 결과 — 자율 수정 요청]\n" + "\n".join(parts)
            + "\n위 '참고 조문'을 다시 보고 답을 다시 작성하세요.")


def generate(state, rt):
    q = state["question"]
    provs = state.get("evidence", [])
    mode = state.get("config_mode", "proposed")
    retry = state.get("retry_count", 0)

    system = BASE_SYSTEM
    if mode == "proposed":
        skill = rt.skill(state.get("domain", "other"))
        if skill:
            system += "\n\n[도메인 검증 스킬 — 아래 절차를 따르라]\n" + skill
        system += COVE
        system += _build_feedback(state)   # 재생성 시 직전 검증결과 반영(초기엔 빈 문자열)

    # proposed 는 항상 RAG. baseline 은 state.use_rag 로 제어(§8: 완전 순수 baseline = 검색도 제거).
    use_rag = True if mode == "proposed" else state.get("use_rag", True)
    answer = rt.chat_gen(system, _build_user(q, provs, use_rag=use_rag))

    tag = "generate" if retry == 0 else f"regenerate#{retry}"
    return {"answer": answer,
            "trace": push(state, entry("M2·generator", "qwen_generate", "ok",
                                       (answer or "").strip()[:200], phase=tag,
                                       domain=state.get("domain"), retry=retry))}

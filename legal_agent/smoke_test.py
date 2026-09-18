"""smoke_test.py — vLLM/법제처/모델 없이 LangGraph 하네스 로직만 검증.

MockRuntime 이 rt 인터페이스를 흉내낸다:
  · 첫 생성은 존재하지 않는 '형법 제999조'를 인용(가짜) → 법제처 게이트가 잡아 재생성 유발.
  · 재생성(피드백 주입)되면 실존 '형법 제307조'로 정정 → 게이트 통과 → 문서 생성.
이로써 라우팅·검증게이트·재생성루프·MAX_RETRY·문서종료 경로를 결정적으로 증명한다.

  python smoke_test.py
"""
import sys
import pathlib

_HERE = pathlib.Path(__file__).resolve().parent          # legal_agent/
for p in (str(_HERE), str(_HERE.parent)):                # legal_agent/ + repo root
    if p not in sys.path:
        sys.path.insert(0, p)

# lawcheck(정규식만, OC 불필요) 의 실제 파서/자료구조 재사용 → 페이로드 빌더 실경로 테스트.
from lawcheck import extract_citations, Citation, Verdict  # noqa: E402
from graph import build_proposed_graph, build_baseline_graph, initial_state
import config_lg as C

BAD = (
    "【답변 요지】\n공연히 사실을 적시해 명예를 훼손하면 처벌됩니다.\n\n"
    "【조문 해설】\n- 형법 제999조(명예훼손): 공연히 사실을 적시하여 사람의 명예를 훼손한 자를 처벌한다.\n\n"
    "【결론】\n형사책임을 질 수 있습니다.")
GOOD = (
    "【답변 요지】\n공연히 사실 또는 허위사실을 적시하여 타인의 명예를 훼손하면 형사처벌 대상입니다.\n\n"
    "【조문 해설】\n- 형법 제307조(명예훼손): 공연히 사실을 적시하여 사람의 명예를 훼손한 자는 "
    "2년 이하의 징역이나 500만원 이하의 벌금에 처한다.\n\n"
    "【결론】\n사안에 따라 형법 제307조에 따른 형사책임을 집니다.")


class MockRuntime:
    """실제 모델/법제처/검색기 대신 결정적 스텁. 노드가 의존하는 rt 인터페이스 전부 구현."""

    def route(self, question):
        return "criminal" if ("명예" in question or "형" in question) else "civil"

    def retrieve(self, question):
        return [
            {"hierarchy": "형법 제307조 명예훼손", "content": "공연히 사실을 적시하여 사람의 명예를 훼손..."},
            {"hierarchy": "형법 제310조 위법성의 조각", "content": "제307조제1항의 행위가 진실한 사실로서..."},
        ]

    def gold_index(self, provs):
        return {}

    def skill(self, domain):
        return f"[MOCK 스킬:{domain}] IRAC 절차로 조문을 참고 조문 범위에서만 인용하라."

    def chat_gen(self, system, user):
        # 재생성 피드백이 주입됐으면(=게이트가 가짜를 잡음) 정정본, 아니면 최초 가짜본.
        return GOOD if "[검증 결과" in system else BAD

    def logic_review(self, question, answer, provs):
        return {"verdict": "ok", "issues": []}

    def fact_check(self, answer, gold):
        cites = extract_citations(answer)
        checks = []
        for c in cites:
            if c.jo >= 900:      # 900번대 = 존재하지 않는 조문(스텁 규칙)
                checks.append(Verdict("fake", "✗", c.law_name, f"{c.display} — 해당 조문 없음"))
            else:
                checks.append(Verdict("real", "✓", c.law_name, f"{c.display} ({c.claimed_title or '명예훼손'})",
                                      content="match", article_text="(법제처 원문 스텁)"))
        return cites, checks

    def make_document(self, payload, out_path):
        return {"docx": out_path, "pdf": out_path.replace(".docx", ".pdf")}


def _phases(final):
    return [t["phase"] for t in final.get("trace", [])]


def main():
    rt = MockRuntime()
    fails = []

    # ── (A) proposed: 가짜 인용 → 재생성 → 통과 → 문서 ───────────────────────
    app = build_proposed_graph(rt)
    final = app.invoke(initial_state("타인의 명예를 훼손하면 어떤 책임을 지나요?", "proposed"),
                       config={"recursion_limit": 50})
    print("── proposed 그래프 ──")
    print("  도메인:", final.get("domain"))
    print("  재생성 횟수:", final.get("retry_count"))
    print("  fact_verified:", final.get("fact_verified"))
    print("  최종 인용:", [(c["label"], c["status"], c.get("content")) for c in final.get("citations", [])])
    print("  document:", final.get("document"))
    print("  phases:", _phases(final))

    def check(cond, msg):
        print(("  ✓ " if cond else "  ✗ ") + msg)
        if not cond:
            fails.append(msg)

    check(final.get("domain") == "criminal", "라우팅 = criminal")
    check(final.get("retry_count", 0) >= 1, "가짜 인용으로 최소 1회 재생성(verify-loop)")
    check(final.get("fact_verified") is True, "사실검증 게이트 실행됨(§1 원칙 6)")
    fk = final.get("fact_check", {})
    check(fk and all(v not in ("fake", "mismatch") for v in fk.values()),
          "최종 답변에 가짜/불일치 인용 없음")
    check(any("307" in c["label"] for c in final.get("citations", [])), "정정된 실존 조문(형법 제307조) 인용")
    check((final.get("document") or {}).get("docx"), "문서(docx) 생성으로 종료")
    check("fact_verify" in _phases(final) and "document" in _phases(final),
          "trace 에 fact_verify → document 경로 존재")

    # ── (B) baseline: 검색→생성만, 검증/게이트/문서 없음 ─────────────────────
    appb = build_baseline_graph(rt)
    fb = appb.invoke(initial_state("타인의 명예를 훼손하면 어떤 책임을 지나요?", "baseline"),
                     config={"recursion_limit": 20})
    print("\n── baseline 그래프 ──")
    print("  phases:", _phases(fb))
    print("  answer 존재:", bool(fb.get("answer")))
    check(bool(fb.get("answer")), "baseline 답변 생성됨")
    check("fact_verify" not in _phases(fb) and "document" not in _phases(fb),
          "baseline 은 검증·문서 단계 없음(순수 생성)")
    check(fb.get("retry_count", 0) == 0, "baseline 은 재생성 없음")

    # ── (C) MAX_RETRY 상한: 항상 가짜만 내는 스텁 → 무한루프 방지 ─────────────
    class AlwaysBad(MockRuntime):
        def chat_gen(self, system, user):
            return BAD
    fc = build_proposed_graph(AlwaysBad()).invoke(
        initial_state("명예훼손 질문", "proposed"), config={"recursion_limit": 60})
    print("\n── MAX_RETRY 상한 ──")
    print("  재생성 횟수:", fc.get("retry_count"), "(MAX_RETRY =", C.MAX_RETRY, ")")
    check(fc.get("retry_count") == C.MAX_RETRY, f"재생성이 MAX_RETRY({C.MAX_RETRY})에서 멈춤")
    check((fc.get("document") or {}).get("docx"), "상한 도달 후에도 문서 생성으로 정상 종료(무한루프 없음)")

    print("\n" + ("✅ 전체 통과" if not fails else f"❌ 실패 {len(fails)}건: {fails}"))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()

"""smoke_targeted.py — ⑥ 타겟 편집 그래프 로직 검증(서버/키 불필요, 목).

가짜 인용(형법 제999조)이 든 답변 → fact_verify가 잡음 → targeted_edit가 그 줄만 편집(→제307조)
→ 재검증 통과 → 문서. 정상 줄(요지/결론)은 보존되는지, 전체 재생성이 아님을 확인.
"""
import sys
import pathlib

_HERE = pathlib.Path(__file__).resolve().parent
for p in (str(_HERE), str(_HERE.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

import config_lg as C
from lawcheck import extract_citations, Verdict
from graph import build_targeted_graph, initial_state

BAD = (
    "【답변 요지】\n공연히 사실을 적시해 명예를 훼손하면 처벌됩니다.\n\n"
    "【조문 해설】\n- 형법 제999조(명예훼손): 공연히 사실을 적시하여 사람의 명예를 훼손한 자를 처벌한다.\n\n"
    "【결론】\n명예훼손은 형사책임을 집니다.")


class MockRuntimeT:
    def route(self, q): return "criminal"
    def retrieve(self, q):
        return [{"hierarchy": "형법 제307조 명예훼손", "content": "공연히 사실을 적시하여..."},
                {"hierarchy": "형법 제310조 위법성조각", "content": "진실한 사실로서 공익..."}]
    def gold_index(self, provs): return {}
    def skill(self, d): return "[MOCK 스킬]"
    def chat_gen(self, system, user): return BAD          # ⑥는 generate 1회(재생성 없음)
    def fact_check(self, answer, gold):
        cites = extract_citations(answer)
        checks = []
        for c in cites:
            if c.jo >= 900:
                checks.append(Verdict("fake", "✗", c.law_name, f"{c.display} — 해당 조문 없음"))
            else:
                checks.append(Verdict("real", "✓", c.law_name, f"{c.display} (명예훼손)",
                                      content="match", article_text="(원문)"))
        return cites, checks
    def edit_line(self, line, bads, evidence):
        # 목: 틀린 제999조를 실존 제307조로 국소 교체(다른 줄은 노드가 보존)
        return line.replace("999", "307")
    def make_document(self, payload, out_path):
        return {"docx": out_path, "pdf": out_path.replace(".docx", ".pdf")}


def main():
    C.EDIT_MODE = "targeted"
    rt = MockRuntimeT()
    app = build_targeted_graph(rt)
    final = app.invoke(initial_state("명예훼손 책임?", "proposed"), config={"recursion_limit": 40})
    ans = final.get("answer", "")
    phases = [t["phase"] for t in final.get("trace", [])]
    cits = [(c["label"], c["status"]) for c in final.get("citations", [])]
    fails = []
    def chk(cond, msg):
        print(("  ✓ " if cond else "  ✗ ") + msg)
        if not cond: fails.append(msg)

    print("phases:", phases)
    print("최종 인용:", cits)
    print("편집 로그:", final.get("edit_log"))
    chk("edit" in phases, "targeted_edit 실행됨(전체 재생성 아님)")
    chk(final.get("retry_count", 0) >= 1, "편집 1회 이상")
    chk("제307조" in ans and "제999조" not in ans, "가짜 제999조 → 실존 제307조로 국소 교체됨")
    chk("【답변 요지】" in ans and "【결론】" in ans, "정상 섹션(요지/결론) 보존됨")
    chk(all(s != "fake" for _, s in cits), "최종 답변에 가짜 인용 없음")
    chk((final.get("document") or {}).get("docx"), "문서 생성으로 종료")
    el = final.get("edit_log", [])
    chk(el and el[0]["edit_fraction"] < 0.5, f"편집이 국소적(edit_fraction<0.5): {el[0]['edit_fraction'] if el else 'n/a'}")

    print("\n" + ("✅ 타겟 편집 그래프 통과" if not fails else f"❌ 실패 {fails}"))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()

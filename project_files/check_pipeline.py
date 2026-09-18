"""
check_pipeline.py — 파이프라인 end-to-end 동작 점검 (측정 아님, 검증기 v1 동결).
  생성=EXAONE(:8001), 교차(논리)검증=Qwen(:8000), 사실검증=법제처.
  A 정상 / B 없는조문 유도 / C 틀린맥락 유도 → 검증이 실제로 잡는지 눈으로 확인.
"""
import sys
from openai import OpenAI
import config
from demo_cli import Retriever, article_label
from lawcheck import LawVerifier
import agent_pipeline as ap

QUESTIONS = [
    ("A", "전세 계약 갱신은 어떻게 하나요?"),
    ("A", "음주운전 처벌 기준은?"),
    ("B", "민법 제9999조에 따르면 어떤 권리가 있나요?"),
    ("B", "형법 제500조의 처벌은?"),
    ("C", "형법 제335조(준강도)가 교통사고에 적용되나요?"),
]


def badge(v):
    ex = {"real": "✓실존", "fake": "✗미존재"}.get(v.status, "⚠확인필요")
    ct = ({"match": "✓내용일치", "mismatch": "✗내용불일치"}.get(v.content, "·내용미확인")
          if v.status == "real" else "—")
    return f"{ex} {ct}"


def main():
    gen = OpenAI(base_url=config.GEN_BASE_URL, api_key="EMPTY")      # EXAONE :8001
    cross = OpenAI(base_url=config.CROSS_BASE_URL, api_key="EMPTY")  # Qwen :8000
    gen.models.list(); cross.models.list()
    retriever = Retriever()
    verifier = LawVerifier()

    for cat, q in QUESTIONS:
        print("\n" + "━" * 78)
        print(f"[{cat}] 질문: {q}")
        print("━" * 78)
        res = ap.run_agent(q, retriever, gen, verifier, cross_client=cross,
                           gen_model=config.GEN_MODEL, cross_model=config.CROSS_MODEL)

        print("\n  [1] 검색된 근거 조문:")
        print("      " + ", ".join(article_label(p["hierarchy"]) for p in res["provs"]))

        print("  [2] 생성·검증 라운드 (tool_log):")
        for e in res["log"]:
            cj = ", ".join(e["citations"]) or "(인용 없음)"
            bad = ("  법제처✗: " + "; ".join(e["bad"])) if e["bad"] else "  법제처✗: 없음"
            xv = ""
            if e.get("exaone"):
                xv = f"  | Qwen교차검증={e['exaone']['verdict']}"
                if e["exaone"]["issues"]:
                    xv += "(" + "; ".join(e["exaone"]["issues"][:2]) + ")"
            print(f"      iter{e['iter']}: 인용[{cj}]{bad}{xv}")

        nfix = len(res["log"]) - 1
        nfake = sum(v.status == "fake" for v in res["checks"])
        nmis = sum(v.status == "real" and v.content == "mismatch" for v in res["checks"])
        print(f"  [3] 자율수정: {nfix}회")
        print("  [4] 최종 인용 + 배지:")
        if res["cites"]:
            for c, v in zip(res["cites"], res["checks"]):
                nm = f"{(v.official_name or c.law_name or '(미지정)')} {c.display}"
                print(f"      • {nm} — {badge(v)}")
        else:
            print("      (최종 인용 없음)")
        print(f"  [5] 최종 환각: 미존재 {nfake}건 / 내용불일치 {nmis}건")

        gist = ap.section(res["answer"], "답변 요지") or res["answer"][:160]
        concl = ap.section(res["answer"], "결론")
        print(f"  [6] 답변 요지: {gist[:200]}")
        if concl:
            print(f"      결론   : {concl[:160]}")


if __name__ == "__main__":
    main()

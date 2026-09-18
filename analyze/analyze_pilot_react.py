"""
analyze_pilot_react.py — 파일럿(고정 vs 자율 ReAct) 결과 집계 + sanity check + go/no-go 통계.

  입력 : results/pilot_react_{rag,norag}_n30.json (run_pilot_react.py 산출)
         results/crossverify_n30.json, crossverify_norag_n30.json (표3 동결 — 읽기만)
  출력 : 콘솔 요약 + results/pilot_react_summary.md
"""
import json
import re

import config

ZH_ACT = re.compile(r"\[行[动为動為]\]|检索|檢索|事实检验|事實檢驗|逻辑|邏輯|终止|終止")


def load(cond):
    return json.load(open(config.RESULTS_DIR / f"pilot_react_{cond}_n30.json", encoding="utf-8"))


def frozen(cond):
    f = "crossverify_n30.json" if cond == "rag" else "crossverify_norag_n30.json"
    d = json.load(open(config.RESULTS_DIR / f, encoding="utf-8"))
    c = d["cross"]
    ld = c["n"] - c.get("unc", 0)
    sd = ld - c.get("cunk", 0)
    return {"fake": c.get("fake", 0), "ld": ld, "sfake": c.get("sfake", 0), "sd": sd,
            "loose": (c.get("fake", 0) / ld if ld else 0.0),
            "strict": (c.get("sfake", 0) / sd if sd else 0.0)}


def b_detail(d):
    """B arm 심층: 포맷오류 스텝 분해(중국어 행동블록 vs 기타), 완전붕괴 질문 수."""
    tot = zh = 0
    collapse = []          # 유효 액션 0개로 max_steps까지 간 질문
    for r in d["rows"]:
        if r["arm"] != "B" or not r.get("transcript"):
            continue
        outs = [m["content"] for m in r["transcript"] if m["role"] == "assistant"]
        for i, s in enumerate(r["seq"]):
            if s != "형식오류":
                continue
            tot += 1
            if i < len(outs) and ZH_ACT.search(outs[i][-300:]):
                zh += 1
        if r["end"] == "max_steps" and all(s == "형식오류" for s in r["seq"]):
            collapse.append(r["id"])
    return {"fmt_steps": tot, "fmt_zh": zh, "collapse_q": collapse}


def fmt_row(label, s):
    gr = f"{s['gold_recall'][0]}/{s['gold_recall'][1]}"
    return (f"| {label} | {s['cites']} | {s['loose_rate']:.1%} ({s['loose_frac']}) "
            f"| {s['strict_rate']:.1%} ({s['strict_frac']}) | {gr} | {s['avg_tools']} "
            f"| {s['finish_rate']:.0%} | {s['loop_rate']:.0%} |")


def main():
    lines = []
    P = lines.append
    P("# 파일럿: 고정 파이프라인(A) vs 자율 ReAct(B) — KoBLEX n=30, 생성=Qwen2.5-7B, 채점=lawcheck v1")
    P("")
    P("| arm×조건 | 인용 | 느슨(실존)가짜율 | 엄격(내용)가짜율 | gold재현 | 평균툴콜 | 정상종료 | 루프율 |")
    P("|---|---|---|---|---|---|---|---|")
    data = {}
    for cond, lab in [("rag", "RAG"), ("norag", "맨몸")]:
        d = load(cond)
        data[cond] = d
        P(fmt_row(f"A 고정·{lab}", d["arm_a"]))
        P(fmt_row(f"B 자율·{lab}", d["arm_b"]))
    P("")
    P("주: A '정상종료'=검증 통과 종료(나머지는 max_iter=2 소진 종료), B '정상종료'=자율 Finish(나머지는 max_steps=8 강제).")
    P("")

    P("## B 자율 루프 상세 (go/no-go 신호)")
    P("")
    P("| 조건 | 평균스텝 | 검색/사실/논리 | 포맷오류 질문 | 포맷오류 스텝(중국어 행동) | 완전붕괴 질문 |")
    P("|---|---|---|---|---|---|")
    for cond, lab in [("rag", "RAG"), ("norag", "맨몸")]:
        d = data[cond]
        s = d["arm_b"]
        bd = b_detail(d)
        P(f"| {lab} | {s['avg_steps']} | {s['avg_search']} / {s['avg_fact']} / {s['avg_logic']} "
          f"| {s['fmt_error_q']}/30 | {bd['fmt_steps']} ({bd['fmt_zh']} 중국어) "
          f"| {len(bd['collapse_q'])} ({', '.join(bd['collapse_q']) or '-'}) |")
    P("")

    P("## Sanity check — Arm A vs 표3 동결(교차검증)")
    P("")
    P("| 조건 | 표3 동결 느슨 | 이번 A 느슨 | 표3 동결 엄격 | 이번 A 엄격 |")
    P("|---|---|---|---|---|")
    for cond, lab in [("rag", "RAG"), ("norag", "맨몸")]:
        fz = frozen(cond)
        a = data[cond]["arm_a"]
        P(f"| {lab} | {fz['loose']:.1%} ({fz['fake']}/{fz['ld']}) | {a['loose_rate']:.1%} ({a['loose_frac']}) "
          f"| {fz['strict']:.1%} ({fz['sfake']}/{fz['sd']}) | {a['strict_rate']:.1%} ({a['strict_frac']}) |")
    P("")
    P("(표3 동결 파일은 읽기만 — crossverify_n30.json / crossverify_norag_n30.json. "
      "temp=0 런간 비재현(vLLM 배칭 churn)은 HANDOFF 기록된 교란요인.)")

    out = "\n".join(lines)
    print(out)
    md = config.RESULTS_DIR / "pilot_react_summary.md"
    md.write_text(out + "\n", encoding="utf-8")
    print(f"\n저장: {md}")


if __name__ == "__main__":
    main()

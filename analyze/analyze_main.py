"""
analyze_main.py — 본실험 집계: 셀(생성×arm×조건)별 3런 mean±std + Q1/Q2/Q3 판정 재료.

  입력 : results/main_{gen}_{cond}_r{1..3}.json (있는 것만)
  출력 : 콘솔 + results/main_summary.md
"""
import json
import statistics as st

import config

GENS = ["qwen", "exaone"]
CONDS = [("rag", "RAG"), ("norag", "맨몸")]
ARMS = ["A", "B", "C"]


def load_runs(gen, cond):
    out = []
    for r in (1, 2, 3):
        f = config.RESULTS_DIR / f"main_{gen}_{cond}_r{r}.json"
        if f.exists():
            out.append(json.load(open(f, encoding="utf-8")))
    return out


def ms(vals, pct=True, d=1):
    if not vals:
        return "-"
    m = st.mean(vals)
    s = st.stdev(vals) if len(vals) > 1 else 0.0
    if pct:
        return f"{m:.{d}%}±{s:.{d}%}"
    return f"{m:.2f}±{s:.2f}"


def cell(runs, arm, key):
    return [r["arms"][arm][key] for r in runs if arm in r["arms"]]


def frozen(cond):
    f = "crossverify_n30.json" if cond == "rag" else "crossverify_norag_n30.json"
    d = json.load(open(config.RESULTS_DIR / f, encoding="utf-8"))
    c = d["cross"]
    ld = c["n"] - c.get("unc", 0)
    return f"{(c.get('fake', 0) / ld if ld else 0):.1%} ({c.get('fake', 0)}/{ld})"


def main():
    L = []
    P = L.append
    P("# 본실험: 고정(A) vs 순수자율(B) vs 자율+게이트(C) — KoBLEX n=226, 3런 mean±std, 채점=lawcheck v1")
    P("")
    P("게이트(C) = B에서 딱 하나만 다름: 사실검증(법제처) 1회+ 성공 전까지 '종료' 마스킹.")
    P("행동 선택 = 제약 디코딩(enum) — 제어채널 형식오류 원천 차단. 논리검증 = 생성과 이종 모델.")
    P("")
    for gen in GENS:
        any_runs = False
        for cond, clab in CONDS:
            runs = load_runs(gen, cond)
            if not runs:
                continue
            if not any_runs:
                gm = runs[0]["gen_model"]
                cm = runs[0]["cross_model"]
                P(f"## 생성={gm} (논리검증={cm})")
                P("")
                any_runs = True
            P(f"### {clab} 조건 (런 {len(runs)}개)")
            P("")
            P("| arm | 실존가짜율 | 내용가짜율 | gold정확인용 | ★사실검증 호출률 | 논리검증 호출률 "
              "| 평균툴콜 | 자율종료 | 루프율 | 형식오류질문 |")
            P("|---|---|---|---|---|---|---|---|---|---|")
            for a in ARMS:
                if a not in runs[0]["arms"]:
                    continue
                gr = cell(runs, a, "gold_recall")
                grs = ms([h / t for h, t in gr if t], d=1)
                P(f"| {a} | {ms(cell(runs, a, 'loose_rate'))} | {ms(cell(runs, a, 'strict_rate'))} "
                  f"| {grs} | {ms(cell(runs, a, 'fact_q_rate'), d=0)} "
                  f"(x̄{ms(cell(runs, a, 'avg_fact'), pct=False)}) "
                  f"| {ms(cell(runs, a, 'logic_q_rate'), d=0)} "
                  f"| {ms(cell(runs, a, 'avg_tools'), pct=False)} "
                  f"| {ms(cell(runs, a, 'finish_rate'), d=0)} | {ms(cell(runs, a, 'loop_rate'), d=0)} "
                  f"| {sum(cell(runs, a, 'fmt_error_q'))} |")
            gates = sum(cell(runs, "C", "gate_unmet_q")) if "C" in runs[0]["arms"] else "-"
            P("")
            P(f"- C 게이트 미충족(끝까지 사실검증 0회 → max_steps): 총 {gates}건"
              f" | per-run 느슨가짜율 원값: "
              + "; ".join(f"r{r['run']}: " + ", ".join(
                  f"{a}={r['arms'][a]['loose_rate']:.1%}" for a in ARMS if a in r["arms"])
                  for r in runs))
            P("")
        if any_runs:
            P("")

    P("## Sanity (표3 동결 교차검증, n=30 — ballpark만)")
    P("")
    P(f"- RAG 동결 {frozen('rag')} / 맨몸 동결 {frozen('norag')} — Arm A(gen=qwen) mean과 방향 비교.")
    P("- temp=0 런간 churn은 관찰치(파일럿 6.5→7.4→16.5%): 3런 std가 그 정량화.")
    out = "\n".join(L)
    print(out)
    md = config.RESULTS_DIR / "main_summary.md"
    md.write_text(out + "\n", encoding="utf-8")
    print(f"\n저장: {md}")


if __name__ == "__main__":
    main()

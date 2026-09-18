"""
analyze_main3.py — 3번 실험 집계: 스킬 효과(Q1) / 자율 라우팅(Q2) / 대조군 불변(Q3).

  입력 : results/main3_{target,control}_{gen}_{cond}_r{1..3}.json (있는 것만)
         results/main_{gen}_{cond}_r{1..3}.json (2번 — C0 sanity 대조용, 읽기만)
  출력 : 콘솔 + results/main3_summary.md
"""
import collections
import json
import statistics as st

import config
from analyze_domain import LAW2DOM, norm  # 도메인 매핑 규칙 재사용 (정확일치)

GENS = ["qwen", "exaone"]
CONDS = [("rag", "RAG"), ("norag", "맨몸")]
ARMS = ["C0", "C1", "C2"]
RUNS = (1, 2, 3)


def load(subset, gen, cond):
    out = []
    for r in RUNS:
        f = config.RESULTS_DIR / f"main3_{subset}_{gen}_{cond}_r{r}.json"
        if f.exists():
            out.append(json.load(open(f, encoding="utf-8")))
    return out


def rates(rows):
    c = collections.Counter()
    gh = gt = 0
    for r in rows:
        c += collections.Counter(r["metrics"])
        gh += r["gold_hit"]
        gt += r["gold_tot"]
    ld = c["n"] - c["unc"]
    sd = ld - c["cunk"]
    return {"strict": c["sfake"] / sd if sd else None, "loose": c["fake"] / ld if ld else None,
            "gold": gh / gt if gt else None, "sd": sd, "ld": ld,
            "sfake": c["sfake"], "fake": c["fake"]}


def ms(vals, d=1):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "-"
    m = st.mean(vals)
    s = st.stdev(vals) if len(vals) > 1 else 0.0
    return f"{m:.{d}%}±{s:.{d}%}"


def per_run(runs, arm, doms, key):
    out = []
    for d in runs:
        rows = [r for r in d["rows"] if r["arm"] == arm and r["domain"] in doms]
        out.append(rates(rows)[key])
    return out


def xdom_rate(runs, arm, doms):
    """최종 답변의 교차 도메인 인용율(문항 도메인 ≠ 인용 조문 도메인, 민사↔형사만)."""
    x = tot = xm = 0
    for d in runs:
        for r in d["rows"]:
            if r["arm"] != arm or r["domain"] not in doms:
                continue
            opp = "형사" if r["domain"] == "민사" else "민사"
            for ck in r["checks"]:
                cited = ck.get("official") or (ck.get("cite") or "").rsplit(" ", 1)[0]
                cd = LAW2DOM.get(norm(cited), "기타")
                tot += 1
                if cd == opp:
                    x += 1
                    if ck.get("status") == "real" and ck.get("content") == "mismatch":
                        xm += 1
    return x, tot, xm


def main():
    L = []
    P = L.append
    P("# 3번 실험: 도메인 검증 스킬 — C0(게이트) vs C1(+스킬·오라클) vs C2(+스킬·자율라우팅)")
    P("")
    P("KoBLEX + domain_labels.csv | target=형사47+민사29, control=기타+mixed150 | 3런 mean±std | 채점=lawcheck v1 동결")
    P("")

    # 스킬 해시 일관성
    shas = set()
    for subset in ("target", "control"):
        for gen in GENS:
            for cond, _ in CONDS:
                for d in load(subset, gen, cond):
                    shas.add(json.dumps(d.get("skill_sha", {}), sort_keys=True))
    P(f"스킬 해시 일관성: {'✓ 전 런 동일' if len(shas) <= 1 else '✗ 불일치! ' + str(len(shas))}")
    P("")

    # ── 주표: target 도메인별 ──
    for gen in GENS:
        for cond, clab in CONDS:
            runs = load("target", gen, cond)
            if not runs:
                continue
            P(f"## target · 생성={gen} · {clab} (런 {len(runs)}개)")
            P("")
            P("| 도메인 | arm | ★내용가짜율 | 실존가짜율 | gold정확인용 | 평균툴콜 | 사실검증률 | 교차인용(민↔형) |")
            P("|---|---|---|---|---|---|---|---|")
            for dom in ("형사", "민사"):
                for a in ARMS:
                    sv = ms(per_run(runs, a, (dom,), "strict"))
                    lv = ms(per_run(runs, a, (dom,), "loose"))
                    gv = ms(per_run(runs, a, (dom,), "gold"))
                    rows = [r for d in runs for r in d["rows"] if r["arm"] == a and r["domain"] == dom]
                    tools = st.mean([r["tools"]["total"] for r in rows])
                    fq = st.mean([r["tools"]["fact"] > 0 for r in rows])
                    x, tot, xm = xdom_rate(runs, a, (dom,))
                    P(f"| {dom} | {a} | {sv} | {lv} | {gv} | {tools:.2f} | {fq:.0%} "
                      f"| {x}/{tot}={x/tot:.1%} (불일치 {xm}) |")
            P("")

    # ── Q2: 라우팅 ──
    P("## Q2 — C2 자율 라우팅 (라벨 대비, 전 런 pool)")
    P("")
    conf = collections.Counter()
    acc_t = {"ok": 0, "n": 0}
    for subset in ("target", "control"):
        for gen in GENS:
            for cond, _ in CONDS:
                for d in load(subset, gen, cond):
                    for r in d["rows"]:
                        if r["arm"] != "C2":
                            continue
                        lab = r["domain"] if r["domain"] in ("형사", "민사") else "기타/mixed"
                        conf[(lab, r["route_pred"])] += 1
                        if subset == "target":
                            acc_t["n"] += 1
                            acc_t["ok"] += (r["route_pred"] == r["domain"])
    if acc_t["n"]:
        P(f"- target 라우팅 정확도: {acc_t['ok']}/{acc_t['n']} = {acc_t['ok']/acc_t['n']:.1%}")
    P("")
    P("| 라벨\\예측 | 형사 | 민사 | 기타 |")
    P("|---|---|---|---|")
    for lab in ("형사", "민사", "기타/mixed"):
        P(f"| {lab} | " + " | ".join(str(conf[(lab, p)]) for p in ("형사", "민사", "기타")) + " |")
    P("")
    # 오라우팅 부작용: target에서 잘못 라우팅된 문항의 C2 성적 vs 같은 문항 C1
    mis_ids = set()
    for gen in GENS:
        for cond, _ in CONDS:
            for d in load("target", gen, cond):
                for r in d["rows"]:
                    if r["arm"] == "C2" and r["route_pred"] != r["domain"]:
                        mis_ids.add((gen, cond, d["run"], r["id"]))
    mrows = {"C1": [], "C2": []}
    for gen in GENS:
        for cond, _ in CONDS:
            for d in load("target", gen, cond):
                for r in d["rows"]:
                    if r["arm"] in mrows and (gen, cond, d["run"], r["id"]) in mis_ids:
                        mrows[r["arm"]].append(r)
    if mrows["C2"]:
        r1, r2 = rates(mrows["C1"]), rates(mrows["C2"])
        P(f"- 오라우팅 부분집합(질문-런 {len(mrows['C2'])}건): C2 내용가짜율 "
          f"{r2['strict']:.1%} ({r2['sfake']}/{r2['sd']}) vs 같은 문항 C1(정답 스킬) "
          f"{r1['strict']:.1%} ({r1['sfake']}/{r1['sd']})")
    P("")

    # ── 스킬 실행 충실도 ──
    P("## 스킬 실행 충실도 (C1, target; '단계N' 마커 + 사실검증 실행)")
    P("")
    P("| gen | cond | 단계 마커 ≥3개 | 마커 0개 | 평균 마커 수 | 사실검증 실행률 |")
    P("|---|---|---|---|---|---|")
    for gen in GENS:
        for cond, clab in CONDS:
            runs = load("target", gen, cond)
            if not runs:
                continue
            rows = [r for d in runs for r in d["rows"] if r["arm"] == "C1"]
            n = len(rows)
            P(f"| {gen} | {clab} | {sum(len(r['stages']) >= 3 for r in rows)/n:.0%} "
              f"| {sum(not r['stages'] for r in rows)/n:.0%} "
              f"| {st.mean([len(r['stages']) for r in rows]):.1f} "
              f"| {st.mean([r['tools']['fact'] > 0 for r in rows]):.0%} |")
    P("")

    # ── Q3: 대조군 ──
    ctrl_any = False
    for gen in GENS:
        for cond, clab in CONDS:
            runs = load("control", gen, cond)
            if not runs:
                continue
            if not ctrl_any:
                P("## Q3 — 대조군(기타+mixed 150, 스킬 미로드 기대)")
                P("")
                P("| gen | cond | C0 내용 | C1 내용 | C2 내용 | C2 스킬 로드율(오라우팅) |")
                P("|---|---|---|---|---|---|")
                ctrl_any = True
            doms = ("기타", "mixed")
            c2rows = [r for d in runs for r in d["rows"] if r["arm"] == "C2"]
            loaded = sum(1 for r in c2rows if r["skill"]) / len(c2rows) if c2rows else 0
            P(f"| {gen} | {clab} | {ms(per_run(runs, 'C0', doms, 'strict'))} "
              f"| {ms(per_run(runs, 'C1', doms, 'strict'))} "
              f"| {ms(per_run(runs, 'C2', doms, 'strict'))} | {loaded:.0%} |")
    if ctrl_any:
        P("")

    # ── Sanity: C0 vs 2번 C(도메인 분해값) ──
    P("## Sanity — C0 vs 2번 자율+게이트(같은 코드 경로, main_* 도메인 분해 재계산)")
    P("")
    P("| gen | cond | 도메인 | 2번 C 내용가짜율 | 3번 C0 내용가짜율 |")
    P("|---|---|---|---|---|")
    import csv as _csv
    labels = {r["question_id"]: r["domain"]
              for r in _csv.DictReader(open(config.RESULTS_DIR / "domain_labels.csv",
                                            encoding="utf-8-sig"))}
    for gen in GENS:
        for cond, clab in CONDS:
            runs3 = load("target", gen, cond)
            if not runs3:
                continue
            old = []
            for rr in RUNS:
                f = config.RESULTS_DIR / f"main_{gen}_{cond}_r{rr}.json"
                if f.exists():
                    old.append(json.load(open(f, encoding="utf-8")))
            for dom in ("형사", "민사"):
                ov = []
                for d in old:
                    rows = [r for r in d["rows"] if r["arm"] == "C" and labels[r["id"]] == dom]
                    ov.append(rates(rows)["strict"])
                P(f"| {gen} | {clab} | {dom} | {ms(ov)} | {ms(per_run(runs3, 'C0', (dom,), 'strict'))} |")
    P("")

    out = "\n".join(L)
    print(out)
    md = config.RESULTS_DIR / "main3_summary.md"
    md.write_text(out + "\n", encoding="utf-8")
    print(f"\n저장: {md}")


if __name__ == "__main__":
    main()

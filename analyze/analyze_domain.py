"""
analyze_domain.py — 2번 본실험 결과의 민사/형사 도메인 분해 + 3번용 도메인 라벨 생성.

  ★ 재측정·재채점 아님: 모델 호출 0, lawcheck 호출 0.
    main_{gen}_{cond}_r{1..3}.json에 저장된 per-question metrics/checks를 읽어
    문항 도메인으로 group-by 하는 순수 재집계 (CPU, 수 분).

  도메인 규칙(결정론·하드코딩, 브리프 명시):
    법령명 정확일치(공백 제거): 민법→민사 / 형법·형사소송법→형사 / 그 외→기타.
    (정확일치라 '군형법'·'민사소송법' 등 유사명은 기타 — 은폐하지 않고 기타 빈도로 보고)
    문항 도메인 = gold 조문 다수결({민사,형사,기타} 3버킷), 동수면 mixed.
    주분석 = 민사·형사만. mixed·기타는 제외하되 건수·목록 보고(은폐 금지).

  출력: results/domain_labels.csv, results/domain_breakdown.md
"""
import collections
import csv
import json
import re

import config
from questions import load_koblex

LAW2DOM = {"민법": "민사", "형법": "형사", "형사소송법": "형사"}
GENS = ["qwen", "exaone"]
CONDS = [("rag", "RAG"), ("norag", "맨몸")]
ARMS = ["A", "B", "C"]
RUNS = (1, 2, 3)


def norm(s):
    return re.sub(r"\s+", "", s or "")


def law_dom(name):
    return LAW2DOM.get(norm(name), "기타")


def label_question(gold):
    laws = [g["law"] for g in gold]
    votes = collections.Counter(law_dom(l) for l in laws)
    if not votes:
        return "기타", "gold없음", False, laws
    top = votes.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        rule = "tie(" + ",".join(f"{d}{c}" for d, c in top) + ")→mixed"
        return "mixed", rule, True, laws
    rule = "majority(" + "/".join(f"{d}{c}" for d, c in top) + ")"
    return top[0][0], rule, len(votes) > 1, laws


def agg_rates(rows):
    """저장된 per-question metrics 합산 → (느슨율, 엄격율, gold율, 인용n, 문항n, 분모들)."""
    c = collections.Counter()
    gh = gt = 0
    for r in rows:
        c += collections.Counter(r["metrics"])
        gh += r["gold_hit"]
        gt += r["gold_tot"]
    ld = c["n"] - c["unc"]
    sd = ld - c["cunk"]
    return {"loose": c["fake"] / ld if ld else None, "strict": c["sfake"] / sd if sd else None,
            "gold": gh / gt if gt else None, "cites": c["n"], "ld": ld, "sd": sd,
            "fake": c["fake"], "sfake": c["sfake"], "q": len(rows), "c": c}


def fmt(v, frac=""):
    return "-" if v is None else f"{v:.1%}{frac}"


def main():
    # ── 1. 문항 도메인 라벨 (gold는 KoBLEX 데이터셋 자체 — 모델 무관) ──
    items = load_koblex(226)
    labels = {}
    for it in items:
        dom, rule, is_mixed, laws = label_question(it["gold"])
        labels[it["id"]] = {"domain": dom, "rule": rule, "is_mixed": is_mixed, "laws": laws}
    lp = config.RESULTS_DIR / "domain_labels.csv"
    with lp.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["question_id", "gold_statutes", "domain", "rule_applied", "is_mixed"])
        for qid, v in labels.items():
            w.writerow([qid, "|".join(v["laws"]), v["domain"], v["rule"], int(v["is_mixed"])])

    cov = collections.Counter(v["domain"] for v in labels.values())
    admin_q = sum(1 for v in labels.values() if any(norm(l).startswith("행정") for l in v["laws"]))
    etc_laws = collections.Counter(norm(l) for v in labels.values() for l in v["laws"]
                                   if law_dom(l) == "기타")

    # ── 2. main_* 로드 (읽기 전용) ──
    data = {}   # (gen, cond) -> {"runs": [file dicts], "rows": pooled rows}
    for gen in GENS:
        for cond, _ in CONDS:
            runs, pooled = [], []
            for r in RUNS:
                f = config.RESULTS_DIR / f"main_{gen}_{cond}_r{r}.json"
                if not f.exists():
                    continue
                d = json.load(open(f, encoding="utf-8"))
                runs.append(d)
                pooled.extend(d["rows"])
            if runs:
                data[(gen, cond)] = {"runs": runs, "rows": pooled}

    # ── sanity ①: 도메인 n 합 = 전체 / ②: 도메인 무시 재합산 = 저장된 arm 요약 재현 ──
    sane = []
    assert sum(cov.values()) == len(items) == 226, "문항 유실!"
    sane.append(f"라벨 합계 {sum(cov.values())}/226 — 유실 0 ✓")
    for (gen, cond), dd in data.items():
        for d in dd["runs"]:
            for a in ARMS:
                rows = [r for r in d["rows"] if r["arm"] == a]
                assert len(rows) == 226, f"{gen}/{cond}/r{d['run']}/{a}: 문항 {len(rows)}"
                got = agg_rates(rows)["loose"]
                want = d["arms"][a]["loose_rate"]
                assert abs(got - want) < 1e-9, f"{gen}/{cond}/r{d['run']}/{a}: {got} != {want}"
    sane.append("전 12파일×3arm 재합산 느슨율 == 저장된 요약과 정확 일치 ✓ (group-by 무손상)")
    unmapped = [qid for qid, v in labels.items() if v["domain"] == "기타" and not v["laws"]]
    sane.append(f"매핑 실패(gold 자체 없음): {len(unmapped)}건 — 전부 '기타'로 잔존, 무단 탈락 0 ✓")

    # ── 3. 도메인 × gen × cond × arm 재집계 ──
    L = []
    P = L.append
    P("# 도메인 분해: 2번 본실험(main_*, 3런 pool)의 민사/형사 재집계 — 재측정 아님, lawcheck v1 판정값 그대로")
    P("")
    P(f"매핑 규칙: 법령명 정확일치 — 민법→민사 / 형법·형사소송법→형사 / 그 외→기타. "
      f"문항 도메인 = gold 다수결, 동수→mixed.")
    P("")
    P(f"## 커버리지 (226문항): 민사 {cov['민사']} ({cov['민사']/226:.0%}) / "
      f"형사 {cov['형사']} ({cov['형사']/226:.0%}) / 기타 {cov['기타']} ({cov['기타']/226:.0%}) / "
      f"mixed {cov['mixed']} ({cov['mixed']/226:.0%})")
    P("")
    P(f"- 행정 관련 문항(gold에 '행정*' 법령 포함): {admin_q}건 — 소수라 주분석 제외(브리프대로).")
    P(f"- mixed 문항 목록: " + ", ".join(q for q, v in labels.items() if v["domain"] == "mixed"))
    P(f"- '기타' 상위 법령(3번 매핑 확장 후보): "
      + ", ".join(f"{l}({c})" for l, c in etc_laws.most_common(10)))
    P("")

    P("## 도메인별 지표 (3런 pool, 인용 단위; n=문항수×3런)")
    P("")
    for gen in GENS:
        for cond, clab in CONDS:
            if (gen, cond) not in data:
                continue
            P(f"### 생성={gen} · {clab}")
            P("")
            P("| arm | 도메인 | 실존가짜율 | 내용가짜율 | gold정확인용 | 인용수(판정가능) | 문항수(pool) |")
            P("|---|---|---|---|---|---|---|")
            for a in ARMS:
                for dom in ("민사", "형사", "기타", "mixed"):
                    rows = [r for r in data[(gen, cond)]["rows"]
                            if r["arm"] == a and labels[r["id"]]["domain"] == dom]
                    if not rows:
                        continue
                    g = agg_rates(rows)
                    P(f"| {a} | {dom} | {fmt(g['loose'])} ({g['fake']}/{g['ld']}) "
                      f"| {fmt(g['strict'])} ({g['sfake']}/{g['sd']}) | {fmt(g['gold'])} "
                      f"| {g['cites']}({g['ld']}) | {g['q']} |")
            P("")

    # ── 4. ★핵심: 내용 가짜율 민사 vs 형사 ──
    P("## ★ 내용 가짜율(내용 오적용): 민사 vs 형사")
    P("")
    P("| 슬라이스 | 민사 | 형사 | 더 심한 쪽 |")
    P("|---|---|---|---|")
    worse_votes = collections.Counter()
    overall = {}
    for dom in ("민사", "형사"):
        rows = [r for dd in data.values() for r in dd["rows"] if labels[r["id"]]["domain"] == dom]
        overall[dom] = agg_rates(rows)
    for gen in GENS:
        for cond, clab in CONDS:
            if (gen, cond) not in data:
                continue
            vals = {}
            for dom in ("민사", "형사"):
                rows = [r for r in data[(gen, cond)]["rows"] if labels[r["id"]]["domain"] == dom]
                vals[dom] = agg_rates(rows)
            worse = "민사" if (vals["민사"]["strict"] or 0) > (vals["형사"]["strict"] or 0) else "형사"
            worse_votes[worse] += 1
            P(f"| {gen}·{clab} (arm 전체) | {fmt(vals['민사']['strict'])} "
              f"({vals['민사']['sfake']}/{vals['민사']['sd']}) | {fmt(vals['형사']['strict'])} "
              f"({vals['형사']['sfake']}/{vals['형사']['sd']}) | {worse} |")
    P(f"| **전체 pool** | **{fmt(overall['민사']['strict'])}** "
      f"({overall['민사']['sfake']}/{overall['민사']['sd']}) | **{fmt(overall['형사']['strict'])}** "
      f"({overall['형사']['sfake']}/{overall['형사']['sd']}) | "
      f"**{'민사' if overall['민사']['strict'] > overall['형사']['strict'] else '형사'}** |")
    P("")

    # ── 5. 교차표: 문항 도메인 × 인용 조문 도메인 (오적용 직격 신호) ──
    P("## 교차표: 문항 도메인 × 인용된 조문의 도메인 (전 arm·gen·런 pool; 괄호=그중 내용불일치)")
    P("")
    for cond, clab in CONDS:
        xt = collections.defaultdict(collections.Counter)
        mism = collections.defaultdict(collections.Counter)
        for (gen, c2), dd in data.items():
            if c2 != cond:
                continue
            for r in dd["rows"]:
                qd = labels[r["id"]]["domain"]
                if qd not in ("민사", "형사"):
                    continue
                for ck in r["checks"]:
                    cited = ck.get("official") or (ck.get("cite") or "").rsplit(" ", 1)[0]
                    cd = law_dom(cited)
                    xt[qd][cd] += 1
                    if ck.get("status") == "real" and ck.get("content") == "mismatch":
                        mism[qd][cd] += 1
        P(f"### {clab}")
        P("")
        P("| 문항\\인용 | 민사(민법) | 형사(형법·형소법) | 기타 | 교차인용율(대각선 밖 민사↔형사) |")
        P("|---|---|---|---|---|")
        for qd in ("민사", "형사"):
            row = xt[qd]
            tot = sum(row.values())
            offdom = row["형사"] if qd == "민사" else row["민사"]
            P(f"| {qd} 문항 | {row['민사']} ({mism[qd]['민사']}) | {row['형사']} ({mism[qd]['형사']}) "
              f"| {row['기타']} ({mism[qd]['기타']}) | {offdom}/{tot} = {offdom/tot:.1%} |")
        P("")

    # ── 6. sanity & 한계 & 판정 ──
    P("## Sanity check")
    P("")
    for s in sane:
        P(f"- {s}")
    P("")
    P("## 정직한 한계")
    P("")
    P("- 도메인 분해로 셀당 n 축소(특히 형사 쪽 소수 슬라이스) → 이 분해는 **방향**만 신뢰. "
      "3번 본실험은 도메인별 충분한 n 별도 확보 필요.")
    P("- 매핑이 보수적(정확일치 3개 법령만) → '기타'가 큼. 3번에서 매핑 확장 시 위 '기타 상위 법령' 참조.")
    P("")
    o_min, o_hyeong = overall["민사"]["strict"], overall["형사"]["strict"]
    worse = "민사" if o_min > o_hyeong else "형사"
    P(f"## 판정 한 줄")
    P("")
    P(f"**내용 오적용은 {worse}에서 더 심함 — 전체 pool 내용가짜율 민사 {o_min:.1%} vs 형사 {o_hyeong:.1%} "
      f"(슬라이스별 다수결 {worse_votes['민사']}:{worse_votes['형사']}) → 3번 스킬 우선순위 = {worse} 스킬부터.**")

    out = "\n".join(L)
    print(out)
    mp = config.RESULTS_DIR / "domain_breakdown.md"
    mp.write_text(out + "\n", encoding="utf-8")
    print(f"\n저장: {lp}\n      {mp}")


if __name__ == "__main__":
    main()

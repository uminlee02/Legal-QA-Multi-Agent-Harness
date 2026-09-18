"""fourway.py — 4-way 비교표 (조건 ④ RAG-근거 논리검증 추가).

입력(재측정 아님, 기존 결과 JSON 재사용):
  --both     : lg_eval_n226_both.json     → ①순수생성(baseline) + ③풀스택-근거없는검증(proposed)
  --ragonly  : lg_eval_ragonly_n226.json  → ②RAG-only(baseline)
  --ragverify: lg_eval_ragverify_n226.json→ ④풀스택-근거기반검증(proposed)

출력: 4-way 표 + 핵심 감소율(③→④ 검증에 근거준 효과, ②→④ 근거검증이 RAG단독보다 나은가) +
      재생성/내용불일치 ③ vs ④ 비교 + results/lg_4way.{json,csv}
"""
import sys
import csv
import json
import pathlib
import argparse

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent


def load(path, key):
    d = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    return d["conditions"][key], d.get("n_done")


def pct(x):
    return f"{x*100:.1f}%"


def red(a, b):
    return (a - b) / a * 100 if a else 0.0


def goldpct(c):
    return c["gold_hit"] / c["gold_tot"] * 100 if c["gold_tot"] else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--both", default=str(_ROOT / "results/lg_eval_n226_both.json"))
    ap.add_argument("--ragonly", default=str(_ROOT / "results/lg_eval_ragonly_n226.json"))
    ap.add_argument("--ragverify", default=str(_ROOT / "results/lg_eval_ragverify_n226.json"))
    ap.add_argument("--out-prefix", default=str(_ROOT / "results/lg_4way"))
    args = ap.parse_args()

    c1, n1 = load(args.both, "baseline")       # ① 순수 생성
    c2, n2 = load(args.ragonly, "baseline")    # ② RAG-only
    c3, n3 = load(args.both, "proposed")        # ③ 풀스택 (근거 없는 검증)
    c4, n4 = load(args.ragverify, "proposed")   # ④ 풀스택 (근거 기반 검증)

    rows = [("① 순수 생성", c1), ("② RAG-only", c2),
            ("③ 풀스택(근거X 검증)", c3), ("④ 풀스택(근거O 검증)", c4)]

    print(f"\n{'='*82}")
    print("4-WAY 비교 (KoBLEX | gen=Qwen3.5-27B | logic=DeepSeek-R1-32B | 채점=법제처 실제 API + lawcheck v1)")
    print(f"n: ①③={n1} · ②={n2} · ④={n4}")
    print(f"{'='*82}")
    print(f"{'조건':<20} {'느슨':>9} {'엄격':>9} {'내용불일치':>10} {'gold recall':>14} {'재생성':>7}")
    print("-" * 82)
    for name, c in rows:
        regen = c.get("retry_sum", 0)
        print(f"{name:<20} {pct(c['loose']):>9} {pct(c['strict']):>9} {c['mism']:>10} "
              f"{c['gold_hit']}/{c['gold_tot']}({goldpct(c):.0f}%)".rjust(14) + f"{regen:>8}")

    print(f"\n{'핵심 감소율 (느슨 / 엄격 가짜율)'}")
    print("-" * 82)
    steps = [("③→④ (검증에 근거를 준 효과)", c3, c4),
             ("②→④ (근거기반 검증이 RAG단독보다?)", c2, c4),
             ("①→④ (전체)", c1, c4)]
    reductions = {}
    for label, a, b in steps:
        rl, rs = red(a["loose"], b["loose"]), red(a["strict"], b["strict"])
        reductions[label] = {"loose_pct": rl, "strict_pct": rs}
        print(f"{label:<36} 느슨 {pct(a['loose'])}→{pct(b['loose'])}={rl:+.1f}%  |  "
              f"엄격 {pct(a['strict'])}→{pct(b['strict'])}={rs:+.1f}%")

    # ③ vs ④ 부작용 관찰
    print(f"\n[③ vs ④ 재생성 부작용 관찰]")
    print(f"  내용불일치: ③ {c3['mism']} → ④ {c4['mism']}  | 재생성합: ③ {c3.get('retry_sum',0)} → ④ {c4.get('retry_sum',0)}")
    print(f"  → 근거 주입이 재생성 부작용(내용불일치)을 {'줄임' if c4['mism'] < c3['mism'] else ('늘림' if c4['mism']>c3['mism'] else '변화없음')}")

    out = {"benchmark": "KoBLEX", "gen": "Qwen3.5-27B", "logic": "DeepSeek-R1-Distill-Qwen-32B",
           "scorer": "법제처 Open API(real) + lawcheck v1",
           "conditions": {"1_pure": c1, "2_ragonly": c2, "3_full_ungrounded": c3, "4_full_grounded": c4},
           "reductions": reductions}
    pathlib.Path(args.out_prefix + ".json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(args.out_prefix + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["condition", "loose_fake", "strict_fake", "content_mismatch",
                    "gold_hit", "gold_tot", "gold_recall_pct", "regen_sum"])
        for name, c in rows:
            w.writerow([name, f"{c['loose']:.4f}", f"{c['strict']:.4f}", c["mism"],
                        c["gold_hit"], c["gold_tot"], f"{goldpct(c):.1f}", c.get("retry_sum", 0)])
    print(f"\n저장: {args.out_prefix}.json / .csv")


if __name__ == "__main__":
    main()

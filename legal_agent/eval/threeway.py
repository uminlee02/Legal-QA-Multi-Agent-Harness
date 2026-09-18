"""threeway.py — 3-way 비교표 구성 (RAG 효과와 검증/멀티에이전트 효과 분리).

입력(재측정 아님, 기존 결과 JSON 재사용):
  --both    : lg_eval_n226_both.json  → baseline(①순수생성) + proposed(③풀스택)
  --ragonly : lg_eval_ragonly_*.json  → baseline(②RAG-only)

출력: 3-way 표 + 감소율 3종(①→② RAG효과 / ②→③ 검증효과 / ①→③ 전체) + results/lg_3way.{json,csv}

  python eval/threeway.py --both results/lg_eval_n226_both.json --ragonly results/lg_eval_ragonly_n226.json
"""
import sys
import csv
import json
import pathlib
import argparse

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent


def load_cond(path, key):
    d = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    return d["conditions"][key], d.get("n_done")


def pct(x):
    return f"{x*100:.1f}%"


def red(a, b):
    """감소율 (a→b): (a-b)/a×100."""
    return (a - b) / a * 100 if a else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--both", default=str(_ROOT / "results/lg_eval_n226_both.json"))
    ap.add_argument("--ragonly", default=str(_ROOT / "results/lg_eval_ragonly_n226.json"))
    ap.add_argument("--out-prefix", default=str(_ROOT / "results/lg_3way"))
    args = ap.parse_args()

    c1, n1 = load_cond(args.both, "baseline")      # ① 순수 생성
    c2, n2 = load_cond(args.ragonly, "baseline")   # ② RAG-only
    c3, n3 = load_cond(args.both, "proposed")       # ③ 풀스택(제안)

    rows = [("① 순수 생성", c1), ("② RAG-only", c2), ("③ 풀스택(제안)", c3)]

    def goldpct(c):
        return c["gold_hit"] / c["gold_tot"] * 100 if c["gold_tot"] else 0.0

    # ── 표 출력 ──
    print(f"\n{'='*78}")
    print(f"3-WAY 비교 (KoBLEX | gen=Qwen3.5-27B | 채점=법제처 실제 API + frozen lawcheck v1)")
    print(f"n: 순수/풀스택={n1} · RAG-only={n2}")
    print(f"{'='*78}")
    hdr = f"{'조건':<16} {'느슨가짜율':>12} {'엄격가짜율':>12} {'내용불일치':>10} {'gold recall':>14}"
    print(hdr)
    print("-" * 78)
    for name, c in rows:
        print(f"{name:<16} {pct(c['loose']):>12} {pct(c['strict']):>12} "
              f"{c['mism']:>10} {c['gold_hit']}/{c['gold_tot']}({goldpct(c):.0f}%)".rjust(0))

    # ── 감소율 ──
    print(f"\n{'감소율 (느슨 / 엄격 가짜율)':<40}")
    print("-" * 78)
    steps = [("①→② (RAG 순효과)", c1, c2),
             ("②→③ (검증·멀티에이전트 순효과)", c2, c3),
             ("①→③ (전체 효과)", c1, c3)]
    reductions = {}
    for label, a, b in steps:
        rl, rs = red(a["loose"], b["loose"]), red(a["strict"], b["strict"])
        reductions[label] = {"loose_pct": rl, "strict_pct": rs,
                             "loose_from": a["loose"], "loose_to": b["loose"],
                             "strict_from": a["strict"], "strict_to": b["strict"]}
        print(f"{label:<34} 느슨 {pct(a['loose'])}→{pct(b['loose'])} = {rl:+.1f}%  |  "
              f"엄격 {pct(a['strict'])}→{pct(b['strict'])} = {rs:+.1f}%")

    # ── 저장 ──
    out = {"benchmark": "KoBLEX", "gen": "Qwen3.5-27B", "scorer": "법제처 Open API(real) + lawcheck v1",
           "n": {"pure/full": n1, "ragonly": n2},
           "conditions": {"1_pure": c1, "2_ragonly": c2, "3_full": c3},
           "reductions": reductions}
    jpath = pathlib.Path(args.out_prefix + ".json")
    jpath.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    cpath = pathlib.Path(args.out_prefix + ".csv")
    with open(cpath, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["condition", "loose_fake", "strict_fake", "content_mismatch",
                    "gold_hit", "gold_tot", "gold_recall_pct", "fake", "ld", "sfake", "sd"])
        for name, c in rows:
            w.writerow([name, f"{c['loose']:.4f}", f"{c['strict']:.4f}", c["mism"],
                        c["gold_hit"], c["gold_tot"], f"{goldpct(c):.1f}",
                        c["fake"], c["ld"], c["sfake"], c["sd"]])
    print(f"\n저장: {jpath}\n     {cpath}")


if __name__ == "__main__":
    main()

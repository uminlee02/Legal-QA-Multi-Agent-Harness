"""compare6.py — ①②③⑥ 비교 + ③→⑥/②→⑥ 감소율 + 전이(고침/깸) + 편집 국소성.

기존 ①②③ 재사용, ⑥(타겟편집)만 신규. GPU 불필요.
  python eval/compare6.py
"""
import sys
import csv
import json
import pathlib

_HERE = pathlib.Path(__file__).resolve().parent
for p in (str(_HERE.parent), str(_HERE.parent.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)
R = pathlib.Path("/root/legal-citation-faithfulness/results")

FILES = {
    "①순수":   ("lg_eval_n226_both.json", "baseline"),
    "②RAGonly": ("lg_eval_ragonly_n226.json", "baseline"),
    "③풀스택-재생성": ("lg_eval_n226_both.json", "proposed"),
    "⑥풀스택-타겟편집": ("lg_eval_targeted_n226.json", "proposed"),
}


def cond(path, key):
    return json.loads((R / path).read_text(encoding="utf-8"))["conditions"][key]


def per(path, key):
    return {r["id"]: r[key] for r in json.loads((R / path).read_text(encoding="utf-8"))["per_item"] if key in r}


def pctstr(c):
    return f"{c['loose']*100:.1f}% / {c['strict']*100:.1f}%"


def gp(c):
    return c["gold_hit"] / c["gold_tot"] * 100 if c["gold_tot"] else 0


def red(a, b):
    return (a - b) / a * 100 if a else 0.0


def transitions(A, B):
    """A→B 엄격가짜 전이. dirty=sfake>0."""
    ids = [i for i in A if i in B]
    broke = [i for i in ids if A[i]["sfake"] == 0 and B[i]["sfake"] > 0]
    fixed = [i for i in ids if A[i]["sfake"] > 0 and B[i]["sfake"] == 0]
    return len(fixed), len(broke), fixed, broke


def main():
    C = {k: cond(*v) for k, v in FILES.items()}

    print("=" * 84)
    print("①②③⑥ 비교 (KoBLEX n=226 | gen=Qwen3.5-27B | edit=DeepSeek-R1-32B | 법제처 실제 API)")
    print("=" * 84)
    print(f"{'조건':<18}{'느슨/엄격 가짜율':>16}{'내용불일치':>10}{'gold recall':>16}{'수정횟수':>9}")
    print("-" * 84)
    for k in FILES:
        c = C[k]
        gold = f"{c['gold_hit']}/{c['gold_tot']}({gp(c):.0f}%)"
        print(f"{k:<18}{pctstr(c):>16}{c['mism']:>10}{gold:>16}{c.get('retry_sum',0):>9}")

    c2, c3, c6 = C["②RAGonly"], C["③풀스택-재생성"], C["⑥풀스택-타겟편집"]
    print("\n핵심 감소율 (느슨 / 엄격)")
    print("-" * 84)
    for label, a, b in [("③→⑥ (재생성→타겟편집 교체 효과)", c3, c6),
                        ("②→⑥ (타겟편집 멀티에이전트 vs RAG단독)", c2, c6),
                        ("①→⑥ (전체)", C["①순수"], c6)]:
        print(f"  {label:<36} 느슨 {a['loose']*100:.1f}→{b['loose']*100:.1f}%={red(a['loose'],b['loose']):+.1f}%"
              f"  | 엄격 {a['strict']*100:.1f}→{b['strict']*100:.1f}%={red(a['strict'],b['strict']):+.1f}%")

    # 전이 분석
    p2 = per(*FILES["②RAGonly"]); p3 = per(*FILES["③풀스택-재생성"]); p6 = per(*FILES["⑥풀스택-타겟편집"])
    print("\n전이 분석 (② 대비: ②가 맞힌 걸 깨는가 / 틀린 걸 고치는가)")
    print("-" * 84)
    for name, P in [("③ 재생성", p3), ("⑥ 타겟편집", p6)]:
        fx, br, _, _ = transitions(p2, P)
        print(f"  ②→{name:<10}: 고침 {fx} − 깸 {br} = 순 {fx-br:+d}")

    # ⑥ 편집 국소성
    edits = [r for r in p6.values() if r.get("n_edits")]
    if edits:
        import statistics as st
        ef = [r["edit_fraction_mean"] for r in edits]
        print("\n⑥ 편집 국소성 (전체 재생성 대비 얼마나 국소적인가)")
        print("-" * 84)
        print(f"  편집 발생 문항: {len(edits)}/{len(p6)} | 평균 편집률(1=전체교체): {st.mean(ef):.3f} "
              f"(중앙값 {st.median(ef):.3f}) | 편집줄 합 {sum(r.get('lines_edited',0) for r in edits)} "
              f"삭제줄 합 {sum(r.get('lines_deleted',0) for r in edits)}")

    # 저장
    out = {"conditions": {k: C[k] for k in FILES},
           "reductions": {
               "3to6": {"loose": red(c3["loose"], c6["loose"]), "strict": red(c3["strict"], c6["strict"])},
               "2to6": {"loose": red(c2["loose"], c6["loose"]), "strict": red(c2["strict"], c6["strict"])}},
           "transitions_vs2": {
               "3": dict(zip(("fixed", "broke"), transitions(p2, p3)[:2])),
               "6": dict(zip(("fixed", "broke"), transitions(p2, p6)[:2]))}}
    (R / "lg_compare6.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(R / "lg_compare6.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["condition", "loose", "strict", "mism", "gold_hit", "gold_tot", "fix_count"])
        for k in FILES:
            c = C[k]
            w.writerow([k, f"{c['loose']:.4f}", f"{c['strict']:.4f}", c["mism"],
                        c["gold_hit"], c["gold_tot"], c.get("retry_sum", 0)])
    print(f"\n저장: {R/'lg_compare6.json'} / .csv")


if __name__ == "__main__":
    main()

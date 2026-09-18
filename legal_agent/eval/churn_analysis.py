"""churn_analysis.py — ②(RAG-only) vs ③(풀스택) 문항별 대조: 재생성이 멀쩡한 인용을 깨는가?

기존 결과 JSON만 사용(GPU 불필요). 답변 텍스트는 저장 안 돼 있어 '어떤 조문이 깨졌는지'까진
못 보지만, (1) ②는 맞았는데 ③에서 틀어진 문항, (2) 그 문항들의 ③ 재생성 횟수,
(3) ③ 내부에서 재생성한 문항 vs 안 한 문항의 오류율을 계량한다.

  python eval/churn_analysis.py
"""
import re
import sys
import json
import pathlib

_HERE = pathlib.Path(__file__).resolve().parent
for p in (str(_HERE.parent), str(_HERE.parent.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

R = pathlib.Path("/root/legal-citation-faithfulness/results")


def load_items(path, key):
    d = json.loads((R / path).read_text(encoding="utf-8"))
    return {r["id"]: r[key] for r in d["per_item"] if key in r}


def q_short(prompt):
    m = re.search(r"\[질문\]\s*(.+)", prompt or "")
    q = (m.group(1) if m else prompt or "").strip().replace("\n", " ")
    return q[:52]


def main():
    two = load_items("lg_eval_ragonly_n226.json", "baseline")   # ② RAG-only
    three = load_items("lg_eval_n226_both.json", "proposed")     # ③ 풀스택
    ids = [i for i in two if i in three]

    # KoBLEX 질문 (문항 주제 표시용)
    try:
        from questions import load_koblex
        qmap = {it["id"]: q_short(it["prompt"]) for it in load_koblex(None)}
    except Exception:
        qmap = {}

    def dirty(r):        # 엄격 가짜(미실존 or 내용불일치)가 하나라도?
        return r["sfake"] > 0
    def mism(r):
        return r["mism"] > 0

    broke, fixed, both_bad, both_ok = [], [], [], []
    for i in ids:
        a, b = two[i], three[i]
        if not dirty(a) and dirty(b):
            broke.append(i)
        elif dirty(a) and not dirty(b):
            fixed.append(i)
        elif dirty(a) and dirty(b):
            both_bad.append(i)
        else:
            both_ok.append(i)

    print("=" * 78)
    print(f"② RAG-only  vs  ③ 풀스택(+검증+재생성)  — 문항별 엄격가짜 전이 (n={len(ids)})")
    print("=" * 78)
    print(f"  양쪽 clean        : {len(both_ok)}")
    print(f"  ② dirty→③ clean (③이 고침): {len(fixed)}")
    print(f"  ② clean→③ dirty (③이 깨뜨림): {len(broke)}   ← 핵심")
    print(f"  양쪽 dirty        : {len(both_bad)}")
    print(f"  ⇒ 순효과: 고친 {len(fixed)} − 깨뜨린 {len(broke)} = {len(fixed)-len(broke):+d}")

    def retry_of(i):
        return three[i].get("retry", 0)

    br_regen = sum(1 for i in broke if retry_of(i) > 0)
    fx_regen = sum(1 for i in fixed if retry_of(i) > 0)
    print(f"\n  '③이 깨뜨린' {len(broke)}건 중 재생성(retry>0) 발생: {br_regen}건")
    print(f"  '③이 고친'   {len(fixed)}건 중 재생성 발생: {fx_regen}건")

    print(f"\n[② clean → ③ dirty 문항 목록] (id | ③재생성 | ③내용불일치 | 질문)")
    print("-" * 78)
    for i in sorted(broke, key=lambda x: -retry_of(x)):
        b = three[i]
        print(f"  {i:<20} r={b.get('retry',0)} mism={b['mism']} sfake={b['sfake']}  {qmap.get(i,'')}")

    # ③ 내부: 재생성한 문항 vs 안 한 문항의 오류율 (선택편향 주의)
    reg = [i for i in ids if retry_of(i) > 0]
    nreg = [i for i in ids if retry_of(i) == 0]
    def rate(group, f):
        n = sum(1 for i in group if f(three[i]))
        return n, len(group), (n/len(group)*100 if group else 0)
    print(f"\n[③ 내부: 재생성 문항 vs 단발 문항의 내용불일치율]")
    print("-" * 78)
    n1, d1, p1 = rate(reg, mism)
    n2, d2, p2 = rate(nreg, mism)
    print(f"  재생성함(retry>0) : 내용불일치 {n1}/{d1} = {p1:.1f}%")
    print(f"  단발(retry=0)     : 내용불일치 {n2}/{d2} = {p2:.1f}%")
    print(f"  (주의: 재생성은 '문제가 감지돼서' 유발되므로 선택편향 존재 — 인과 아님, 연관.)")

    # 총 내용불일치 합 sanity
    print(f"\n[sanity] 내용불일치 합: ② {sum(two[i]['mism'] for i in ids)} / ③ {sum(three[i]['mism'] for i in ids)}")
    print(f"         엄격가짜 합:   ② {sum(two[i]['sfake'] for i in ids)} / ③ {sum(three[i]['sfake'] for i in ids)}")


if __name__ == "__main__":
    main()

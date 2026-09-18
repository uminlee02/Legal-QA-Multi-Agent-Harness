"""rag_recall.py — 검색 recall@k 측정(LLM 불필요, KoE5 검색기만).

각 k에 대해: KoBLEX gold 조문이 KoE5 top-k 검색 결과에 몇 개나 들어오는가(= 정확인용의 천장).
gold recall(생성기가 실제 인용) 과 비교하면 '검색 천장 vs 모델 활용'을 분리할 수 있다.

  python eval/rag_recall.py --n 226 --ks 3,5,10
"""
import re
import sys
import json
import pathlib
import argparse

_HERE = pathlib.Path(__file__).resolve().parent
for p in (str(_HERE.parent), str(_HERE.parent.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from lawcheck import _norm
from questions import load_koblex

_ART = re.compile(r"^(.*?)\s*(\d+)\s*조(?:의\s*(\d+))?")


def prov_key(hierarchy):
    m = _ART.match(hierarchy or "")
    if not m:
        return None
    return (_norm(m.group(1).strip()), int(m.group(2)), int(m.group(3) or 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--ks", default="3,5,10")
    ap.add_argument("--out", default=str(_HERE.parent.parent / "results/lg_rag_recall.json"))
    args = ap.parse_args()
    ks = [int(x) for x in args.ks.split(",")]

    from demo_cli import Retriever
    rt = Retriever()
    items = load_koblex(args.n)
    kmax = max(ks)

    hit = {k: 0 for k in ks}
    tot = 0
    per = []
    for it in items:
        goldset = {(_norm(g["law"]), g["jo"], g["branch"]) for g in it["gold"]}
        tot += len(goldset)
        provs = rt.search(it["prompt"], k=kmax)            # eval 과 동일 쿼리(prompt), kmax 후 접두부분집합
        pkeys = [prov_key(p["hierarchy"]) for p in provs]
        row = {"id": it["id"], "n_gold": len(goldset)}
        for k in ks:
            topk = set(pk for pk in pkeys[:k] if pk)
            h = sum(1 for g in goldset if g in topk)
            hit[k] += h
            row[f"hit@{k}"] = h
        per.append(row)

    print(f"\n{'='*56}")
    print(f"검색 recall@k (KoE5 | KoBLEX gold | n={len(items)}, gold조문 {tot}개)")
    print(f"{'='*56}")
    print(f"{'k':>4} {'recall@k':>12} {'(hit/gold)':>14}")
    for k in ks:
        print(f"{k:>4} {hit[k]/tot*100:>11.1f}% {f'{hit[k]}/{tot}':>14}")
    print("\n※ gold recall(생성 인용)이 recall@k 에 가까우면 '모델이 검색천장을 잘 활용', "
          "\n  recall@k↑인데 gold recall 정체면 '검색은 됐으나 모델이 인용 안 함'.")

    out = {"n": len(items), "gold_total": tot,
           "recall_at_k": {k: {"hit": hit[k], "recall": hit[k]/tot} for k in ks},
           "per_item": per}
    pathlib.Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()

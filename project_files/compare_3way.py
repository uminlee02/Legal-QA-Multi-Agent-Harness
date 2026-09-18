"""
compare_3way.py — 맨몸 vs 사후검증 vs 사전검색(RAG) 3-way 비교표.

  맨몸(bare)      = koblex_n226.json 의 OFF
  사후검증(post)  = koblex_n226.json 의 ON  (before/after 피드백)
  사전검색(RAG)   = koblex_rag.json  의 OFF
  (보너스) RAG+사후검증 = koblex_rag.json 의 ON
"""
import json
import collections
import config


def load(name):
    return json.load(open(config.RESULTS_DIR / name, encoding="utf-8"))


def loose(c):
    return c["fake"] / c["ld"] if c["ld"] else 0.0


def strict(c):
    return c["sfake"] / c["sd"] if c["sd"] else 0.0


def cov(c):
    return c["sd"] / c["ld"] if c["ld"] else 0.0


def gold_recall_from_rows(d, which="off"):
    """rows 의 checks_{which} 에서 gold 정확 인용율 재계산 (있을 때만)."""
    import re
    def norm(s): return re.sub(r"\s+", "", s or "")
    # gold 는 별도 저장 안 했으므로 summary 의 gold_recall(off 기준)만 신뢰. off 외엔 None.
    return None


def main():
    base = load("koblex_n226.json")
    try:
        rag = load("koblex_rag.json")
    except FileNotFoundError:
        print("koblex_rag.json 아직 없음 — RAG 실행 완료 후 다시.")
        return

    conds = [
        ("맨몸(검증X·RAG X)", base["off"], base["gold_recall"]),
        ("사후검증(피드백)", base["on"], None),
        ("사전검색 RAG", rag["off"], rag["gold_recall"]),
        ("RAG+사후검증", rag["on"], None),
    ]
    n = base["n"]
    print(f"\n{'='*92}\nKoBLEX n={n} | {base['model']} | 3-way (페어드/seed42 결정적)\n{'='*92}")
    hdr = f"{'조건':<20}{'인용':>5}{'느슨가짜율':>12}{'엄격가짜율':>12}{'내용불일치':>9}{'gold정확인용':>14}{'엄격커버':>9}"
    print(hdr)
    print("-" * 92)
    for name, c, gr in conds:
        c = collections.Counter(c)
        grs = f"{gr[0]}/{gr[1]}={gr[0]/gr[1]:.1%}" if gr else "—"
        print(f"{name:<20}{c['n']:>5}{loose(c):>11.1%}{strict(c):>12.1%}"
              f"{c['mism']:>9}{grs:>14}{cov(c):>8.0%}")
    print("-" * 92)
    # 핵심 델타
    b, r = collections.Counter(base["off"]), collections.Counter(rag["off"])
    print("\n[맨몸 → 사전검색(RAG) 효과]")
    print(f"  느슨 가짜율 : {loose(b):.1%} → {loose(r):.1%}")
    print(f"  엄격 가짜율 : {strict(b):.1%} → {strict(r):.1%}")
    print(f"  내용 불일치 : {b['mism']} → {r['mism']} 건")
    bg, rg = base["gold_recall"], rag["gold_recall"]
    print(f"  gold 정확인용: {bg[0]/bg[1]:.1%} ({bg[0]}/{bg[1]}) → {rg[0]/rg[1]:.1%} ({rg[0]}/{rg[1]})")
    print(f"  (검색 상한 recall@5 = 58.3% 조문단위 / hit@5 82.3% 질문단위)")


if __name__ == "__main__":
    main()

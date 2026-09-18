"""live_test.py — OC 키로 법제처 라이브 검증 (vLLM 불필요).

  LAW_OC="발급키" python live_test.py
"""
import config
from lawcheck import extract_citations, LawVerifier

CASES = [
    ("민법 제750조", "real"),        # 불법행위 손해배상 — 실존해야 함
    ("형법 제9999조", "fake"),       # 미존재 조문 — 가짜로 잡혀야 함
    ("근로기준법 제60조", "real"),    # 연차 유급휴가 — 실존해야 함
]


def main():
    print(f"OC set: {config.oc_is_set()} (tail=...{config.OC[-4:]})\n")
    v = LawVerifier()
    ok = True
    for text, expect in CASES:
        c = extract_citations(text)[0]
        r = v.verify(c)
        passed = (r.status == expect)
        ok = ok and passed
        print(f"[{'PASS' if passed else 'FAIL'}] {text:<16} → {r.symbol} {r.status:<9}"
              f"(기대 {expect}) | {r.official_name or '-'} | {r.note}")
    print(f"\nlive_calls={v.live_calls}  cache_hits={v.cache_hits}")
    print("✓ 라이브 API 검증 통과 — vLLM 올려도 됨" if ok else "✗ 일부 실패 — 점검 필요")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

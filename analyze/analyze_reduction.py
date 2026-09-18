"""
analyze_reduction.py — 인용 감소 분석 (추가 실행 없이 기존 결과 JSON만 사용).

검증ON에서 총 인용이 baseline 85 → 63으로 줄었다. 이 감소가
  (a) 원래 가짜(환각)를 제거한 것인가  → "검증이 환각을 고친다"
  (b) 실존하는 유효 인용을 버린 것인가  → "검증이 침묵으로 회피한다"
를 baseline 인용 집합 vs verify-ON 인용 집합을 질문별로 대조해 분류한다.
추가로 (c) 검증ON이 새로 만든 환각, (d) 못 고치고 남은 환각도 본다.
"""
import json
import re
import collections

B = json.load(open("results/kcl_baseline.json", encoding="utf-8"))
V = json.load(open("results/kcl_verifyON.json", encoding="utf-8"))


def art_key(cite: str) -> str:
    m = re.search(r"제\s*\d+\s*조(?:의\s*\d+)?(?:\s*제\s*\d+\s*항)?(?:\s*제\s*\d+\s*호)?", cite)
    return re.sub(r"\s+", "", m.group(0)) if m else re.sub(r"\s+", "", cite)


def law_key(ch: dict) -> str:
    # 실존(real)은 법제처가 준 official 사용(안정적). 그 외엔 cite에서 법령명 파싱.
    law = ch.get("official") or ""
    if not law:
        m = re.match(r"(.*?)\s*제\s*\d+\s*조", ch["cite"])
        law = (m.group(1) if m else ch["cite"]).strip()
    return re.sub(r"\s+", "", law)


def index_run(run):
    """idx -> { (law,art) : status }"""
    d = {}
    for r in run["rows"]:
        m = {}
        for ch in r["checks"]:
            m[(law_key(ch), art_key(ch["cite"]))] = ch["status"]
        d[r["idx"]] = m
    return d


BI, VI = index_run(B), index_run(V)

dropped, added, kept = [], [], []
for idx, bk in BI.items():
    vk = VI.get(idx, {})
    for k, st in bk.items():
        (dropped if k not in vk else kept).append((idx, k, st, vk.get(k)))
    for k, st in vk.items():
        if k not in bk:
            added.append((idx, k, st, None))

nb = sum(len(x) for x in BI.values())
nv = sum(len(x) for x in VI.values())
dc = collections.Counter(st for _, _, st, _ in dropped)
ac = collections.Counter(st for _, _, st, _ in added)
kc = collections.Counter((st, vst) for _, _, st, vst in kept)

print(f"baseline 인용 {nb} | verify-ON 인용 {nv} | 순감소 {nb - nv}")
print(f"DROPPED(baseline⊃, verifyON에 없음) {len(dropped)} : {dict(dc)}")
print(f"ADDED(verifyON 신규)              {len(added)} : {dict(ac)}")
print(f"KEPT(양쪽 공통)                   {len(kept)}")

print("\n=== 핵심 분류 ===")
drop_fake = [x for x in dropped if x[2] == "fake"]
drop_real = [x for x in dropped if x[2] == "real"]
drop_unc = [x for x in dropped if x[2] == "uncertain"]
add_fake = [x for x in added if x[2] == "fake"]   # added 튜플은 (idx,k,st,None) → st=x[2]
kept_fake_fake = [x for x in kept if x[2] == "fake" and x[3] == "fake"]
print(f"(a) 환각 제거 (baseline fake가 사라짐)      : {len(drop_fake)}")
print(f"(b) 침묵 회피 (baseline real을 버림)         : {len(drop_real)}")
print(f"    + baseline uncertain 버림                : {len(drop_unc)}")
print(f"(c) 새 환각 (verifyON이 새로 만든 fake)      : {len(add_fake)}")
print(f"(d) 못 고친 환각 (양쪽 다 fake로 잔존)       : {len(kept_fake_fake)}")

print("\n-- (a) 제거된 환각 --")
for idx, k, *_ in drop_fake:
    print(f"   Q{idx} ✗→없음  {k[0]} {k[1]}")
print("\n-- (b) 버려진 실존 인용 (침묵회피) --")
for idx, k, *_ in drop_real:
    print(f"   Q{idx} ✓→없음  {k[0]} {k[1]}")
print("\n-- (c) 새로 생긴 환각 --")
for idx, k, *_ in add_fake:
    print(f"   Q{idx} 없음→✗  {k[0]} {k[1]}")
print("\n-- (d) 못 고치고 남은 환각 --")
for idx, k, *_ in kept_fake_fake:
    print(f"   Q{idx} ✗→✗    {k[0]} {k[1]}")

# baseline 13개 fake의 운명 (인용-동일성 추적 → 비결정성에 강건)
print("\n=== baseline 가짜 13건의 운명 (robust) ===")
fate = collections.Counter()
for idx, bk in BI.items():
    for k, st in bk.items():
        if st == "fake":
            vst = VI.get(idx, {}).get(k)
            fate["사라짐(제거)" if vst is None else f"잔존({vst})"] += 1
print("  ", dict(fate))

# ── 노이즈 바닥 분리 ──────────────────────────────────────────────────────
# baseline에 fake가 없던 질문은 verify-ON이 '동일 프롬프트(피드백 0)'로 재생성하므로
# 이론상 인용이 같아야 한다. 그래도 churn이 있으면 그건 순수 temp=0 비재현성(노이즈).
fake_q = {idx for idx, bk in BI.items() if any(st == "fake" for st in bk.values())}
nofake_q = set(BI) - fake_q


def part(lst):
    return ([x for x in lst if x[0] in nofake_q], [x for x in lst if x[0] in fake_q])


dn, df = part(dropped)
an, af = part(added)


def rc(lst, s):
    return sum(1 for x in lst if x[2] == s)


print(f"\n=== 노이즈(비결정성) vs 피드백 분리 ===")
print(f"fake 있던 질문 {sorted(fake_q)} ({len(fake_q)}개) / fake 없던 질문 {len(nofake_q)}개")
print(f"[노이즈 바닥] fake 없던 질문(피드백 0)인데도 발생한 churn:")
print(f"   dropped {len(dn)} (real {rc(dn,'real')}, unc {rc(dn,'uncertain')}) "
      f"| added {len(an)} (real {rc(an,'real')}, unc {rc(an,'uncertain')}, fake {rc(an,'fake')})")
print(f"   → 이만큼은 검증과 무관한 생성 비재현성. '버려진 real'을 침묵회피로 직결하면 과대평가.")
print(f"[피드백+노이즈] fake 있던 질문에서의 churn:")
print(f"   dropped {len(df)} (real {rc(df,'real')}, fake {rc(df,'fake')}) "
      f"| added {len(af)} (real {rc(af,'real')}, fake {rc(af,'fake')})")
print(f"\n결론 지표(강건): baseline fake 13 → 제거 {fate.get('사라짐(제거)',0)}, "
      f"잔존 {fate.get('잔존(fake)',0)}, 신규 fake {len(add_fake)} → verify-ON fake "
      f"{fate.get('잔존(fake)',0)+len(add_fake)}")

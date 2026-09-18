"""
run_koblex.py — KoBLEX(서술형)로 엄격 모드 실재화 + before/after 페어드 측정.

  질문(배경+질문, RAG 없음) → Qwen 답변 → 인용추출 →
    (a) 느슨 가짜율: 법제처 실존
    (b) 엄격 가짜율: 조문 내용을 법제처 본문 + KoBLEX gold와 대조

  검증 ON/OFF 비교는 **같은 답변 lineage의 before/after delta**로 측정한다.
  (두 번 따로 돌리는 temp=0 churn 회피 — 앞선 KCL 분석에서 확인된 교란요인)
    OFF = round-0 답변
    ON  = 그 답변을 가짜 인용 피드백으로 재생성한 최종본

  사용:
    export LAW_OC=...; vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000
    python run_koblex.py            # n=226 전체
    python run_koblex.py --n 30     # 일부
"""
import argparse
import csv
import json
import time
import collections

from openai import OpenAI

import config
from lawcheck import extract_citations, LawVerifier, _norm
from questions import load_koblex

SYSTEM_PROMPT = (
    "당신은 대한민국 법률 전문가입니다. 주어진 상황에 적용되는 법조문을 반드시 "
    "'법령명 제N조(조문제목)' 형식으로 구체적으로 인용하고, 그 조문이 무슨 내용인지 "
    "설명하세요. 확실하지 않은 조문은 지어내지 말고 모른다고 하세요."
)


def generate(client, prompt, extra=""):
    r = client.chat.completions.create(
        model=config.MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT + extra},
                  {"role": "user", "content": prompt}],
        temperature=0.0, max_tokens=config.MAX_TOKENS, seed=42)
    return r.choices[0].message.content or ""


def gold_index(gold):
    """{(법령명norm, jo, branch): gold본문+계층} — 엄격 대조용."""
    idx = {}
    for g in gold:
        idx[(_norm(g["law"]), g["jo"], g["branch"])] = \
            (g.get("content") or "") + " " + (g.get("hierarchy") or "")
    return idx


def verify_all(verifier, answer, gidx):
    cites = extract_citations(answer)
    checks = [verifier.verify(c, answer=answer, strict=True, gold_by_article=gidx)
              for c in cites]
    return cites, checks


def rates(cites, checks):
    n = len(cites)
    fake = sum(v.is_fake for v in checks)
    unc = sum(v.status == "uncertain" for v in checks)
    sfake = sum(v.strict_fake for v in checks)
    cunk = sum(v.status == "real" and v.content in (None, "unknown") for v in checks)
    mism = sum(v.status == "real" and v.content == "mismatch" for v in checks)
    ld = n - unc           # 느슨 판정가능
    sd = ld - cunk         # 엄격 판정가능
    return collections.Counter(n=n, fake=fake, unc=unc, sfake=sfake, cunk=cunk,
                               mism=mism, ld=ld, sd=sd)


def gold_recall(cites, checks, gold):
    """모델이 gold 조문을 몇 개나 실제로 인용했나 (실존 real만 인정)."""
    cited = {(_norm(v.official_name or c.law_name or ""), c.jo, c.jo_branch)
             for c, v in zip(cites, checks) if v.status == "real"}
    goldset = {(_norm(g["law"]), g["jo"], g["branch"]) for g in gold}
    hit = sum(1 for g in goldset if g in cited)
    return hit, len(goldset)


def fr(c):
    return (c["fake"] / c["ld"] if c["ld"] else 0.0,
            c["sfake"] / c["sd"] if c["sd"] else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="질문 수 (기본 전체 226)")
    ap.add_argument("--max-rounds", type=int, default=2, help="검증ON 재생성 최대 라운드")
    ap.add_argument("--rag", default=None,
                    help="사전검색(RAG): 검색결과 JSON 경로 ({id:[{hierarchy,content}]})")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rag_map = json.load(open(args.rag, encoding="utf-8")) if args.rag else None

    if not config.oc_is_set():
        raise SystemExit("✗ OC 키 미설정. export LAW_OC=...")

    client = OpenAI(base_url=config.VLLM_BASE_URL, api_key="EMPTY")
    verifier = LawVerifier()
    items = load_koblex(args.n)
    print(f"== KoBLEX paired (before/after) | model={config.MODEL} | n={len(items)} ==\n")

    rows = []
    AOFF, AON = collections.Counter(), collections.Counter()
    gold_hit = gold_tot = 0
    mismatch_cases = []
    t0 = time.time()

    for i, it in enumerate(items, 1):
        gidx = gold_index(it["gold"])

        # 사전검색(RAG): 검색된 관련 조문을 컨텍스트로 주입
        prompt = it["prompt"]
        if rag_map is not None:
            provs = rag_map.get(it["id"], [])
            if provs:
                ctx = "\n".join(
                    f"- {p.get('hierarchy','')}: {(p.get('content') or '').strip()[:400]}"
                    for p in provs)
                prompt = ("[참고 조문] 아래는 검색된 관련 법조문이다. 관련된 것만 골라 근거로 "
                          "인용하고, 무관하면 쓰지 마라. 여기 없는 조문을 지어내지 마라.\n"
                          + ctx + "\n\n" + it["prompt"])

        ans_off = generate(client, prompt)
        cites_off, checks_off = verify_all(verifier, ans_off, gidx)

        # ON: 같은 답변을 가짜 인용 피드백으로 재생성 (before/after)
        ans_on, cites_on, checks_on = ans_off, cites_off, checks_off
        for _ in range(args.max_rounds):
            fakes = [c for c, v in zip(cites_on, checks_on) if v.is_fake]
            if not fakes:
                break
            bad = ", ".join(f"{c.law_name or ''} {c.display}" for c in fakes)
            ans_on = generate(client, prompt,
                              extra=f"\n\n[검증 피드백] 다음 인용은 법제처 DB에 존재하지 않습니다: "
                                    f"{bad}. 해당 인용을 제거하거나 실존 조문으로 정정해 다시 답하세요.")
            cites_on, checks_on = verify_all(verifier, ans_on, gidx)

        roff, ron = rates(cites_off, checks_off), rates(cites_on, checks_on)
        AOFF += roff
        AON += ron
        h, tt = gold_recall(cites_off, checks_off, it["gold"])
        gold_hit += h
        gold_tot += tt

        for c, v in zip(cites_off, checks_off):
            if v.status == "real" and v.content == "mismatch":
                mismatch_cases.append({
                    "id": it["id"], "cite": f"{v.official_name} {c.display}",
                    "claimed": c.claimed_title, "note": v.content_note,
                    "ctx": ans_off[max(0, c.pos - 60): c.pos + 60].replace("\n", " ")})

        lo, so = fr(roff)
        ln, sn = fr(ron)
        print(f"[{i}/{len(items)}] {it['id']:<16} OFF 느슨{roff['fake']}/{roff['ld']}={lo:.0%} "
              f"엄격{roff['sfake']}/{roff['sd']}={so:.0%}(내용✗{roff['mism']}) | "
              f"ON 느슨{ron['fake']}/{ron['ld']}={ln:.0%} 엄격{ron['sfake']}/{ron['sd']}={sn:.0%}")

        rows.append({
            "id": it["id"], "n_hops": None,
            "off": dict(roff), "on": dict(ron),
            "answer_off": ans_off, "answer_on": ans_on,
            "checks_off": [{"cite": f"{(c.law_name or '')} {c.display}".strip(),
                            "official": v.official_name, "status": v.status,
                            "content": v.content, "note": v.note,
                            "content_note": v.content_note}
                           for c, v in zip(cites_off, checks_off)],
        })

    def block(A, label):
        lo, so = fr(A)
        print(f"  {label:<10} 인용 {A['n']:4d} | ⚠실존불명 {A['unc']:3d} | 내용판정불가 {A['cunk']:3d} "
              f"| 내용불일치 {A['mism']:3d}")
        print(f"             ★ 느슨 가짜율 {lo:.1%} ({A['fake']}/{A['ld']}) "
              f"| ★ 엄격 가짜율 {so:.1%} ({A['sfake']}/{A['sd']})")

    print("\n" + "=" * 74)
    print(f"KoBLEX n={len(items)} | {config.MODEL} | before/after 페어드")
    block(AOFF, "OFF(검증X)")
    block(AON, "ON(검증)")
    cov = AOFF['sd'] / AOFF['ld'] if AOFF['ld'] else 0
    print(f"  엄격 판정가능 커버리지(OFF) = {cov:.0%}  (KCL은 26%였음 → 서술형이라 ↑)")
    print(f"  gold 조문 재현(모델이 정답근거 조문 인용) = {gold_hit}/{gold_tot} = "
          f"{(gold_hit/gold_tot if gold_tot else 0):.1%}")
    print(f"  법제처 라이브호출 {verifier.live_calls} / 캐시히트 {verifier.cache_hits} "
          f"| {time.time()-t0:.0f}s")
    print("=" * 74)

    prefix = args.out or f"koblex_{time.strftime('%Y%m%d_%H%M%S')}"
    jpath = config.RESULTS_DIR / f"{prefix}.json"
    summary = {
        "model": config.MODEL, "dataset": "koblex", "n": len(items),
        "off": dict(AOFF), "on": dict(AON),
        "off_loose_rate": round(fr(AOFF)[0], 4), "off_strict_rate": round(fr(AOFF)[1], 4),
        "on_loose_rate": round(fr(AON)[0], 4), "on_strict_rate": round(fr(AON)[1], 4),
        "strict_coverage_off": round(cov, 4),
        "gold_recall": [gold_hit, gold_tot],
        "mismatch_cases": mismatch_cases, "rows": rows,
    }
    jpath.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    cpath = config.RESULTS_DIR / f"{prefix}.csv"
    with cpath.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "off_cites", "off_fake", "off_sfake", "off_mism",
                    "on_cites", "on_fake", "on_sfake"])
        for r in rows:
            w.writerow([r["id"], r["off"]["n"], r["off"]["fake"], r["off"]["sfake"],
                        r["off"]["mism"], r["on"]["n"], r["on"]["fake"], r["on"]["sfake"]])
    print(f"저장: {jpath}\n      {cpath}  | 내용불일치 사례 {len(mismatch_cases)}건")


if __name__ == "__main__":
    main()

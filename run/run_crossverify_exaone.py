"""
run_crossverify_exaone.py — [실배포 구성] 생성=EXAONE 기준, 자기검증 vs 이종 교차검증 [신규 측정].

  같은 KoBLEX 부분집합, EXAONE round-0 답변(temp=0·seed42 → paired)에서:
    (a) self  : run_agent(gen=EXAONE, cross=EXAONE)  — 동종 자기검증
    (b) cross : run_agent(gen=EXAONE, cross=Qwen)    — 이종 교차검증
  측정은 lawcheck(법제처)만 — 검증기는 cites/checks를 안 바꿈(측정 v1 불변).
  비교: 느슨/엄격 가짜율, 내용불일치, 수정 횟수, 검증기 revise 빈도(동종 vs 이종).

  실행: LAW_OC=.. python run_crossverify_exaone.py --n 30 [--no-rag]  (EXAONE:8001 + Qwen:8000 필요)
"""
import argparse
import collections
import json
import time

from openai import OpenAI

import config
from demo_cli import Retriever
from lawcheck import LawVerifier
import agent_pipeline as ap
from questions import load_koblex


def metrics(res):
    ck = res["checks"]
    return collections.Counter(
        n=len(ck),
        fake=sum(v.is_fake for v in ck),
        unc=sum(v.status == "uncertain" for v in ck),
        sfake=sum(v.strict_fake for v in ck),
        cunk=sum(v.status == "real" and v.content in (None, "unknown") for v in ck),
        mism=sum(v.status == "real" and v.content == "mismatch" for v in ck),
        fixes=len(res["log"]) - 1,
    )


def rates(c):
    ld = c["n"] - c["unc"]
    sd = ld - c["cunk"]
    return (c["fake"] / ld if ld else 0.0, c["sfake"] / sd if sd else 0.0, ld, sd)


def r0_verdict(res):
    v = res["log"][0].get("exaone")   # log 'exaone' 키 = 교차검증기 출력(모델 무관)
    return v["verdict"] if v else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--no-rag", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    use_rag = not args.no_rag
    if not config.oc_is_set():
        raise SystemExit("✗ LAW_OC 미설정")

    exaone = OpenAI(base_url=config.GEN_BASE_URL, api_key="EMPTY")    # 생성 + 동종검증
    qwen = OpenAI(base_url=config.CROSS_BASE_URL, api_key="EMPTY")    # 이종검증
    for c, name in [(exaone, "EXAONE(:8001)"), (qwen, "Qwen(:8000)")]:
        try:
            c.models.list()
        except Exception:
            raise SystemExit(f"✗ {name} 미실행")

    retriever = Retriever()
    verifier = LawVerifier()
    items = load_koblex(args.n)
    print(f"== [생성=EXAONE] 자기검증(EXAONE) vs 교차검증(Qwen) | KoBLEX n={len(items)} | "
          f"{'RAG' if use_rag else '맨몸'} ==\n")

    agg = {"self": collections.Counter(), "cross": collections.Counter()}
    revise_q = {"self": 0, "cross": 0}
    rows = []
    t0 = time.time()
    for i, it in enumerate(items, 1):
        q = it["prompt"]
        res_self = ap.run_agent(q, retriever, exaone, verifier, cross_client=exaone,
                                use_rag=use_rag, gen_model=config.GEN_MODEL,
                                cross_model=config.GEN_MODEL)
        res_cross = ap.run_agent(q, retriever, exaone, verifier, cross_client=qwen,
                                 use_rag=use_rag, gen_model=config.GEN_MODEL,
                                 cross_model=config.CROSS_MODEL)
        ms, mc = metrics(res_self), metrics(res_cross)
        agg["self"] += ms
        agg["cross"] += mc
        vs, vc = r0_verdict(res_self), r0_verdict(res_cross)
        revise_q["self"] += (vs == "revise")
        revise_q["cross"] += (vc == "revise")
        print(f"[{i}/{len(items)}] {it['id']:<16} self(EXAONE) fake{ms['fake']}/mism{ms['mism']}/fix{ms['fixes']}"
              f"  |  cross(Qwen) fake{mc['fake']}/mism{mc['mism']}/fix{mc['fixes']}"
              f"  (r0 self={vs} cross={vc})", flush=True)
        rows.append({"id": it["id"], "self": dict(ms), "cross": dict(mc),
                     "self_r0": vs, "cross_r0": vc})

    print("\n" + "=" * 78)
    print(f"KoBLEX n={len(items)} | 생성 EXAONE-3.5-7.8B | 자기검증 EXAONE / 교차검증 Qwen2.5-7B | 측정=법제처(v1)")
    print(f"{'조건':<24}{'인용':>5}{'느슨가짜율':>13}{'엄격가짜율':>13}{'내용불일치':>10}{'총수정':>8}")
    print("-" * 78)
    for key, label in [("self", "자기검증(EXAONE→EXAONE)"), ("cross", "교차검증(Qwen→EXAONE)")]:
        c = agg[key]
        lo, so, ld, sd = rates(c)
        loose = f"{lo:.1%}({c['fake']}/{ld})"
        strict = f"{so:.1%}({c['sfake']}/{sd})"
        print(f"{label:<24}{c['n']:>5}{loose:>14}{strict:>14}{c['mism']:>9}{c['fixes']:>8}")
    print("-" * 78)
    print(f"revise 판정 질문수(round-0): 자기검증(EXAONE)={revise_q['self']}/{len(items)} "
          f"| 교차검증(Qwen)={revise_q['cross']}/{len(items)}")
    print(f"({time.time()-t0:.0f}s)")

    prefix = args.out or f"crossverify_exaone_n{len(items)}_{time.strftime('%Y%m%d_%H%M%S')}"
    summary = {"model_gen": config.GEN_MODEL, "verify_self": config.GEN_MODEL,
               "verify_cross": config.CROSS_MODEL, "use_rag": use_rag, "n": len(items),
               "self": dict(agg["self"]), "cross": dict(agg["cross"]),
               "revise_q": revise_q, "rows": rows}
    out = config.RESULTS_DIR / f"{prefix}.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()

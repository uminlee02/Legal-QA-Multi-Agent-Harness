"""
run_crossverify.py — 자기검증(단일 모델) vs 이종 교차검증(+EXAONE) 비교 [신규 측정].

  같은 KoBLEX 부분집합, 같은 round-0 답변(temp=0·seed42 → paired)에서:
    (a) self  : run_agent(cross_client=None)   — 법제처 가짜만 트리거 (기존 사후검증)
    (b) cross : run_agent(cross_client=EXAONE) — 법제처 OR EXAONE revise 트리거 (신규)
  측정은 **lawcheck(법제처)만** — EXAONE은 cites/checks를 안 바꿈(측정 v1 불변 원칙 유지).
  비교: 느슨/엄격 가짜율, 내용불일치, 수정 횟수, EXAONE revise 빈도.

  실행: LAW_OC=.. python run_crossverify.py --n 30   (vLLM Qwen:8000 + EXAONE:8001 필요)
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=30, help="KoBLEX 부분집합 크기")
    p.add_argument("--no-rag", action="store_true", help="맨몸 생성(환각 존재 → 가설 검증)")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    use_rag = not args.no_rag
    if not config.oc_is_set():
        raise SystemExit("✗ LAW_OC 미설정")

    qwen = OpenAI(base_url=config.VLLM_BASE_URL, api_key="EMPTY")
    exaone = OpenAI(base_url=config.EXAONE_BASE_URL, api_key="EMPTY")
    try:
        qwen.models.list()
    except Exception:
        raise SystemExit("✗ Qwen vLLM(:8000) 미실행")
    try:
        exaone.models.list()
    except Exception:
        raise SystemExit("✗ EXAONE vLLM(:8001) 미실행")

    retriever = Retriever()
    verifier = LawVerifier()
    items = load_koblex(args.n)
    print(f"== 자기검증 vs 교차검증(+EXAONE) | KoBLEX n={len(items)} | "
          f"{'RAG' if use_rag else '맨몸(non-RAG)'} ==\n")

    agg = {"self": collections.Counter(), "cross": collections.Counter()}
    ex_revise_q = 0
    rows = []
    t0 = time.time()
    for i, it in enumerate(items, 1):
        q = it["prompt"]
        res_self = ap.run_agent(q, retriever, qwen, verifier, cross_client=None, use_rag=use_rag)
        res_cross = ap.run_agent(q, retriever, qwen, verifier, cross_client=exaone, use_rag=use_rag)
        ms, mc = metrics(res_self), metrics(res_cross)
        agg["self"] += ms
        agg["cross"] += mc
        ev0 = res_cross["log"][0].get("exaone")
        if ev0 and ev0["verdict"] == "revise":
            ex_revise_q += 1
        print(f"[{i}/{len(items)}] {it['id']:<16} self ✗{ms['fake']}/내용✗{ms['mism']}/수정{ms['fixes']}"
              f"  |  cross ✗{mc['fake']}/내용✗{mc['mism']}/수정{mc['fixes']}"
              f"  (EXAONE r0={ev0['verdict'] if ev0 else '-'})", flush=True)
        rows.append({"id": it["id"], "self": dict(ms), "cross": dict(mc),
                     "exaone_r0": ev0})

    print("\n" + "=" * 74)
    print(f"KoBLEX n={len(items)} | 생성 Qwen2.5-7B / 교차검증 EXAONE-3.5-7.8B | 측정=법제처(lawcheck v1)")
    print(f"{'조건':<22}{'인용':>5}{'느슨가짜율':>13}{'엄격가짜율':>13}{'내용불일치':>10}{'총수정':>8}")
    print("-" * 74)
    for key, label in [("self", "자기검증(단일·Qwen)"), ("cross", "교차검증(+EXAONE)")]:
        c = agg[key]
        lo, so, ld, sd = rates(c)
        loose = f"{lo:.1%}({c['fake']}/{ld})"
        strict = f"{so:.1%}({c['sfake']}/{sd})"
        print(f"{label:<22}{c['n']:>5}{loose:>14}{strict:>14}{c['mism']:>9}{c['fixes']:>8}")
    print("-" * 74)
    print(f"EXAONE이 round-0에서 'revise' 판정한 질문: {ex_revise_q}/{len(items)} "
          f"({ex_revise_q/len(items):.0%})")
    print(f"({time.time()-t0:.0f}s)")

    prefix = args.out or f"crossverify_n{len(items)}_{time.strftime('%Y%m%d_%H%M%S')}"
    summary = {"model_gen": config.MODEL, "model_cross": config.EXAONE_MODEL,
               "use_rag": use_rag, "n": len(items),
               "self": dict(agg["self"]), "cross": dict(agg["cross"]),
               "exaone_revise_q": ex_revise_q, "rows": rows}
    out = config.RESULTS_DIR / f"{prefix}.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()

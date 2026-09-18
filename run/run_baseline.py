"""
run_baseline.py — 4단계 측정 루프 (검증 OFF 베이스라인)  ⭐ 이 연구의 심장

  질문 → Qwen 답변(vLLM) → 인용추출(정규식) → 법제처 검증 → 가짜인용율 → CSV/JSON 저장

  사용:
    export LAW_OC="발급키"
    vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000     # (별도 터미널)
    python run_baseline.py --n 5                          # 시드 질문 5개
    python run_baseline.py --dataset kcl --n 20           # KCL

  검증 ON(=재생성 루프)은 --verify-loop 한 줄로 켠다 (regenerate_until_clean).
"""
import argparse
import csv
import json
import time

from openai import OpenAI

import config
from lawcheck import extract_citations, LawVerifier
from questions import load_questions

SYSTEM_PROMPT = (
    "당신은 대한민국 법률 전문가입니다. 사용자의 법률 질문에 정확히 답하고, "
    "관련 법조문을 반드시 '법령명 제N조 제M항' 형식으로 구체적으로 인용하세요. "
    "확실하지 않은 조문은 지어내지 말고 모른다고 하세요."
)


def generate(client, question: str, extra_system: str = "") -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT + extra_system},
                {"role": "user", "content": question}]
    r = client.chat.completions.create(
        model=config.MODEL, messages=messages,
        temperature=config.TEMPERATURE, max_tokens=config.MAX_TOKENS)
    return r.choices[0].message.content or ""


def verify_answer(verifier: LawVerifier, answer: str):
    cites = extract_citations(answer)
    checks = [verifier.verify(c) for c in cites]
    return cites, checks


def regenerate_until_clean(client, verifier, question, max_rounds=3):
    """검증 ON: 가짜로 뜬 인용을 모델에 되먹여 깨끗해질 때까지 재생성."""
    answer = generate(client, question)
    for _ in range(max_rounds):
        cites, checks = verify_answer(verifier, answer)
        fakes = [c for c, v in zip(cites, checks) if v.is_fake]
        if not fakes:
            return answer, cites, checks
        bad = ", ".join(f"{c.law_name or ''} {c.display}" for c in fakes)
        answer = generate(client, question,
                          extra_system=f"\n\n[검증 피드백] 다음 인용은 법제처 DB에 존재하지 "
                                       f"않습니다: {bad}. 해당 인용을 제거하거나 실존 조문으로 "
                                       f"정정해서 다시 답하세요.")
    cites, checks = verify_answer(verifier, answer)
    return answer, cites, checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="질문 개수")
    ap.add_argument("--dataset", default="seed", choices=["seed", "kcl"])
    ap.add_argument("--verify-loop", action="store_true",
                    help="검증 ON: 가짜 인용 재생성 루프")
    ap.add_argument("--out", default=None, help="결과 파일 prefix")
    args = ap.parse_args()

    if not config.oc_is_set():
        raise SystemExit("✗ OC 키 미설정. `export LAW_OC=발급키` 후 다시 실행하세요. "
                         "(발급: open.law.go.kr → OPEN API 신청)")

    client = OpenAI(base_url=config.VLLM_BASE_URL, api_key="EMPTY")
    verifier = LawVerifier()
    questions = load_questions(args.dataset, args.n)

    mode = "verify-ON" if args.verify_loop else "baseline(OFF)"
    print(f"== 측정 시작 | {mode} | model={config.MODEL} | n={len(questions)} ==\n")

    rows = []
    tot = {"cites": 0, "fake": 0, "unc": 0, "sfake": 0, "cunk": 0}
    for i, q in enumerate(questions, 1):
        if args.verify_loop:
            ans, _, _ = regenerate_until_clean(client, verifier, q)
        else:
            ans = generate(client, q)
        # 최종 답변을 엄격모드로 검증 (실존 + 내용정합성 동시 산출)
        cites = extract_citations(ans)
        checks = [verifier.verify(c, answer=ans, strict=True) for c in cites]

        n_fake = sum(v.is_fake for v in checks)
        n_unc = sum(v.status == "uncertain" for v in checks)
        n_sfake = sum(v.strict_fake for v in checks)
        n_cunk = sum(v.status == "real" and v.content in (None, "unknown") for v in checks)
        loose_dec = len(cites) - n_unc
        strict_dec = loose_dec - n_cunk
        loose_rate = (n_fake / loose_dec) if loose_dec else 0.0
        strict_rate = (n_sfake / strict_dec) if strict_dec else 0.0
        for k, val in (("cites", len(cites)), ("fake", n_fake), ("unc", n_unc),
                       ("sfake", n_sfake), ("cunk", n_cunk)):
            tot[k] += val

        print(f"[{i}/{len(questions)}] 인용 {len(cites)} | 느슨 ✗{n_fake}/{loose_dec}={loose_rate:.0%}"
              f" | 엄격 ✗{n_sfake}/{strict_dec}={strict_rate:.0%} | ⚠불명 {n_unc}")
        for c, v in zip(cites, checks):
            ctag = {"match": " 〔내용✓〕",
                    "mismatch": f" 〔내용✗ {v.content_note}〕",
                    "unknown": " 〔내용?〕"}.get(v.content, "")
            print(f"      {v.symbol} {(c.law_name or '(미지정)')} {c.display} — {v.note}{ctag}")
        print()

        rows.append({
            "idx": i, "question": q, "answer": ans,
            "n_citations": len(cites), "n_fake": n_fake, "n_uncertain": n_unc,
            "n_strict_fake": n_sfake, "n_content_unknown": n_cunk,
            "loose_fake_rate": round(loose_rate, 4), "strict_fake_rate": round(strict_rate, 4),
            "checks": [{"cite": f"{(c.law_name or '')} {c.display}".strip(),
                        "official": v.official_name, "status": v.status, "note": v.note,
                        "content": v.content, "content_note": v.content_note}
                       for c, v in zip(cites, checks)],
        })

    loose_dec_t = tot["cites"] - tot["unc"]
    strict_dec_t = loose_dec_t - tot["cunk"]
    loose_overall = (tot["fake"] / loose_dec_t) if loose_dec_t else 0.0
    strict_overall = (tot["sfake"] / strict_dec_t) if strict_dec_t else 0.0

    print("=" * 66)
    print(f"총 인용 {tot['cites']} | ⚠실존불명 {tot['unc']} | 내용판정불가 {tot['cunk']}")
    print(f"★ 느슨한 가짜율(실존만)     = {loose_overall:.1%}  ({tot['fake']}/{loose_dec_t})")
    print(f"★ 엄격한 가짜율(실존+내용)  = {strict_overall:.1%}  ({tot['sfake']}/{strict_dec_t})")
    print(f"   (법제처 라이브호출 {verifier.live_calls} / 캐시히트 {verifier.cache_hits})")
    print("=" * 66)

    prefix = args.out or f"{args.dataset}_{mode.split('(')[0]}_{time.strftime('%Y%m%d_%H%M%S')}"
    jpath = config.RESULTS_DIR / f"{prefix}.json"
    cpath = config.RESULTS_DIR / f"{prefix}.csv"
    summary = {"model": config.MODEL, "dataset": args.dataset, "mode": mode,
               "n_questions": len(questions), "total_citations": tot["cites"],
               "total_uncertain": tot["unc"], "total_content_unknown": tot["cunk"],
               "loose_fake": tot["fake"], "strict_fake": tot["sfake"],
               "loose_decidable": loose_dec_t, "strict_decidable": strict_dec_t,
               "loose_fake_rate": round(loose_overall, 4),
               "strict_fake_rate": round(strict_overall, 4), "rows": rows}
    jpath.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with cpath.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["idx", "n_citations", "n_fake", "n_strict_fake", "n_uncertain",
                    "loose_fake_rate", "strict_fake_rate", "question"])
        for r in rows:
            w.writerow([r["idx"], r["n_citations"], r["n_fake"], r["n_strict_fake"],
                        r["n_uncertain"], r["loose_fake_rate"], r["strict_fake_rate"],
                        r["question"][:60]])
    print(f"저장: {jpath}\n      {cpath}")


if __name__ == "__main__":
    main()

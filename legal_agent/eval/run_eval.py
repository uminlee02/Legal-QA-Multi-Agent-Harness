"""run_eval.py — 핵심 비교(브리프 §8): 베이스라인 vs 제안(전체 시스템).

동일한 큰 모델(Qwen3.5-27B)로 두 조건 비교.
  baseline : generate 만 (--baseline-rag off = 완전 순수, 검색도 제거 / on = 검색→생성)
  proposed : route→retrieve→generate→fact_verify→logic_verify→(verify-loop)→document

측정(frozen 정의 — run_koblex 의 rates/fr/gold_recall 재사용, KoBLEX gold 로 재채점):
  · 느슨 가짜율 = fake / (전체인용 − 실존판정불가)
  · 엄격 가짜율 = strict_fake / (판정가능 − 내용판정불가)
  · 감소율 = (baseline − proposed) / baseline × 100
함께: 도메인 분포, 재생성 횟수, 사실검증게이트 실행률, 청정통과율, gold recall.

체크포인트: 매 문항 후 --out JSON 저장 + 재실행 시 done id 건너뜀(장시간 런 안전).

  python eval/run_eval.py --n 226 --mode both --baseline-rag off --out results/lg_eval_n226.json
"""
import sys
import json
import pathlib
import argparse
import collections

_HERE = pathlib.Path(__file__).resolve().parent           # legal_agent/eval
for p in (str(_HERE.parent), str(_HERE.parent.parent)):    # legal_agent/ + repo root
    if p not in sys.path:
        sys.path.insert(0, p)

import config_lg as C
from graph import build_proposed_graph, build_baseline_graph, initial_state
from run_koblex import rates, fr, gold_recall, gold_index, verify_all  # frozen 채점기
from questions import load_koblex

# per_item row[m] 에 저장/집계하는 카운터 필드(집계=단순 합 → 재개 가능)
CNT = ("n", "fake", "unc", "sfake", "cunk", "mism", "ld", "sd", "gold_hit", "gold_tot")


def score(verifier, answer, gidx, gold):
    cites, checks = verify_all(verifier, answer, gidx)
    r = rates(cites, checks)
    h, tt = gold_recall(cites, checks, gold)
    row = {k: int(r[k]) for k in CNT if k not in ("gold_hit", "gold_tot")}
    row["gold_hit"], row["gold_tot"] = h, tt
    row["loose"] = r["fake"] / r["ld"] if r["ld"] else 0.0
    return row


def aggregate(per_item, modes):
    """per_item 리스트 → 조건별 합계 카운터 + 파생 지표."""
    out = {}
    for m in modes:
        acc = collections.Counter()
        dom = collections.Counter()
        retry_sum = gate_run = gate_clean = 0
        for row in per_item:
            if m not in row:
                continue
            for k in CNT:
                acc[k] += row[m].get(k, 0)
            dom[row[m].get("domain", "other")] += 1
            retry_sum += row[m].get("retry", 0)
            gate_run += 1 if row[m].get("fact_gate_run") else 0
            gate_clean += 1 if row[m].get("gate_clean") else 0
        lo = acc["fake"] / acc["ld"] if acc["ld"] else 0.0
        so = acc["sfake"] / acc["sd"] if acc["sd"] else 0.0
        out[m] = {"loose": lo, "strict": so, **{k: acc[k] for k in CNT},
                  "domain_dist": dict(dom), "retry_sum": retry_sum,
                  "fact_gate_run": gate_run, "gate_pass_clean": gate_clean}
    return out


def save(outp, meta, per_item, modes):
    agg = aggregate(per_item, modes)
    res = dict(meta, conditions=agg, n_done=len(per_item), per_item=per_item)
    if "baseline" in agg and "proposed" in agg:
        bl, pr = agg["baseline"], agg["proposed"]
        res["reduction"] = {
            "loose_pct": (bl["loose"] - pr["loose"]) / bl["loose"] * 100 if bl["loose"] else 0.0,
            "strict_pct": (bl["strict"] - pr["strict"]) / bl["strict"] * 100 if bl["strict"] else 0.0}
    outp.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--mode", choices=["baseline", "proposed", "both"], default="both")
    ap.add_argument("--baseline-rag", choices=["on", "off"], default="off",
                    help="off=완전 순수 baseline(검색 제거, §8) / on=검색→생성")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not C.oc_is_set():
        raise SystemExit("✗ LAW_OC 미설정 — 법제처 채점 필수. export LAW_OC=...")
    C.assert_heterogeneous()
    base_rag = args.baseline_rag == "on"

    from runtime import Runtime
    rt = Runtime()
    print("서버:", {k: ("✓" if v else "✗") for k, v in rt.check_servers().items()})
    verifier = rt.verifier
    # 법제처 검증 소스 명시(요구사항): mock 아님, 실제 Open API.
    print(f"· 법제처 사실검증: {'실제 Open API (LAW_OC 설정됨, mock 아님)' if C.oc_is_set() else 'MOCK'} "
          f"| 검증기=ResilientVerifier(404 재시도→uncertain 폴백) | frozen lawcheck v1", flush=True)
    print(f"· 논리검증(DeepSeek): {'④ 근거기반(evidence 주입, 근거대조)' if C.LOGIC_GROUNDED else '③ 근거없음(자기지식 논리검증)'}",
          flush=True)
    print(f"· 검증실패 처리: {'⑥ 타겟 편집(틀린 줄만 국소수정, 나머지 보존)' if C.EDIT_MODE=='targeted' else '③ 전체 재생성'}",
          flush=True)

    items = load_koblex(args.n)
    modes = (["baseline", "proposed"] if args.mode == "both" else [args.mode])
    apps = {}
    if "baseline" in modes:
        apps["baseline"] = build_baseline_graph(rt, use_rag=base_rag)
    if "proposed" in modes:
        apps["proposed"] = build_proposed_graph(rt)

    out = args.out or f"results/lg_eval_n{len(items)}_{args.mode}.json"
    outp = _HERE.parent.parent / out
    outp.parent.mkdir(exist_ok=True)
    meta = {"n": len(items), "gen": C.GEN_NAME, "logic": C.LOGIC_NAME,
            "mode": args.mode, "baseline_rag": base_rag}

    # ── 체크포인트 재개 ──
    per_item, done = [], set()
    if outp.exists():
        try:
            prev = json.loads(outp.read_text(encoding="utf-8"))
            per_item = prev.get("per_item", [])
            done = {r["id"] for r in per_item}
            print(f"· 재개: 기존 {len(done)}문항 건너뜀")
        except Exception:
            per_item, done = [], set()

    print(f"== LangGraph 평가 | gen={C.GEN_NAME} logic={C.LOGIC_NAME} | n={len(items)} "
          f"| mode={args.mode} | baseline_rag={base_rag} ==")

    for i, it in enumerate(items, 1):
        if it["id"] in done:
            continue
        gidx = gold_index(it["gold"])
        row = {"id": it["id"]}
        for m, app in apps.items():
            use_rag = True if m == "proposed" else base_rag
            try:      # 안전망: 한 조건/문항 예외가 전체 런을 죽이지 않게(문항은 done 처리→무한재시도 방지)
                final = app.invoke(initial_state(it["prompt"], m, use_rag=use_rag),
                                   config={"recursion_limit": 60})
                r = score(verifier, final.get("answer", ""), gidx, it["gold"])
                fk = final.get("fact_check", {}) or {}
                r.update({"domain": final.get("domain", "other"),
                          "retry": final.get("retry_count", 0),
                          "fact_gate_run": bool(final.get("fact_verified")),
                          "gate_clean": bool(fk) and all(v not in ("fake", "mismatch") for v in fk.values())})
                elog = final.get("edit_log", [])          # ⑥ 타겟편집 통계
                if elog:
                    r["n_edits"] = len(elog)
                    r["edit_fraction_mean"] = round(sum(e["edit_fraction"] for e in elog) / len(elog), 3)
                    r["lines_edited"] = sum(e["lines_edited"] for e in elog)
                    r["lines_deleted"] = sum(e["lines_deleted"] for e in elog)
                row[m] = r
            except Exception as e:
                row.setdefault("errors", {})[m] = str(e)[:200]
                print(f"  ⚠ {it['id']} [{m}] 예외 skip: {str(e)[:120]}", flush=True)
        per_item.append(row)
        agg = save(outp, meta, per_item, modes)
        msg = " | ".join(f"{m} 느슨{row[m]['fake']}/{row[m]['ld']}(누적{agg[m]['loose']:.0%})" for m in apps)
        print(f"[{len(per_item)}/{len(items)}] {it['id']:<16} {msg}", flush=True)

    # ── 최종 요약 ──
    agg = aggregate(per_item, modes)
    print("\n" + "=" * 60)
    for m in modes:
        a = agg[m]
        print(f"\n[{m}]")
        print(f"  ★ 느슨 가짜율 {a['loose']:.1%} ({a['fake']}/{a['ld']})  "
              f"| ★ 엄격 가짜율 {a['strict']:.1%} ({a['sfake']}/{a['sd']})")
        print(f"  내용불일치 {a['mism']} | 실존판정불가 {a['unc']} | 내용판정불가 {a['cunk']} "
              f"| gold recall {a['gold_hit']}/{a['gold_tot']}"
              + (f"={a['gold_hit']/a['gold_tot']:.1%}" if a['gold_tot'] else ""))
        print(f"  도메인 {a['domain_dist']} | 재생성합 {a['retry_sum']} "
              f"| 사실검증게이트 {a['fact_gate_run']}/{len(per_item)} | 청정통과 {a['gate_pass_clean']}/{len(per_item)}")
    if "baseline" in agg and "proposed" in agg:
        bl, pr = agg["baseline"], agg["proposed"]
        rl = (bl["loose"] - pr["loose"]) / bl["loose"] * 100 if bl["loose"] else 0.0
        rs = (bl["strict"] - pr["strict"]) / bl["strict"] * 100 if bl["strict"] else 0.0
        print(f"\n[감소율 = (베이스라인 − 제안)/베이스라인]")
        print(f"  느슨 {bl['loose']:.1%} → {pr['loose']:.1%} = {rl:+.1f}%")
        print(f"  엄격 {bl['strict']:.1%} → {pr['strict']:.1%} = {rs:+.1f}%")
    print(f"\n저장: {outp}")


if __name__ == "__main__":
    main()

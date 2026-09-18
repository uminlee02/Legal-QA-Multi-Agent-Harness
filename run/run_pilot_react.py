"""
run_pilot_react.py — 파일럿: 고정 파이프라인(Arm A) vs 자율 ReAct 하네스(Arm B) [신규 측정].

  파일럿(n=30)이 답할 것 딱 두 개:
    ① go/no-go : Qwen2.5-7B가 ReAct 자율 루프를 포맷붕괴·무한루프 없이 실제로 도는가?
    ② 방향     : 자율 arm이 고정 대비 환각 (a)↓ (b)↑ (c)조건부 중 어디로 기우나? (잠정)

  두 팔 — 오직 '제어 흐름'만 다름 (생성=Qwen2.5-7B / 도구=KoE5·법제처·EXAONE / 채점=lawcheck v1 동일):
    Arm A(고정) = agent_pipeline.run_agent(cross_client=EXAONE) 그대로
                  검색→생성→검증(법제처+EXAONE)→문제시 재생성(≤2)→종료. (표3 교차검증 경로 재사용)
    Arm B(자율) = 같은 초기 조건에서 Qwen이 매 스텝 [행동]을 스스로 선택하는 수동 ReAct 루프:
                  검색(KoE5) | 사실검증(법제처) | 논리검증(EXAONE) | 종료.
                  생성·수정은 도구가 아님 — 매 턴 [답변] 블록으로 루프 안에서 (재)작성.
                  max_steps=8 도달 시 강제 종료(로그 표기).

  오염 방지 (landmine):
    - 생성 모델은 GEN_MODEL 상수로 명시 고정(★ config.GEN_MODEL 기본값이 EXAONE라 절대 의존 금지).
    - temp=0, seed=42. 채점 gold_by_article = 초기 top-3 검색 결과(양팔·양조건 동일 = 기존 경로와 동일).
    - 기존 결과 파일(koblex_*, crossverify_*) 미접촉 — 출력은 results/pilot_react_* 신규.

  실행: LAW_OC=.. python run_pilot_react.py --cond rag --n 30
        LAW_OC=.. python run_pilot_react.py --cond norag --n 30
        (Qwen:8000 + EXAONE:8001 vLLM 필요 — run_app.sh와 동일 세팅)
"""
import argparse
import collections
import csv
import json
import re
import time

from openai import OpenAI

import config
from demo_cli import Retriever, article_label
from lawcheck import extract_citations, LawVerifier
import agent_pipeline as ap
from questions import load_koblex
from run_koblex import gold_recall

GEN_MODEL = "Qwen/Qwen2.5-7B-Instruct"                    # ★ 명시 고정 — config 기본값(EXAONE) 오염 방지
CROSS_MODEL = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"      # 논리검증 도구 (측정값 불변, 피드백만)
MAX_STEPS = 8
LOOP_RUN = 3          # 동일 액션 연속 ≥3 → 루프 플래그

# ── Arm B ReAct 프로토콜 (답변 형식 규칙은 ap.SYSTEM 그대로 공유 → 내용 스타일 동일) ──
PROTOCOL = (
    "\n\n[자율 에이전트 프로토콜]\n"
    "당신은 스스로 도구를 선택해 위 형식의 답변서를 검증·개선하는 자율 에이전트입니다. "
    "매 턴 반드시 아래 세 블록을 이 순서대로 출력하세요.\n"
    "[생각] 현재 답변 상태 판단과 다음 행동이 필요한 이유를 1~3문장.\n"
    "[답변] 현재 시점의 완성된 답변서 전체(【답변 요지】【조문 해설】【결론】). 매 턴 전체를 다시 쓰세요.\n"
    "[행동] 아래 넷 중 정확히 하나만 한 줄로 쓰고 다른 말을 덧붙이지 마세요:\n"
    "검색: <찾을 내용>\n"
    "사실검증\n"
    "논리검증\n"
    "종료\n"
    "- 검색: 법조문 데이터베이스에서 관련 조문을 더 찾아 줍니다(근거가 부족할 때).\n"
    "- 사실검증: [답변]에 인용된 조문의 실존·내용을 법제처에서 확인해 알려줍니다.\n"
    "- 논리검증: 다른 법률 전문가 모델이 답변의 논리·조문적용을 검토해 알려줍니다.\n"
    "- 종료: 답변을 확정합니다. 충분히 검증되었다고 판단될 때만 쓰세요.")


def count_checks(checks):
    return collections.Counter(
        n=len(checks),
        fake=sum(v.is_fake for v in checks),
        unc=sum(v.status == "uncertain" for v in checks),
        sfake=sum(v.strict_fake for v in checks),
        cunk=sum(v.status == "real" and v.content in (None, "unknown") for v in checks),
        mism=sum(v.status == "real" and v.content == "mismatch" for v in checks))


def rate2(c):
    ld = c["n"] - c["unc"]
    sd = ld - c["cunk"]
    return (c["fake"] / ld if ld else 0.0, c["sfake"] / sd if sd else 0.0, ld, sd)


def checks_dump(cites, checks):
    return [{"cite": f"{(c.law_name or '')} {c.display}".strip(),
             "official": v.official_name, "status": v.status,
             "content": v.content, "note": v.note} for c, v in zip(cites, checks)]


# ════════════════════════════════════════════════════════════════════════════
# Arm A — 고정 파이프라인 (기존 run_agent 경로 재사용, 신규 로그만 계측)
# ════════════════════════════════════════════════════════════════════════════
def run_arm_a(question, retriever, qwen, exaone, verifier, use_rag):
    res = ap.run_agent(question, retriever, qwen, verifier, cross_client=exaone,
                       use_rag=use_rag, gen_model=GEN_MODEL, cross_model=CROSS_MODEL)
    log = res["log"]
    rounds = len(log)                     # 사실검증 = 논리검증 호출 횟수
    fixes = rounds - 1                    # 재생성 횟수
    last = log[-1]
    ex_rev = bool(last["exaone"] and last["exaone"]["verdict"] == "revise")
    passed = not last["bad"] and not ex_rev
    seq = (["검색"] if use_rag else []) + ["생성"]
    for i in range(rounds):
        seq += ["사실검증", "논리검증"]
        if i < rounds - 1:
            seq.append("재생성")
    seq.append("종료" if passed else "종료(max_iter)")
    tools = {"search": 1 if use_rag else 0, "fact": rounds, "logic": rounds}
    tools["total"] = sum(tools.values())
    return {"arm": "A", "answer": res["answer"], "cites": res["cites"], "checks": res["checks"],
            "seq": seq, "tools": tools, "steps": rounds, "fixes": fixes,
            "end": "finish" if passed else "max_iter", "loop": False, "fmt_errors": 0,
            "iters_log": log}


# ════════════════════════════════════════════════════════════════════════════
# Arm B — 자율 ReAct 루프 (수동, 기존 도구 호출 코드 재사용)
# ════════════════════════════════════════════════════════════════════════════
def chat(qwen, messages):
    r = qwen.chat.completions.create(model=GEN_MODEL, messages=messages,
                                     temperature=0.0, max_tokens=2048, seed=42)
    return r.choices[0].message.content or ""


def parse_step(txt):
    """→ (answer|None, action|None, query|None). action ∈ 검색/사실검증/논리검증/종료.

    [행동]은 '마지막' 블록 기준 — 모델이 [생각] 안에서 이전 피드백의 "[행동] 블록이
    누락..." 같은 문구를 인용하면 첫 매치가 그걸 잡아 유효한 최종 [행동]을 놓친다
    (v1 파서 버그, qa_140-맨몸에서 종료 7회를 형식오류로 오판). 중국어 행동블록
    ([行动] 检索 등)은 의도적으로 계속 형식오류 처리 — 코드스위칭 실패 신호 그 자체.
    """
    acts = list(re.finditer(r"\[\s*행동\s*\]", txt))
    act_m = acts[-1] if acts else None
    am = re.search(r"\[\s*답변\s*\]\s*[::]?", txt)
    answer = None
    if am:
        end = act_m.start() if (act_m and act_m.start() > am.end()) else len(txt)
        answer = txt[am.end():end].strip() or None
    action = query = None
    line = ""
    if act_m:
        tail = txt[act_m.end():].lstrip(" ::\n")
        line = tail.splitlines()[0].strip() if tail.strip() else ""
    else:                                          # 관대 폴백: 마지막 비어있지 않은 줄
        lines = [l.strip() for l in txt.splitlines() if l.strip()]
        if lines:
            line = lines[-1]
    s = line.replace(" ", "")
    if s.startswith("검색") or s.startswith("추가검색"):
        action = "검색"
        parts = re.split(r"[::]", line, maxsplit=1)
        query = parts[1].strip() if len(parts) == 2 else ""
    elif s.startswith("사실검증"):
        action = "사실검증"
    elif s.startswith("논리검증"):
        action = "논리검증"
    elif s.startswith("종료") or s.startswith("완료") or s.lower().startswith("finish"):
        action = "종료"
    return answer, action, query


def obs_fact(cites, checks):
    if not cites:
        return "인용된 법조문이 없습니다. '법령명 제N조' 형식으로 인용하세요."
    out = []
    for c, v in zip(cites, checks):
        ex = {"real": "✓실존", "fake": "✗미존재"}.get(v.status, "⚠확인불가")
        ct = ""
        if v.status == "real":
            ct = {"match": ", 내용일치", "mismatch": ", ✗내용불일치",
                  "unknown": ", 내용미확인"}.get(v.content, "")
        name = v.official_name or c.law_name or "(법령명 미상)"
        out.append(f"- {name} {c.display}: {ex}{ct} ({v.note})")
    return "\n".join(out)


def run_arm_b(question, retriever, qwen, exaone, verifier, gidx, init_provs, use_rag):
    base = (ap.rag_prompt(question, init_provs) if use_rag
            else f"[질문] {question}\n\n관련 대한민국 법조문을 '법령명 제N조' 형식으로 인용해 답하세요.")
    messages = [{"role": "system", "content": ap.SYSTEM + PROTOCOL},
                {"role": "user", "content": base + "\n\n지금 [생각]/[답변]/[행동]을 출력하세요."}]
    evidence = list(init_provs) if use_rag else []          # 에이전트가 확보한 근거(논리검증 ctx)
    seen_h = {p["hierarchy"] for p in evidence}
    draft = ""
    seq, steps_log = [], []
    tools = collections.Counter(search=0, fact=0, logic=0)
    fmt_errors = 0
    end = "max_steps"
    steps = 0
    for step in range(1, MAX_STEPS + 1):
        steps = step
        out = chat(qwen, messages)
        messages.append({"role": "assistant", "content": out})
        ans, act, qry = parse_step(out)
        changed = bool(ans) and ans != draft
        if ans:
            draft = ans
        if act is None:
            fmt_errors += 1
            seq.append("형식오류")
            steps_log.append({"step": step, "action": "형식오류", "answer_changed": changed})
            obs = ("[형식 오류] [행동] 블록을 찾지 못했습니다. 반드시 마지막에 [행동] 줄로 "
                   "'검색: <내용>' / '사실검증' / '논리검증' / '종료' 중 하나만 출력하세요.")
            messages.append({"role": "user", "content": obs})
            continue
        if act == "종료":
            seq.append("종료")
            steps_log.append({"step": step, "action": "종료", "answer_changed": changed})
            end = "finish"
            break
        if act == "검색":
            tools["search"] += 1
            qq = (qry or question).strip()
            got = retriever.search(qq)
            new = [p for p in got if p["hierarchy"] not in seen_h]
            for p in new:
                seen_h.add(p["hierarchy"])
                evidence.append(p)
            body = "\n".join(f"- {article_label(p['hierarchy'])}: {(p['content'] or '').strip()[:380]}"
                             for p in got) or "(검색 결과 없음)"
            dup = ("" if len(new) == len(got)
                   else f"\n(이 중 {len(got)-len(new)}건은 이미 확보한 조문과 중복)")
            obs = f"[관측·검색결과] 질의: {qq}\n{body}{dup}"
            label = f"검색:{qq[:24]}"
        elif act == "사실검증":
            tools["fact"] += 1
            cs = extract_citations(draft)
            ck = [verifier.verify(c, answer=draft, strict=True, gold_by_article=gidx) for c in cs]
            obs = "[관측·법제처 사실검증]\n" + obs_fact(cs, ck)
            label = "사실검증"
        else:                                              # 논리검증
            tools["logic"] += 1
            rv = ap.exaone_review(exaone, question, draft, evidence, model=CROSS_MODEL)
            obs = ("[관측·논리검증(EXAONE)] 판정: " + rv["verdict"]
                   + ("; 지적: " + "; ".join(rv["issues"]) if rv["issues"] else " (문제 없음)"))
            label = "논리검증"
        seq.append(label)
        steps_log.append({"step": step, "action": label, "answer_changed": changed})
        messages.append({"role": "user", "content":
                         obs + "\n\n위 관측을 반영해 다시 [생각]/[답변]/[행동]을 출력하세요. "
                               "답변이 충분히 검증·확정되었으면 [행동]에 종료."})

    # 루프 플래그: 동일 액션(검색은 유형 기준) 연속 LOOP_RUN회 이상
    types = [("검색" if s.startswith("검색") else s) for s in seq]
    max_run = run = 0
    for i, t in enumerate(types):
        run = run + 1 if (i and t == types[i - 1]) else 1
        max_run = max(max_run, run)
    loop = max_run >= LOOP_RUN

    cites = extract_citations(draft)
    checks = [verifier.verify(c, answer=draft, strict=True, gold_by_article=gidx) for c in cites]
    tools["total"] = tools["search"] + tools["fact"] + tools["logic"]
    return {"arm": "B", "answer": draft, "cites": cites, "checks": checks,
            "seq": seq, "tools": dict(tools), "steps": steps, "fixes": None,
            "end": end, "loop": loop, "max_run": max_run, "fmt_errors": fmt_errors,
            "answer_empty": not draft, "transcript": messages}


# ════════════════════════════════════════════════════════════════════════════
def summarize(rows, arm):
    rs = [r for r in rows if r["arm"] == arm]
    agg = collections.Counter()
    for r in rs:
        agg += collections.Counter(r["metrics"])
    lo, so, ld, sd = rate2(agg)
    n = len(rs)
    return {
        "n_q": n, "cites": agg["n"], "agg": dict(agg),
        "loose_rate": lo, "loose_frac": f"{agg['fake']}/{ld}",
        "strict_rate": so, "strict_frac": f"{agg['sfake']}/{sd}",
        "mism": agg["mism"],
        "gold_recall": [sum(r["gold_hit"] for r in rs), sum(r["gold_tot"] for r in rs)],
        "avg_tools": round(sum(r["tools"]["total"] for r in rs) / n, 2),
        "avg_search": round(sum(r["tools"]["search"] for r in rs) / n, 2),
        "avg_fact": round(sum(r["tools"]["fact"] for r in rs) / n, 2),
        "avg_logic": round(sum(r["tools"]["logic"] for r in rs) / n, 2),
        "avg_steps": round(sum(r["steps"] for r in rs) / n, 2),
        "finish_rate": sum(r["end"] == "finish" for r in rs) / n,
        "loop_rate": sum(bool(r["loop"]) for r in rs) / n,
        "fmt_error_q": sum(r["fmt_errors"] > 0 for r in rs),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cond", choices=["rag", "norag"], required=True)
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    use_rag = args.cond == "rag"
    if not config.oc_is_set():
        raise SystemExit("✗ LAW_OC 미설정")

    qwen = OpenAI(base_url=config.VLLM_BASE_URL, api_key="EMPTY")
    exaone = OpenAI(base_url=config.EXAONE_BASE_URL, api_key="EMPTY")
    served = [m.id for m in qwen.models.list().data]
    assert GEN_MODEL in served, f"✗ :8000이 {served} 서빙 — {GEN_MODEL} 아님(오염 위험, 중단)"
    served_x = [m.id for m in exaone.models.list().data]
    assert CROSS_MODEL in served_x, f"✗ :8001이 {served_x} 서빙 — {CROSS_MODEL} 아님"

    retriever = Retriever()
    verifier = LawVerifier()
    items = load_koblex(args.n)
    print(f"\n== 파일럿: 고정(A) vs 자율 ReAct(B) | KoBLEX n={len(items)} | "
          f"{'RAG' if use_rag else '맨몸'} | 생성={GEN_MODEL} ==\n", flush=True)

    rows = []
    t0 = time.time()
    for i, it in enumerate(items, 1):
        q = it["prompt"]
        init_provs = retriever.search(q)               # 초기 top-3 (양팔 공통, 채점 gold도 여기서)
        gidx = ap.gold_index(init_provs)

        ra = run_arm_a(q, retriever, qwen, exaone, verifier, use_rag)
        rb = run_arm_b(q, retriever, qwen, exaone, verifier, gidx, init_provs, use_rag)

        for r in (ra, rb):
            m = count_checks(r["checks"])
            gh, gt = gold_recall(r["cites"], r["checks"], it["gold"])
            rows.append({"id": it["id"], "arm": r["arm"], "cond": args.cond,
                         "metrics": dict(m), "gold_hit": gh, "gold_tot": gt,
                         "tools": r["tools"], "steps": r["steps"], "end": r["end"],
                         "loop": r["loop"], "max_run": r.get("max_run", 0),
                         "fmt_errors": r["fmt_errors"],
                         "seq": r["seq"], "answer": r["answer"],
                         "checks": checks_dump(r["cites"], r["checks"]),
                         "transcript": r.get("transcript"),
                         "iters_log": r.get("iters_log")})
        ma, mb = count_checks(ra["checks"]), count_checks(rb["checks"])
        print(f"[{i}/{len(items)}] {it['id']:<16} "
              f"A ✗{ma['fake']}/{ma['n']} 툴{ra['tools']['total']} {ra['end']:<8} | "
              f"B ✗{mb['fake']}/{mb['n']} 툴{rb['tools']['total']} 스텝{rb['steps']} "
              f"{rb['end']:<9} {'루프!' if rb['loop'] else ''} "
              f"{'포맷✗' + str(rb['fmt_errors']) if rb['fmt_errors'] else ''}\n"
              f"    B 시퀀스: {' → '.join(rb['seq'])}", flush=True)

    sum_a, sum_b = summarize(rows, "A"), summarize(rows, "B")
    print("\n" + "=" * 90)
    print(f"파일럿 요약 | KoBLEX n={len(items)} | 조건={'RAG' if use_rag else '맨몸'} | "
          f"생성={GEN_MODEL} | 채점=lawcheck v1")
    hdr = f"{'arm':<14}{'인용':>5}{'느슨가짜율':>14}{'엄격가짜율':>14}{'gold재현':>10}{'평균툴콜':>9}{'정상종료':>9}{'루프율':>8}"
    print(hdr)
    print("-" * 90)
    for lab, s in [("A 고정", sum_a), ("B 자율ReAct", sum_b)]:
        gr = f"{s['gold_recall'][0]}/{s['gold_recall'][1]}"
        print(f"{lab:<14}{s['cites']:>5}"
              f"{s['loose_rate']:>9.1%}({s['loose_frac']})"
              f"{s['strict_rate']:>9.1%}({s['strict_frac']})"
              f"{gr:>10}{s['avg_tools']:>9}{s['finish_rate']:>9.0%}{s['loop_rate']:>8.0%}")
    print("-" * 90)
    print(f"B 세부: 평균스텝 {sum_b['avg_steps']} | 검색 {sum_b['avg_search']} / 사실 {sum_b['avg_fact']}"
          f" / 논리 {sum_b['avg_logic']} | 포맷오류 질문 {sum_b['fmt_error_q']}/{sum_b['n_q']}")
    print(f"({time.time()-t0:.0f}s | 법제처 라이브 {verifier.live_calls} / 캐시 {verifier.cache_hits})")

    prefix = args.out or f"pilot_react_{args.cond}_n{len(items)}"
    summary = {"gen_model": GEN_MODEL, "cross_model": CROSS_MODEL, "cond": args.cond,
               "use_rag": use_rag, "n": len(items), "max_steps": MAX_STEPS,
               "grader": "lawcheck v1 (gold=초기 top-3, 양팔 동일)",
               "arm_a": sum_a, "arm_b": sum_b, "rows": rows}
    jp = config.RESULTS_DIR / f"{prefix}.json"
    jp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    cp = config.RESULTS_DIR / f"{prefix}.csv"
    with cp.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "arm", "cites", "fake", "sfake", "mism", "gold_hit", "gold_tot",
                    "search", "fact", "logic", "tools_total", "steps", "end", "loop",
                    "fmt_errors", "seq"])
        for r in rows:
            w.writerow([r["id"], r["arm"], r["metrics"]["n"], r["metrics"]["fake"],
                        r["metrics"]["sfake"], r["metrics"]["mism"], r["gold_hit"], r["gold_tot"],
                        r["tools"]["search"], r["tools"]["fact"], r["tools"]["logic"],
                        r["tools"]["total"], r["steps"], r["end"], int(bool(r["loop"])),
                        r["fmt_errors"], "→".join(r["seq"])])
    print(f"저장: {jp}\n      {cp}")


if __name__ == "__main__":
    main()

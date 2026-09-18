"""
run_main3.py — 3번 실험: 도메인 검증 스킬이 내용 오적용을 저감하는가 [신규, 동결 불변].

  arm 3개 (baseline = 2번의 자율+게이트 = run_main.run_arm_bc("C")):
    C0 = 자율+게이트, 스킬 없음.               ★2번 코드 경로 그대로 재사용(재현 sanity용)
    C1 = C0 + 스킬(오라클 라우팅: domain_labels.csv 정답 도메인으로 강제 로드)
    C2 = C0 + 스킬(자율 라우팅: 시작 시 모델이 형사/민사/기타 분류 → 로드/미로드)
  C0/C1/C2 차이 = 스킬 유무·라우팅 방식뿐. 모델·도구·프로토콜·게이트·평가셋·채점 100% 동일.

  landmine(2번 계승): GEN 명시+서빙 assert, 이종 assert, 액션 제약 디코딩,
    lawcheck v1 동결(gold=초기 top-3), temp=0 seed42, 스킬 확정본 그대로(해시 기록·편집 금지).

  실행: LAW_OC=.. python run_main3.py --subset target --gen qwen --cond rag --run 1
    subset: target=형사+민사 76문항(페이즈A) / control=기타+mixed 150문항(페이즈B).
"""
import argparse
import collections
import csv
import hashlib
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

import config
from demo_cli import Retriever, article_label
from lawcheck import extract_citations, LawVerifier
import agent_pipeline as ap
from questions import load_koblex
from run_koblex import gold_recall
import run_main as m2   # 2번 코드 경로 재사용 (C0 동일성 보장)

MAX_STEPS = m2.MAX_STEPS
ACTIONS = m2.ACTIONS
ROUTE_OPTS = ["형사", "민사", "기타"]

SKILL_FILES = {"형사": "criminal-law-verification.md", "민사": "civil-law-verification.md"}

SKILL_WRAP = ("\n\n[로드된 스킬: {name}]\n"
              "아래 스킬의 검증 절차(0~4단계)를 순서대로 수행하세요. "
              "매 턴 [생각] 첫머리에 지금 수행 중인 단계를 '단계N' 형식으로 표시하세요.\n\n{body}")


def load_skills():
    out = {}
    for dom, fn in SKILL_FILES.items():
        raw = (config.ROOT / "skills" / fn).read_text(encoding="utf-8")
        m = re.match(r"^---\n.*?\n---\n", raw, re.S)
        body = raw[m.end():].strip() if m else raw.strip()
        name = fn.replace(".md", "")
        out[dom] = {"name": name, "body": body,
                    "sha": hashlib.sha256(raw.encode()).hexdigest()}
    return out


def load_labels():
    p = config.RESULTS_DIR / "domain_labels.csv"
    rows = list(csv.DictReader(open(p, encoding="utf-8-sig")))
    return {r["question_id"]: r["domain"] for r in rows}


def stage_marks(transcript):
    """스킬 실행 충실도: assistant 장문 출력에서 '단계N' 마커 수집."""
    marks = set()
    for msg in transcript:
        if msg["role"] == "assistant" and len(msg["content"]) > 60:
            marks |= {int(x) for x in re.findall(r"단계\s*([0-4])", msg["content"])}
    return sorted(marks)


# ── C1/C2: run_main.run_arm_bc("C")와 동일 루프 + 시스템 프롬프트에 스킬 블록만 추가 ──
def run_arm_skill(arm_label, question, lret, gen_client, gen_model, cross_client, cross_model,
                  verifier, gidx, init_provs, use_rag, choice_mode, skill=None):
    base = (ap.rag_prompt(question, init_provs) if use_rag
            else f"[질문] {question}\n\n관련 대한민국 법조문을 '법령명 제N조' 형식으로 인용해 답하세요.")
    sys_txt = ap.SYSTEM + m2.PROTOCOL2
    if skill:
        sys_txt += SKILL_WRAP.format(name=skill["name"], body=skill["body"])
    messages = [{"role": "system", "content": sys_txt},
                {"role": "user", "content": base + "\n\n먼저 [생각](1~3문장)과 [답변](답변서 전체)을 출력하세요."}]
    evidence = list(init_provs) if use_rag else []
    seen_h = {p["hierarchy"] for p in evidence}
    draft = ""
    seq, tools = [], collections.Counter(search=0, fact=0, logic=0)
    fact_ok = 0
    fmt_errors = answer_miss = 0
    end = "max_steps"
    steps = 0
    for step in range(1, MAX_STEPS + 1):
        steps = step
        out = m2.gen_call(gen_client, gen_model, messages)
        messages.append({"role": "assistant", "content": out})
        ans, miss = m2.parse_answer(out)
        if miss:
            answer_miss += 1
        if ans:
            draft = ans

        gated = fact_ok < 1                                  # 게이트 = 2번 C와 동일
        allowed = [a for a in ACTIONS if a != "종료"] if gated else ACTIONS
        menu = "다음 행동을 하나만 선택하세요: " + " / ".join(allowed)
        if gated:
            menu += "\n(참고: '종료'는 사실검증을 1회 이상 수행한 뒤에만 선택할 수 있습니다.)"
        messages.append({"role": "user", "content": menu})
        act = m2.choice_call(gen_client, gen_model, messages, allowed, choice_mode)
        messages.append({"role": "assistant", "content": act})
        if act not in allowed:
            fmt_errors += 1
            seq.append("형식오류")
            messages.append({"role": "user", "content": "잘못된 선택입니다. 제시된 행동 중 하나만 고르세요."})
            continue

        if act == "종료":
            seq.append("종료")
            end = "finish"
            break
        if act == "검색":
            tools["search"] += 1
            messages.append({"role": "user", "content": "검색어를 한 줄로만 출력하세요(설명 없이)."})
            q_out = m2.gen_call(gen_client, gen_model, messages, max_tokens=40)
            messages.append({"role": "assistant", "content": q_out})
            lines = [l.strip() for l in q_out.strip().splitlines() if l.strip()]
            qq = lines[0] if lines else question
            got = lret.search(qq)
            new = [p for p in got if p["hierarchy"] not in seen_h]
            for p in new:
                seen_h.add(p["hierarchy"])
                evidence.append(p)
            body = "\n".join(f"- {article_label(p['hierarchy'])}: {(p['content'] or '').strip()[:380]}"
                             for p in got) or "(검색 결과 없음)"
            dup = ("" if len(new) == len(got)
                   else f"\n(이 중 {len(got)-len(new)}건은 이미 확보한 조문과 중복)")
            obs = f"[관측·검색결과] 질의: {qq}\n{body}{dup}"
            seq.append(f"검색:{qq[:24]}")
        elif act == "사실검증":
            tools["fact"] += 1
            try:
                cs = extract_citations(draft)
                ck = [verifier.verify(c, answer=draft, strict=True, gold_by_article=gidx)
                      for c in cs]
                obs = "[관측·법제처 사실검증]\n" + m2.obs_fact(cs, ck)
                fact_ok += 1
            except Exception as e:
                obs = f"[관측·법제처 사실검증] 호출 실패: {str(e)[:120]}"
            seq.append("사실검증")
        else:
            tools["logic"] += 1
            rv = ap.exaone_review(cross_client, question, draft, evidence, model=cross_model)
            obs = ("[관측·논리검증] 판정: " + rv["verdict"]
                   + ("; 지적: " + "; ".join(rv["issues"]) if rv["issues"] else " (문제 없음)"))
            seq.append("논리검증")
        messages.append({"role": "user", "content":
                         obs + "\n\n위 관측을 반영해 [생각]과 [답변]을 다시 출력하세요."})

    types = [("검색" if s.startswith("검색") else s) for s in seq]
    max_run = run = 0
    for i, t in enumerate(types):
        run = run + 1 if (i and t == types[i - 1]) else 1
        max_run = max(max_run, run)

    cites = extract_citations(draft)
    checks = [verifier.verify(c, answer=draft, strict=True, gold_by_article=gidx) for c in cites]
    tools["total"] = tools["search"] + tools["fact"] + tools["logic"]
    return {"arm": arm_label, "answer": draft, "cites": cites, "checks": checks,
            "seq": seq, "tools": dict(tools), "steps": steps, "end": end,
            "loop": max_run >= m2.LOOP_RUN, "max_run": max_run,
            "fmt_errors": fmt_errors, "answer_miss": answer_miss,
            "fact_ok": fact_ok, "gate_unmet": fact_ok == 0,
            "skill": skill["name"] if skill else None,
            "transcript": messages}


def route_question(question, gen_client, gen_model, choice_mode):
    msgs = [{"role": "user", "content":
             f"[질문]\n{question}\n\n이 질문의 주된 법 영역을 하나만 고르세요: "
             + " / ".join(ROUTE_OPTS)}]
    return m2.choice_call(gen_client, gen_model, msgs, ROUTE_OPTS, choice_mode)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subset", choices=["target", "control"], required=True)
    p.add_argument("--gen", choices=["qwen", "exaone"], required=True)
    p.add_argument("--cond", choices=["rag", "norag"], required=True)
    p.add_argument("--run", type=int, required=True)
    p.add_argument("--n", type=int, default=None, help="스모크용 부분집합 크기")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--out", default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    use_rag = args.cond == "rag"
    if not config.oc_is_set():
        raise SystemExit("✗ LAW_OC 미설정")

    prefix = args.out or f"main3_{args.subset}_{args.gen}_{args.cond}_r{args.run}"
    jp = config.RESULTS_DIR / f"{prefix}.json"
    if jp.exists() and not args.force:
        print(f"skip (이미 존재): {jp}")
        return

    gcfg = m2.GEN_REG[args.gen]
    ccfg = m2.GEN_REG[gcfg["cross"]]
    gen_client = OpenAI(base_url=gcfg["base"], api_key="EMPTY")
    cross_client = OpenAI(base_url=ccfg["base"], api_key="EMPTY")
    served_g = [m.id for m in gen_client.models.list().data]
    served_c = [m.id for m in cross_client.models.list().data]
    assert gcfg["model"] in served_g, f"✗ 생성 서버 {served_g} ≠ {gcfg['model']} (오염, 중단)"
    assert ccfg["model"] in served_c, f"✗ 논리검증 서버 {served_c} ≠ {ccfg['model']}"
    assert gcfg["family"] != ccfg["family"], "✗ 이종 위반"
    choice_mode = m2.probe_choice_mode(gen_client, gcfg["model"])

    skills = load_skills()
    labels = load_labels()
    want = ("형사", "민사") if args.subset == "target" else ("기타", "mixed")
    items = [it for it in load_koblex(226) if labels.get(it["id"]) in want]
    if args.n:
        items = items[:args.n]

    ap.print = lambda *a, **kw: None
    lret = m2.LockedRetriever(Retriever(), threading.Lock())
    verifier = m2.LockedVerifier(LawVerifier(), threading.Lock())
    INIT = {it["id"]: lret.search(it["prompt"]) for it in items}
    GIDX = {it["id"]: ap.gold_index(INIT[it["id"]]) for it in items}

    m2.log(f"\n== 3번 r{args.run} | {args.subset} n={len(items)} | 생성={gcfg['model']} | "
           f"논리검증={ccfg['model']} | {'RAG' if use_rag else '맨몸'} | choice={choice_mode} | "
           f"skill sha: 형사={skills['형사']['sha'][:12]} 민사={skills['민사']['sha'][:12]} ==")

    rows_lock = threading.Lock()
    rows = []
    done = [0]
    skipped = []
    t0 = time.time()

    def work(it):
        try:
            _work(it)
        except Exception as e:                     # 무인 실행 안전망: 한 문항 예외가 전체를 죽이지 않게
            with rows_lock:
                skipped.append(it["id"])
            m2.log(f"[SKIP {it['id']}] {type(e).__name__}: {str(e)[:120]}")

    def _work(it):
        q = it["prompt"]
        qid = it["id"]
        dom = labels[qid]
        init_provs, gidx = INIT[qid], GIDX[qid]
        common = dict(question=q, lret=lret, gen_client=gen_client, gen_model=gcfg["model"],
                      cross_client=cross_client, cross_model=ccfg["model"], verifier=verifier,
                      gidx=gidx, init_provs=init_provs, use_rag=use_rag, choice_mode=choice_mode)
        # C0 = 2번 코드 경로 그대로
        r0 = m2.run_arm_bc("C", q, lret, gen_client, gcfg["model"], cross_client, ccfg["model"],
                           verifier, gidx, init_provs, use_rag, choice_mode)
        r0["arm"] = "C0"
        r0["skill"] = None
        # C1 = 오라클 라우팅 (라벨 도메인 스킬; 기타/mixed는 스킬 없음)
        r1 = run_arm_skill("C1", skill=skills.get(dom), **common)
        # C2 = 자율 라우팅
        pred = route_question(q, gen_client, gcfg["model"], choice_mode)
        r2 = run_arm_skill("C2", skill=skills.get(pred), **common)
        r2["route_pred"] = pred

        out_rows = []
        for r in (r0, r1, r2):
            m = m2.count_checks(r["checks"])
            gh, gt = gold_recall(r["cites"], r["checks"], it["gold"])
            out_rows.append({"id": qid, "arm": r["arm"], "cond": args.cond, "gen": args.gen,
                             "run": args.run, "domain": dom, "skill": r.get("skill"),
                             "route_pred": r.get("route_pred"),
                             "stages": stage_marks(r.get("transcript") or []),
                             "metrics": dict(m), "gold_hit": gh, "gold_tot": gt,
                             "tools": r["tools"], "steps": r["steps"], "end": r["end"],
                             "loop": r["loop"], "fmt_errors": r["fmt_errors"],
                             "answer_miss": r.get("answer_miss", 0),
                             "fact_ok": r.get("fact_ok", 0), "gate_unmet": r.get("gate_unmet"),
                             "seq": r["seq"], "answer": r["answer"],
                             "checks": m2.checks_dump(r["cites"], r["checks"]),
                             "transcript": r.get("transcript")})
        with rows_lock:
            rows.extend(out_rows)
            done[0] += 1
            i = done[0]
        mm = {r["arm"]: m2.count_checks(r["checks"]) for r in (r0, r1, r2)}
        m2.log(f"[{args.subset[0]}{args.run} {args.cond} {i}/{len(items)}] {qid:<18} {dom:<4} "
               + " | ".join(f"{a} 내✗{mm[a]['mism']}/실✗{mm[a]['fake']}/{mm[a]['n']}"
                            for a in ("C0", "C1", "C2"))
               + f" | C2라우팅={pred}{'✓' if pred == dom else '✗'}")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, items))
    if skipped:
        m2.log(f"⚠ 건너뛴 문항 {len(skipped)}/{len(items)}: {skipped}")

    sums = {a: m2.summarize(rows, a) for a in ("C0", "C1", "C2")}
    route_ok = sum(1 for r in rows if r["arm"] == "C2" and r["route_pred"] == r["domain"])
    n_c2 = sum(1 for r in rows if r["arm"] == "C2")
    m2.log("\n" + "=" * 100)
    m2.log(f"3번 r{args.run} 요약 | {args.subset} | 생성={args.gen} | {'RAG' if use_rag else '맨몸'} | "
           f"n={len(items)} | 채점=lawcheck v1")
    m2.log(f"{'arm':<5}{'인용':>5}{'내용가짜율':>14}{'실존가짜율':>14}{'사실검증률':>9}{'평균툴콜':>8}"
           f"{'자율종료':>8}{'루프':>6}{'형식오류':>8}")
    for a in ("C0", "C1", "C2"):
        s = sums[a]
        m2.log(f"{a:<5}{s['cites']:>5}{s['strict_rate']:>9.1%}({s['strict_frac']})"
               f"{s['loose_rate']:>9.1%}({s['loose_frac']}){s['fact_q_rate']:>9.0%}"
               f"{s['avg_tools']:>8}{s['finish_rate']:>8.0%}{s['loop_rate']:>6.0%}{s['fmt_error_q']:>8}")
    m2.log(f"C2 라우팅 정확도(라벨 대비): {route_ok}/{n_c2} ({route_ok/n_c2:.0%})"
           if n_c2 else "")
    m2.log(f"({time.time()-t0:.0f}s | 법제처 라이브 {verifier.live_calls} / 캐시 {verifier.cache_hits})")

    summary = {"exp": "main3", "subset": args.subset, "gen": args.gen,
               "gen_model": gcfg["model"], "cross_model": ccfg["model"],
               "cond": args.cond, "run": args.run, "n": len(items),
               "choice_mode": choice_mode,
               "skill_sha": {d: s["sha"] for d, s in skills.items()},
               "grader": "lawcheck v1 (gold=초기 top-3, 2번과 동일)",
               "route_acc": (route_ok / n_c2 if n_c2 else None),
               "skipped": skipped, "n_scored": n_c2,
               "arms": sums, "rows": rows}
    jp.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    cp = config.RESULTS_DIR / f"{prefix}.csv"
    with cp.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "domain", "arm", "skill", "route_pred", "stages", "cites", "fake",
                    "sfake", "mism", "gold_hit", "gold_tot", "search", "fact", "logic",
                    "tools_total", "steps", "end", "loop", "fmt_errors", "seq"])
        for r in rows:
            w.writerow([r["id"], r["domain"], r["arm"], r["skill"] or "", r["route_pred"] or "",
                        "".join(map(str, r["stages"])), r["metrics"]["n"], r["metrics"]["fake"],
                        r["metrics"]["sfake"], r["metrics"]["mism"], r["gold_hit"], r["gold_tot"],
                        r["tools"]["search"], r["tools"]["fact"], r["tools"]["logic"],
                        r["tools"]["total"], r["steps"], r["end"], int(bool(r["loop"])),
                        r["fmt_errors"], "→".join(r["seq"])])
    m2.log(f"저장: {jp}\n      {cp}")


if __name__ == "__main__":
    main()

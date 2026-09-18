"""
run_main.py — 본실험(2번 논문): 고정(A) vs 순수자율(B) vs 자율+게이트(C) [신규 측정, 동결 불변].

  Q1. 순수 자율이 고정 대비 환각↑? 경로가 '검증 회피'인가? (n=226 × 3런 분산)
  Q2. 회피가 Qwen 특정(코드스위칭)인가 소형 일반인가? (--gen exaone 페이즈 B가 판정)
  Q3. 종료 전 사실검증 1회+ 강제(게이트)만으로 회피가 치료되는가? (Arm C)

  세 arm — 셀(생성모델×조건) 안에서 오직 제어 흐름만 다름:
    A(고정)      = agent_pipeline.run_agent(cross_client=이종) 그대로 (1번 논문 표3 경로).
    B(순수자율)  = ReAct: 매 스텝 {검색|사실검증|논리검증|종료} 자율 선택. max_steps=8.
                   ★행동은 제약 디코딩(choice enum)으로만 선택 — 제어채널 코드스위칭 원천 차단.
                   생성/수정은 루프 내 [답변] 재작성(액션 아님). 검증 회피는 측정 대상(막지 않음).
    C(게이트)    = B와 완전 동일 + 딱 하나: 사실검증(법제처) 1회+ 성공 전까지 '종료' 마스킹.
                   (시스템 프롬프트는 B와 동일 문구 — 마스크는 선택지 메뉴+제약디코딩에만 반영)

  landmine 방어(파일럿 교훈):
    - 생성 모델 런마다 명시 + 서버 실서빙 assert (config 기본값 EXAONE 오염 방지).
    - 이종 강제: 생성 계열 != 논리검증 계열 assert. 사실검증은 항상 법제처.
    - 채점 lawcheck v1 동결, gold_by_article=초기 top-3(3 arm 동일), temp=0 seed42.
    - 출력 전부 신규(main_*), 동결 파일(koblex_*, crossverify_*) 미접촉.

  실행(런 하나 = 생성×조건×런번호):
    LAW_OC=.. python run_main.py --gen qwen --cond rag --run 1
    (Qwen:8000 + EXAONE:8001 필요. 이미 있으면 skip — run_phaseA.sh가 루프)
"""
import argparse
import collections
import csv
import json
import re
import threading
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

import config
from demo_cli import Retriever, article_label
from lawcheck import extract_citations, LawVerifier, Verdict
import agent_pipeline as ap
from questions import load_koblex
from run_koblex import gold_recall

MAX_STEPS = 8
LOOP_RUN = 3
ACTIONS = ["검색", "사실검증", "논리검증", "종료"]

GEN_REG = {
    "qwen":   {"model": "Qwen/Qwen2.5-7B-Instruct", "base": config.VLLM_BASE_URL,
               "family": "qwen", "cross": "exaone"},
    "exaone": {"model": "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct", "base": config.EXAONE_BASE_URL,
               "family": "exaone", "cross": "qwen"},
}

# B·C 공용(동일 문구 — C의 게이트는 선택지 제시+제약디코딩에서만 다름).
PROTOCOL2 = (
    "\n\n[자율 에이전트 프로토콜]\n"
    "당신은 도구를 스스로 선택해 위 형식의 답변서를 검증·개선하는 자율 에이전트입니다.\n"
    "- 매 턴 [생각](현재 상태 판단, 1~3문장)과 [답변](현재 시점의 완성된 답변서 전체, "
    "【답변 요지】【조문 해설】【결론】)을 출력합니다. 매 턴 답변 전체를 다시 쓰세요.\n"
    "- 그 다음 시스템이 제시하는 행동 선택지 중 하나를 고릅니다"
    "(상황에 따라 일부 선택지만 제시될 수 있음):\n"
    "  검색 = 법조문 데이터베이스에서 관련 조문을 더 확보(근거 부족할 때)\n"
    "  사실검증 = [답변]에 인용된 조문의 실존·내용을 법제처에서 확인\n"
    "  논리검증 = 다른 법률 전문가 모델이 논리·조문적용을 검토\n"
    "  종료 = 답변 확정(충분히 검증되었다고 판단될 때만)")

_print_lock = threading.Lock()


def log(*a):
    with _print_lock:
        print(*a, flush=True)


class LockedRetriever:
    def __init__(self, r, lock):
        self._r, self._l = r, lock

    def search(self, q, k=3):
        with self._l:
            return self._r.search(q, k)


class LockedVerifier:
    """법제처 API 직렬화 + 일시적 오류(404/타임아웃) 재시도.

    lawcheck._get는 429·타임아웃만 재시도하고 404는 즉시 re-raise한다. 그런데 법제처는
    부하 때 간헐적으로 404를 뱉는다(같은 '민법'이 첫 호출 404 → 재시도 OK로 확인). 이 재시도는
    lawcheck v1 판정 로직을 건드리지 않는 순수 네트워크 복원력 — verify를 그대로 다시 부를 뿐이고
    성공 응답은 lawcheck 내부에서 캐시된다. 끝내 실패하면 'uncertain'(분모 제외)로 폴백해
    한 인용의 외부 API 장애가 전체 런을 죽이지 않게 한다(3 arm 동일 적용 → 비교 공정).
    """
    def __init__(self, v, lock):
        self._v, self._l = v, lock

    def verify(self, *a, **kw):
        last = None
        for attempt in range(4):
            try:
                with self._l:
                    return self._v.verify(*a, **kw)
            except (urllib.error.HTTPError, urllib.error.URLError,
                    TimeoutError, RuntimeError) as e:
                last = e
                time.sleep(0.6 * (attempt + 1))
        c = a[0] if a else kw.get("c")
        return Verdict("uncertain", "⚠", None,
                       f"법제처 API 반복 실패(외부 장애, 판정 제외): {str(last)[:80]}")

    def __getattr__(self, n):
        return getattr(self._v, n)


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


# ── LLM 호출 ────────────────────────────────────────────────────────────────
def gen_call(client, model, messages, max_tokens=2048):
    r = client.chat.completions.create(model=model, messages=messages,
                                       temperature=0.0, max_tokens=max_tokens, seed=42)
    return r.choices[0].message.content or ""


def probe_choice_mode(client, model):
    """vLLM 제약 디코딩 API 탐지: guided_choice(구) / structured_outputs(신) / json_schema enum."""
    trials = [
        ("guided_choice", {"guided_choice": ["가", "나"]}, None),
        ("structured_outputs", {"structured_outputs": {"choice": ["가", "나"]}}, None),
        ("json_schema_enum", None,
         {"type": "json_schema",
          "json_schema": {"name": "act", "schema": {"type": "string", "enum": ["가", "나"]}}}),
    ]
    for mode, extra, rf in trials:
        try:
            kw = {}
            if extra:
                kw["extra_body"] = extra
            if rf:
                kw["response_format"] = rf
            r = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": "하나 골라"}],
                temperature=0.0, max_tokens=8, seed=42, **kw)
            out = (r.choices[0].message.content or "").strip()
            if mode == "json_schema_enum":
                out = json.loads(out)
            if out in ("가", "나"):
                return mode
        except Exception:
            continue
    raise SystemExit("✗ 제약 디코딩 API 미지원 — 액션 enum 강제 불가(브리프 요구), 중단")


def choice_call(client, model, messages, options, mode):
    kw = {}
    if mode == "guided_choice":
        kw["extra_body"] = {"guided_choice": list(options)}
    elif mode == "structured_outputs":
        kw["extra_body"] = {"structured_outputs": {"choice": list(options)}}
    else:
        kw["response_format"] = {"type": "json_schema", "json_schema": {
            "name": "act", "schema": {"type": "string", "enum": list(options)}}}
    r = client.chat.completions.create(model=model, messages=messages,
                                       temperature=0.0, max_tokens=16, seed=42, **kw)
    out = (r.choices[0].message.content or "").strip()
    if mode == "json_schema_enum":
        try:
            out = json.loads(out)
        except Exception:
            pass
    return out


def parse_answer(txt):
    """[답변] 이후 전체를 답변으로. 태그 없으면 출력 전체(내용은 측정 대상이므로 버리지 않음)."""
    m = re.search(r"\[\s*답변\s*\]\s*[::]?", txt)
    if m:
        return txt[m.end():].strip(), False
    body = re.sub(r"^\[\s*생각\s*\][^\n]*\n?", "", txt).strip()
    return body, True


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


# ════════════════════════════════════════════════════════════════════════════
# Arm A — 고정 (1번 논문 경로 재사용 + 계측)
# ════════════════════════════════════════════════════════════════════════════
def run_arm_a(question, lret, gen_client, gen_model, cross_client, cross_model,
              verifier, use_rag):
    res = ap.run_agent(question, lret, gen_client, verifier, cross_client=cross_client,
                       use_rag=use_rag, gen_model=gen_model, cross_model=cross_model)
    rounds = len(res["log"])
    last = res["log"][-1]
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
            "seq": seq, "tools": tools, "steps": rounds,
            "end": "finish" if passed else "max_iter", "loop": False,
            "fmt_errors": 0, "answer_miss": 0, "fact_ok": rounds, "gate_unmet": False}


# ════════════════════════════════════════════════════════════════════════════
# Arm B/C — 자율 ReAct (행동=제약 디코딩 enum; C만 종료 마스킹)
# ════════════════════════════════════════════════════════════════════════════
def run_arm_bc(arm, question, lret, gen_client, gen_model, cross_client, cross_model,
               verifier, gidx, init_provs, use_rag, choice_mode):
    base = (ap.rag_prompt(question, init_provs) if use_rag
            else f"[질문] {question}\n\n관련 대한민국 법조문을 '법령명 제N조' 형식으로 인용해 답하세요.")
    messages = [{"role": "system", "content": ap.SYSTEM + PROTOCOL2},
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
        out = gen_call(gen_client, gen_model, messages)
        messages.append({"role": "assistant", "content": out})
        ans, miss = parse_answer(out)
        if miss:
            answer_miss += 1
        if ans:
            draft = ans

        # C 게이트 = B와의 유일한 차이: 사실검증 1회+ 성공 전 '종료' 마스킹 + 그 규칙 고지.
        # (고지 없인 모델이 게이트를 발견할 수 없어 '강제'가 성립 안 함 — 숨은마스크 스모크에서
        #  사실검증 0회·논리검증 루프로 퇴화 확인. 고지는 마스크의 가시화이지 프롬프트 튜닝 아님.)
        gated = (arm == "C" and fact_ok < 1)
        allowed = [a for a in ACTIONS if a != "종료"] if gated else ACTIONS
        menu = "다음 행동을 하나만 선택하세요: " + " / ".join(allowed)
        if gated:
            menu += "\n(참고: '종료'는 사실검증을 1회 이상 수행한 뒤에만 선택할 수 있습니다.)"
        messages.append({"role": "user", "content": menu})
        act = choice_call(gen_client, gen_model, messages, allowed, choice_mode)
        messages.append({"role": "assistant", "content": act})
        if act not in allowed:                      # 제약 디코딩이면 도달 불가(방어선)
            fmt_errors += 1
            seq.append("형식오류")
            messages.append({"role": "user", "content":
                             "잘못된 선택입니다. 제시된 행동 중 하나만 고르세요."})
            continue

        if act == "종료":
            seq.append("종료")
            end = "finish"
            break
        if act == "검색":
            tools["search"] += 1
            messages.append({"role": "user", "content": "검색어를 한 줄로만 출력하세요(설명 없이)."})
            q_out = gen_call(gen_client, gen_model, messages, max_tokens=40)
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
                obs = "[관측·법제처 사실검증]\n" + obs_fact(cs, ck)
                fact_ok += 1                      # 성공 호출 → C 게이트 해제
            except Exception as e:
                obs = f"[관측·법제처 사실검증] 호출 실패: {str(e)[:120]}"
            seq.append("사실검증")
        else:                                     # 논리검증
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
    return {"arm": arm, "answer": draft, "cites": cites, "checks": checks,
            "seq": seq, "tools": dict(tools), "steps": steps, "end": end,
            "loop": max_run >= LOOP_RUN, "max_run": max_run,
            "fmt_errors": fmt_errors, "answer_miss": answer_miss,
            "fact_ok": fact_ok, "gate_unmet": (arm == "C" and fact_ok == 0),
            "transcript": messages}


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
        "fact_q_rate": sum(r["tools"]["fact"] > 0 for r in rs) / n,
        "logic_q_rate": sum(r["tools"]["logic"] > 0 for r in rs) / n,
        "avg_fact": round(sum(r["tools"]["fact"] for r in rs) / n, 2),
        "avg_logic": round(sum(r["tools"]["logic"] for r in rs) / n, 2),
        "avg_search": round(sum(r["tools"]["search"] for r in rs) / n, 2),
        "avg_tools": round(sum(r["tools"]["total"] for r in rs) / n, 2),
        "avg_steps": round(sum(r["steps"] for r in rs) / n, 2),
        "finish_rate": sum(r["end"] == "finish" for r in rs) / n,
        "loop_rate": sum(bool(r["loop"]) for r in rs) / n,
        "fmt_error_q": sum(r["fmt_errors"] > 0 for r in rs),
        "answer_miss_q": sum(r.get("answer_miss", 0) > 0 for r in rs),
        "gate_unmet_q": sum(bool(r.get("gate_unmet")) for r in rs),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gen", choices=["qwen", "exaone"], required=True)
    p.add_argument("--cond", choices=["rag", "norag"], required=True)
    p.add_argument("--run", type=int, required=True, help="런 번호(1..3; 스모크는 9)")
    p.add_argument("--n", type=int, default=226)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--arms", default="ABC")
    p.add_argument("--out", default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    use_rag = args.cond == "rag"
    if not config.oc_is_set():
        raise SystemExit("✗ LAW_OC 미설정")

    prefix = args.out or f"main_{args.gen}_{args.cond}_r{args.run}"
    jp = config.RESULTS_DIR / f"{prefix}.json"
    if jp.exists() and not args.force:
        print(f"skip (이미 존재): {jp}")
        return

    # ── landmine 방어: 생성 모델 명시 + 실서빙 assert + 이종 assert ──
    gcfg = GEN_REG[args.gen]
    ccfg = GEN_REG[gcfg["cross"]]
    gen_client = OpenAI(base_url=gcfg["base"], api_key="EMPTY")
    cross_client = OpenAI(base_url=ccfg["base"], api_key="EMPTY")
    served_g = [m.id for m in gen_client.models.list().data]
    served_c = [m.id for m in cross_client.models.list().data]
    assert gcfg["model"] in served_g, f"✗ 생성 서버가 {served_g} 서빙 — {gcfg['model']} 아님(오염, 중단)"
    assert ccfg["model"] in served_c, f"✗ 논리검증 서버가 {served_c} 서빙 — {ccfg['model']} 아님(중단)"
    assert gcfg["model"] != ccfg["model"], "✗ 생성==논리검증 (자기검증 금지)"
    assert gcfg["family"] != ccfg["family"], "✗ 생성·논리검증 동일 계열 (이종 위반)"
    choice_mode = probe_choice_mode(gen_client, gcfg["model"])

    ap.print = lambda *a, **kw: None          # run_agent 내부 print 소음 차단(병렬)
    retr_lock, ver_lock = threading.Lock(), threading.Lock()
    lret = LockedRetriever(Retriever(), retr_lock)
    verifier = LockedVerifier(LawVerifier(), ver_lock)
    items = load_koblex(args.n)

    # 초기 top-3 검색 사전계산 — 3 arm·전 런 공통(채점 gold도 여기서)
    INIT = {it["id"]: lret.search(it["prompt"]) for it in items}
    GIDX = {it["id"]: ap.gold_index(INIT[it["id"]]) for it in items}

    log(f"\n== 본실험 r{args.run} | 생성={gcfg['model']} | 논리검증={ccfg['model']} | "
        f"{'RAG' if use_rag else '맨몸'} | n={len(items)} | arms={args.arms} | "
        f"choice={choice_mode} | workers={args.workers} ==")

    rows_lock = threading.Lock()
    rows = []
    done = [0]
    t0 = time.time()

    def work(it):
        q = it["prompt"]
        init_provs, gidx = INIT[it["id"]], GIDX[it["id"]]
        res = {}
        if "A" in args.arms:
            res["A"] = run_arm_a(q, lret, gen_client, gcfg["model"], cross_client,
                                 ccfg["model"], verifier, use_rag)
        if "B" in args.arms:
            res["B"] = run_arm_bc("B", q, lret, gen_client, gcfg["model"], cross_client,
                                  ccfg["model"], verifier, gidx, init_provs, use_rag, choice_mode)
        if "C" in args.arms:
            res["C"] = run_arm_bc("C", q, lret, gen_client, gcfg["model"], cross_client,
                                  ccfg["model"], verifier, gidx, init_provs, use_rag, choice_mode)
        out_rows = []
        for r in res.values():
            m = count_checks(r["checks"])
            gh, gt = gold_recall(r["cites"], r["checks"], it["gold"])
            out_rows.append({"id": it["id"], "arm": r["arm"], "cond": args.cond,
                             "gen": args.gen, "run": args.run,
                             "metrics": dict(m), "gold_hit": gh, "gold_tot": gt,
                             "tools": r["tools"], "steps": r["steps"], "end": r["end"],
                             "loop": r["loop"], "max_run": r.get("max_run", 0),
                             "fmt_errors": r["fmt_errors"], "answer_miss": r.get("answer_miss", 0),
                             "fact_ok": r.get("fact_ok", 0), "gate_unmet": r.get("gate_unmet", False),
                             "seq": r["seq"], "answer": r["answer"],
                             "checks": checks_dump(r["cites"], r["checks"]),
                             "transcript": r.get("transcript")})
        with rows_lock:
            rows.extend(out_rows)
            done[0] += 1
            i = done[0]
        parts = []
        for k in "ABC":
            if k in res:
                mm = count_checks(res[k]["checks"])
                parts.append(f"{k} ✗{mm['fake']}/{mm['n']} 툴{res[k]['tools']['total']} "
                             f"{res[k]['end'][:6]}")
        log(f"[r{args.run} {args.cond} {i}/{len(items)}] {it['id']:<18} " + " | ".join(parts))

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, items))

    sums = {a: summarize(rows, a) for a in args.arms}
    log("\n" + "=" * 100)
    log(f"r{args.run} 요약 | 생성={args.gen} | {'RAG' if use_rag else '맨몸'} | n={len(items)} | 채점=lawcheck v1")
    log(f"{'arm':<10}{'인용':>5}{'느슨가짜율':>14}{'엄격가짜율':>14}{'검증호출률':>9}{'평균툴콜':>8}"
        f"{'정상종료':>8}{'루프':>6}{'형식오류':>8}")
    for a in args.arms:
        s = sums[a]
        log(f"{a:<10}{s['cites']:>5}{s['loose_rate']:>9.1%}({s['loose_frac']})"
            f"{s['strict_rate']:>9.1%}({s['strict_frac']}){s['fact_q_rate']:>9.0%}"
            f"{s['avg_tools']:>8}{s['finish_rate']:>8.0%}{s['loop_rate']:>6.0%}{s['fmt_error_q']:>8}")
    log(f"({time.time()-t0:.0f}s | 법제처 라이브 {verifier.live_calls} / 캐시 {verifier.cache_hits})")

    summary = {"gen": args.gen, "gen_model": gcfg["model"], "cross_model": ccfg["model"],
               "cond": args.cond, "run": args.run, "n": len(items), "max_steps": MAX_STEPS,
               "choice_mode": choice_mode,
               "grader": "lawcheck v1 (gold=초기 top-3, 3arm 동일)",
               "arms": {a: sums[a] for a in args.arms}, "rows": rows}
    jp.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    cp = config.RESULTS_DIR / f"{prefix}.csv"
    with cp.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "arm", "cites", "fake", "sfake", "mism", "gold_hit", "gold_tot",
                    "search", "fact", "logic", "tools_total", "steps", "end", "loop",
                    "fmt_errors", "answer_miss", "gate_unmet", "seq"])
        for r in rows:
            w.writerow([r["id"], r["arm"], r["metrics"]["n"], r["metrics"]["fake"],
                        r["metrics"]["sfake"], r["metrics"]["mism"], r["gold_hit"], r["gold_tot"],
                        r["tools"]["search"], r["tools"]["fact"], r["tools"]["logic"],
                        r["tools"]["total"], r["steps"], r["end"], int(bool(r["loop"])),
                        r["fmt_errors"], r["answer_miss"], int(bool(r["gate_unmet"])),
                        "→".join(r["seq"])])
    log(f"저장: {jp}\n      {cp}")


if __name__ == "__main__":
    main()

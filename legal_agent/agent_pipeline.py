"""
agent_pipeline.py — 법률 QA 에이전트 (검색 → 생성 → 검증+자율수정 → docx MCP).

  ① 검색  : KoE5 RAG (build_retrieval/demo_cli.Retriever) — 항상 ON
  ② 생성  : EXAONE-3.5-7.8B(vLLM, 한국어 네이티브 → 코드스위칭 없음) + 근거 조문 주입
            ※ 측정 스크립트는 config.MODEL(Qwen) 그대로 — gen_model 인자로만 전환(동결 유지)
  ③ 검증 + 자율 수정 루프 (핵심): lawcheck.verify 로 실존·내용 검증 →
            ✗(가짜/내용불일치) 발견 시 '검색된 근거 조문을 다시 보라'고 피드백 →
            재생성 → 재검증. 통과 또는 max_iter 까지. (RAG 접지 상태의 수정)
  ④ 출력  : docx_mcp_server(MCP, stdio)의 create_legal_document 도구 호출 → 워드 생성.

  실행: python agent_pipeline.py "타인의 명예를 훼손하면 어떤 책임을 지나요?"
        (vLLM + LAW_OC + 임베딩 캐시 필요)
"""
import os
import re
import sys
import json
import asyncio

from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import config
from demo_cli import Retriever, article_label, title_from_note
from build_retrieval import hier_key
from lawcheck import extract_citations, LawVerifier

MAX_ITER = 2
SYSTEM = (
    "당신은 대한민국 법률 전문가입니다. 반드시 한국어로만(영어·중국어·한자 단어 혼용 금지) 아래 형식을 "
    "정확히 지켜 충실하고 상세한 공식 법률 답변서를 작성하세요.\n\n"
    "【답변 요지】\n핵심 결론과 그 성립·적용 요건(어떤 경우에 성립하고 어떤 경우는 그렇지 않은지)을 "
    "3~4문장으로 구체적으로 쓰세요.\n\n"
    "【조문 해설】\n관련 조문마다 **정확히 한 줄로**, '- 법령명 제N조(조문제목): ' 로 시작해 "
    "콜론(:) 뒤에는 별표나 어떤 기호도 없이 바로 설명 문장을 시작하세요. 한 조문의 설명에는 "
    "그 조문이 규정하는 내용, 이 질문 사안에 어떻게 적용되는지, "
    "핵심 요건과 효과(형량·책임·기간 등), 그리고 조문에 예외나 유의할 점이 있으면 그것까지를 "
    "각각 한 문장 이상으로 충분히 풀어, **질문과 직접 관련된 핵심 조문은 4~5문장**으로 자세히 쓰세요"
    "(부차적이거나 간접적으로만 관련된 조문은 2~3문장으로 짧게). 단 **항목 기호((a)(b)(c)(d)·①②③·1.2.3. 등)나 소제목, 줄바꿈, "
    "별표(*)·마크다운 없이, 번호를 붙이지 말고 자연스러운 평문 문장으로 이어서** 한 줄로 쓰세요"
    "(한 조문 = 한 줄). **조문 원문에 없는 내용은 절대 지어내지 말고, 주어진 '참고 조문' 범위 "
    "안에서만** 설명하세요(분량을 늘리려고 없는 내용을 보태지 마세요). 질문과 직접 관련된 조문 위주로 다루세요.\n\n"
    "【결론】\n사안에 대한 종합 판단, 실무적 유의점, 그리고 관련되면 절차나 구제 방법을 "
    "3~4문장으로 쓰세요.\n\n"
    "규칙: 주어진 '참고 조문' 중 질문과 직접 관련된 것만 인용하고, 거기 없는 새 조문을 끌어오지 마세요. "
    "매번 '법령명 제N조' 형식으로(‘제3조’처럼 번호만, ‘③’처럼 항번호만 쓰지 마세요).")


def section(text, name):
    """【name】 섹션 본문 추출 (다음 【 또는 끝까지)."""
    m = re.search(rf"【\s*{re.escape(name)}\s*】", text)
    if not m:
        return ""
    start = m.end()
    nxt = re.search(r"【", text[start:])
    return (text[start:start + nxt.start()] if nxt else text[start:]).strip()


def explain_map(haeseol, cites):
    """【조문 해설】 블록의 각 줄을 (조,가지) → 해설텍스트 로 매핑."""
    out = {}
    for line in haeseol.splitlines():
        line = line.strip()
        m = re.search(r"제\s*(\d+)\s*조(?:의\s*(\d+))?", line)
        if not m:
            continue
        key = (int(m.group(1)), int(m.group(2) or 0))
        parts = re.split(r"[:：]", line, maxsplit=1)
        expl = parts[1].strip() if len(parts) == 2 else re.sub(r"^[-•·\s]+", "", line)
        out.setdefault(key, expl)
    return out


def generate(client, prompt, extra="", model=None):
    r = client.chat.completions.create(
        model=model or config.MODEL,
        messages=[{"role": "system", "content": SYSTEM + extra},
                  {"role": "user", "content": prompt}],
        temperature=0.0, max_tokens=2048, seed=42)
    return r.choices[0].message.content or ""


def rag_context(provs):
    return "\n".join(f"- {article_label(p['hierarchy'])}: {(p['content'] or '').strip()[:380]}"
                     for p in provs)


def rag_prompt(question, provs):
    return ("[참고 조문] 아래 검색된 조문 중 관련된 것만 근거로 인용하라. 여기 없는 조문을 지어내지 마라.\n"
            + rag_context(provs) + f"\n\n[질문] {question}")


def gold_index(provs):
    g = {}
    for p in provs:
        hk = hier_key(p["hierarchy"])
        if hk:
            g[hk] = (p["content"] or "") + " " + p["hierarchy"]
    return g


def verify_answer(verifier, answer, gold):
    cites = extract_citations(answer)
    checks = [verifier.verify(c, answer=answer, strict=True, gold_by_article=gold) for c in cites]
    return cites, checks


def is_bad(v):
    return v.status == "fake" or (v.status == "real" and v.content == "mismatch")


# ── ③-b 이종 모델 교차검증 (EXAONE-3.5-7.8B) ──────────────────────────────────
# 역할: 논리·조문적용 지적만. 사실(실존/내용)은 법제처가 최종 판단(아래 run_agent의 cites/checks).
# 안전장치: EXAONE은 cites/checks(측정값)를 절대 바꾸지 않음 — 재생성 트리거+피드백 텍스트에만 반영.
EXAONE_SYS = (
    "당신은 한국 법률 답변을 교차검증하는 또 다른 법률 전문가입니다. [질문], [답변], [검색된 근거 조문]을 "
    "보고 답변의 논리와 조문 적용을 검토하세요. 조문의 실존 여부·내용 정확성은 법제처가 별도로 최종 판단하니, "
    "당신은 (1) 인용 조문이 이 질문 상황에 적절히 적용됐는지 (2) 답변 논리에 비약·오류·모순이 있는지만 "
    "지적하세요. 반드시 아래 JSON 형식만 출력(다른 말 금지):\n"
    '{"verdict": "ok|revise", "issues": ["문제1","문제2"]}\n'
    "문제 없으면 verdict=ok, issues=[]. 사소한 트집 금지, 실제 오류만.")


def exaone_review(cross_client, question, answer, provs, model=None):
    ctx = "\n".join(f"- {article_label(p['hierarchy'])}: {(p.get('content') or '').strip()[:300]}"
                    for p in provs)
    user = (f"[질문]\n{question}\n\n[답변]\n{answer}\n\n[검색된 근거 조문]\n{ctx}\n\n"
            "위 답변의 논리·조문적용을 교차검증해 JSON으로만 답하세요.")
    try:
        r = cross_client.chat.completions.create(
            model=model or config.EXAONE_MODEL,
            messages=[{"role": "system", "content": EXAONE_SYS},
                      {"role": "user", "content": user}],
            temperature=0.0, max_tokens=400, seed=42)
        txt = r.choices[0].message.content or ""
    except Exception as e:
        return {"verdict": "ok", "issues": [], "error": str(e)[:120]}
    obj = {}
    m = re.search(r"\{.*\}", txt, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
        except Exception:
            obj = {}
    issues = obj.get("issues", [])
    if not isinstance(issues, list):
        issues = [str(issues)]
    issues = [str(x).strip() for x in issues if str(x).strip()][:5]
    verdict = obj.get("verdict", "ok")
    if verdict not in ("ok", "revise"):
        verdict = "revise" if issues else "ok"
    return {"verdict": verdict, "issues": issues}


def run_agent(question, retriever, client, verifier, cross_client=None, use_rag=True,
              gen_model=None, cross_model=None):
    print(f"\n[①검색] KoE5로 근거 조문 검색...", flush=True)
    provs = retriever.search(question)
    print("    근거:", ", ".join(article_label(p["hierarchy"]) for p in provs))
    gold = gold_index(provs)
    base = (rag_prompt(question, provs) if use_rag
            else f"[질문] {question}\n\n관련 대한민국 법조문을 '법령명 제N조' 형식으로 인용해 답하세요.")

    print("[②생성] 근거 주입 후 답변 생성...", flush=True)
    answer = generate(client, base, model=gen_model)

    log = []
    for it in range(MAX_ITER + 1):
        cites, checks = verify_answer(verifier, answer, gold)          # ③-a 법제처 사실검증 (측정의 심판)
        bad = [(c, v) for c, v in zip(cites, checks) if is_bad(v)]
        bad_str = [f"{(c.law_name or '')} {c.display}"
                   f"({'미존재' if v.status == 'fake' else '내용불일치'})" for c, v in bad]

        review = None                                                 # ③-b 이종 모델 교차검증 (논리·적용)
        if cross_client is not None:
            review = exaone_review(cross_client, question, answer, provs, model=cross_model)
        ex_revise = bool(review and review["verdict"] == "revise")

        log.append({"iter": it,
                    "citations": [f"{(v.official_name or c.law_name or '(미지정)')} {c.display}"
                                  for c, v in zip(cites, checks)],
                    "bad": bad_str,
                    "exaone": ({"verdict": review["verdict"], "issues": review["issues"]}
                               if review else None)})
        tag = "[③검증]" if it == 0 else f"[③재검증 {it}]"
        exmsg = (f" | 교차검증={review['verdict']}"
                 + (f"({len(review['issues'])})" if review["issues"] else "")) if review else ""
        print(f"{tag} 인용 {len(cites)} | 법제처✗ {len(bad)}{exmsg}"
              + (" → 통과" if (not bad and not ex_revise) else ""), flush=True)

        if (not bad and not ex_revise) or it == MAX_ITER:
            break

        fb_parts = []
        if bad:
            fb_parts.append("[법제처 사실검증] 다음 인용이 실존하지 않거나 내용이 다릅니다: "
                            + "; ".join(bad_str) + ". 제거하거나 실존·정확한 조문으로 정정하세요.")
        if ex_revise:
            fb_parts.append("[교차검증·EXAONE] 논리·조문적용 문제 지적: "
                            + "; ".join(review["issues"]) + ". 이 부분을 바로잡으세요.")
        fb = ("\n\n[검증 결과 — 자율 수정 요청]\n" + "\n".join(fb_parts)
              + ("\n아래 '검색된 근거 조문'을 다시 보고 답을 다시 작성하세요:\n" + rag_context(provs)
                 if use_rag else "\n위 지적을 반영해 답을 다시 작성하세요."))
        src = "법제처+교차검증" if (bad and ex_revise) else ("법제처" if bad else "교차검증")
        print(f"    ↳ [자율수정 {it+1}] {src} 피드백으로 재생성...", flush=True)
        answer = generate(client, base, extra=fb, model=gen_model)

    return {"question": question, "answer": answer, "provs": provs,
            "cites": cites, "checks": checks, "log": log}


def build_payload(res):
    answer = res["answer"]
    cites, checks = res["cites"], res["checks"]
    gist = section(answer, "답변 요지") or answer.strip()[:200]
    conclusion = section(answer, "결론") or gist   # 모델이 결론 섹션 누락 시 요지로 폴백
    expl = explain_map(section(answer, "조문 해설"), cites)

    citations = []
    seen = set()
    for c, v in zip(cites, checks):
        ttl = title_from_note(v.note)
        label = f"{(v.official_name or c.law_name or '(미지정)')} {c.display}"
        if ttl and v.status == "real":
            label += f"({ttl})"
        if label in seen:        # 동일 조문 중복 표시 제거
            continue
        seen.add(label)
        citations.append({
            "label": label, "status": v.status, "content": v.content,
            "article_text": getattr(v, "article_text", "") or "",   # 법제처 원문(코드 주입)
            "explanation": expl.get((c.jo, c.jo_branch), ""),        # 모델 해설
        })
    summary = {
        "n_total": len(citations),
        "n_real": sum(x["status"] == "real" for x in citations),
        "n_match": sum(x["status"] == "real" and x["content"] == "match" for x in citations),
        "n_content": sum(x["status"] == "real" and x["content"] in ("match", "mismatch")
                         for x in citations),
        "n_fake": sum(x["status"] == "fake" for x in citations),
        "iterations": len(res["log"]) - 1,
    }
    return {"question": res["question"], "gist": gist, "conclusion": conclusion,
            "citations": citations, "summary": summary}


async def make_docx(payload, out_path):
    payload = dict(payload, out_path=out_path)
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(config.ROOT / "docx_mcp_server.py")],
        env=dict(os.environ))
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = [t.name for t in (await session.list_tools()).tools]
            print(f"[④출력] MCP 서버 연결 | 도구: {tools}", flush=True)
            result = await session.call_tool("create_legal_document", payload)
            if result.isError:
                raise RuntimeError(f"MCP 도구 오류: {result.content}")
            for blk in result.content:
                if getattr(blk, "text", None):
                    try:
                        return json.loads(blk.text)        # {"docx":, "pdf":}
                    except Exception:
                        return {"docx": blk.text, "pdf": None}
            return None


def slug(q):
    s = re.sub(r"[^0-9A-Za-z가-힣]+", "_", q).strip("_")
    return s[:30] or "query"


def main():
    question = " ".join(sys.argv[1:]).strip() or "타인의 명예를 훼손하면 어떤 책임을 지나요?"
    # 생성 = EXAONE(한국어 네이티브 → 코드스위칭 없음)
    client = OpenAI(base_url=config.GEN_BASE_URL, api_key="EMPTY")
    try:
        client.models.list()
    except Exception:
        sys.exit("✗ 생성기(EXAONE) 미실행. vllm serve LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct "
                 "--port 8001 --trust-remote-code")
    if not config.oc_is_set():
        sys.exit("✗ OC 키 미설정: export LAW_OC=...")

    # 교차검증 = Qwen(이종 모델, :8000)이 떠 있으면 사용, 없으면 단일 모델
    cross_client = OpenAI(base_url=config.CROSS_BASE_URL, api_key="EMPTY")
    try:
        cross_client.models.list()
        print(f"[교차검증] {config.CROSS_NAME} 사용 (이종 모델)")
    except Exception:
        cross_client = None
        print("[교차검증] 교차모델 미실행 → 단일 모델 자기검증")

    retriever = Retriever()
    verifier = LawVerifier()
    res = run_agent(question, retriever, client, verifier, cross_client=cross_client,
                    gen_model=config.GEN_MODEL, cross_model=config.CROSS_MODEL)

    s = build_payload(res)["summary"]
    print(f"\n[요약] 실존 {s['n_real']}/{s['n_total']} | 내용일치 {s['n_match']}/{s['n_content']} "
          f"| 환각 {s['n_fake']} | 자율수정 {s['iterations']}회")

    payload = build_payload(res)
    out = config.ROOT / "documents" / f"법률답변서_{slug(question)}.docx"
    paths = asyncio.run(make_docx(payload, str(out)))
    print(f"\n✓ 워드: {paths.get('docx')}")
    print(f"✓ PDF : {paths.get('pdf') or '(변환 실패/생략)'}")


if __name__ == "__main__":
    main()

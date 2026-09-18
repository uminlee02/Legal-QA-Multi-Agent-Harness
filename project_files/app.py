"""
app.py — 법제처 근거 법률 QA 에이전트 웹앱 (FastAPI). 기존 부품 재사용.

  POST /api/ask  {question}  → agent_pipeline(검색→생성→검증→자율수정) → 검증된 답변서 JSON + docx
  GET  /                      → legal_qa_agent_demo.html
  GET  /documents/{name}      → 생성된 .docx 다운로드

  전제: vLLM(OpenAI 호환 :8000) 가 떠 있어야 함. 검증기 동결, 조문 원문은 법제처 article_text.

  실행: ./run_app.sh   (또는 uvicorn app:app --host 0.0.0.0 --port 7860)
"""
import os
import json
import threading
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from openai import OpenAI

import config
from demo_cli import Retriever
from lawcheck import LawVerifier
import agent_pipeline as ap
from agent_pipeline import section, explain_map, title_from_note, slug, article_label
from docx_mcp_server import create_legal_document

DOCS = config.ROOT / "documents"
DOCS.mkdir(exist_ok=True)
HTML = config.ROOT / "legal_qa_agent_demo.html"

_state = {}
_lock = threading.Lock()   # GPU(vLLM) 직렬화 — MAD와 공유하므로 동시 처리 방지


def get_state():
    if not _state:
        # 생성 = EXAONE(한국어 네이티브 → 코드스위칭 없음)
        _state["client"] = OpenAI(base_url=config.GEN_BASE_URL, api_key="EMPTY")
        _state["retriever"] = Retriever()        # KoE5 + 233k 임베딩 (~15s, 1회)
        _state["verifier"] = LawVerifier()
        # 이종 교차검증 = Qwen(:8000)이 떠 있으면 사용
        cross = OpenAI(base_url=config.CROSS_BASE_URL, api_key="EMPTY")
        try:
            cross.models.list()
            _state["exaone"] = cross           # (키名 유지) 교차검증 클라이언트
        except Exception:
            _state["exaone"] = None
    return _state


@asynccontextmanager
async def lifespan(app):
    if config.oc_is_set():
        get_state()                              # 서버 시작 시 1회 웜업
    yield


app = FastAPI(title="법제처 근거 법률 QA 에이전트", lifespan=lifespan)


class AskReq(BaseModel):
    question: str
    dryrun: bool = False           # 드라이런: 검증까지만, 문서(docx) 생성 생략


def _tri_exist(status):
    return True if status == "real" else (False if status == "fake" else None)


def _tri_content(status, content):
    if status != "real":
        return None
    return True if content == "match" else (False if content == "mismatch" else None)


def to_api(res):
    """agent_pipeline 결과 → 프론트 계약 JSON."""
    answer = res["answer"]
    gist = section(answer, "답변 요지") or answer.strip()[:200]
    conclusion = section(answer, "결론") or gist
    expl = explain_map(section(answer, "조문 해설"), res["cites"])

    provisions, seen = [], set()
    for c, v in zip(res["cites"], res["checks"]):
        law = v.official_name or c.law_name
        key = ((law or ""), c.display)
        if key in seen:
            continue
        seen.add(key)
        provisions.append({
            "law": law,
            "article": c.display,
            "title": title_from_note(v.note) if v.status == "real" else "",
            "exist": _tri_exist(v.status),                       # 실존 검증
            "content": _tri_content(v.status, v.content),        # 내용일치 검증
            "text": getattr(v, "article_text", "") or "",        # 법제처 조문 원문
            "note": expl.get((c.jo, c.jo_branch), ""),           # 해설(생성)
        })
    return {
        "question": res["question"],
        "summary": gist,                 # 답변 요지(생성)
        "conclusion": conclusion,        # 결론(생성)
        "self_correct": len(res["log"]) - 1,
        "provisions": provisions,
    }


def build_tool_log(res, docx_name, dryrun=False, pdf_name=None):
    """파이프라인을 MCP/툴 호출 트레이스로 (선배 그림4 JSON 스타일)."""
    rows = []

    def add(agent, tool, phase, status, args, excerpt):
        rows.append({"agent": agent, "tool": tool, "phase": phase, "status": status,
                     "arguments": args, "result_excerpt": excerpt})

    provs = res["provs"]
    q = res["question"]
    add("M1·retriever", "koe5_search", "run", "ok",
        {"query": q, "top_k": len(provs), "corpus": "법제처 조문 233k"},
        f"근거 조문 {len(provs)}건: " + ", ".join(article_label(p["hierarchy"]) for p in provs))
    add("M2·generator", "exaone_generate", "run", "ok",
        {"model": config.GEN_NAME, "grounded": True, "max_tokens": 2048},
        "검색 근거 주입 후 답변 초안 생성 (한국어 네이티브)")
    for c, v in zip(res["cites"], res["checks"]):
        name = f"{(v.official_name or c.law_name or '(미지정)')} {c.display}"
        ex = {"real": "실존 ✓", "fake": "미존재 ✗"}.get(v.status, "확인필요 ⚠")
        ct = ({"match": " · 내용일치 ✓", "mismatch": " · 내용불일치 ✗"}.get(v.content, "")
              if v.status == "real" else "")
        add("M3·verifier", "law_verify", "run", "ok" if v.status != "fake" else "fail",
            {"law": v.official_name or c.law_name, "article": c.display,
             "source": "법제처 국가법령정보 API"},
            f"{name} → {ex}{ct}")
    # ③-b 이종 교차검증 (EXAONE) — 라운드별
    for e in res["log"]:
        rev = e.get("exaone")
        if rev:
            add("M4·cross-verifier", "cross_review", "run",
                "ok" if rev["verdict"] == "ok" else "revise",
                {"model": config.CROSS_NAME, "round": e["iter"],
                 "issues": rev["issues"]},
                f"논리·조문적용 검토 → {rev['verdict']}"
                + (f" : {'; '.join(rev['issues'][:2])}" if rev["issues"] else " (문제 없음)"))
    # 자율수정 트리거 (법제처 OR EXAONE)
    for e in res["log"][:-1]:
        rev = e.get("exaone")
        ex_rev = bool(rev and rev["verdict"] == "revise")
        if e["bad"] or ex_rev:
            src = ([f"법제처 {len(e['bad'])}건"] if e["bad"] else []) + \
                  ([f"교차검증 {len(rev['issues'])}건"] if ex_rev else [])
            add("M2·generator", "self_correct", "run", "retry",
                {"trigger": src, "hallucinated": e["bad"]},
                f"{' + '.join(src)} 피드백 → 근거 재참조 후 재생성")
    if not dryrun:
        fmts = ["docx"] + (["pdf"] if pdf_name else [])
        add("M5·writer", "create_legal_document", "run", "ok",
            {"server": "docx-MCP", "formats": fmts, "docx": docx_name, "pdf": pdf_name},
            f"{docx_name} 생성" + (f" + {pdf_name}" if pdf_name else " (PDF 생략)"))

    base = datetime.now()
    for i, r in enumerate(rows):
        t = base + timedelta(milliseconds=i * 130)
        r["ts"] = round(t.timestamp(), 6)
        r["ts_iso"] = t.strftime("%Y-%m-%dT%H:%M:%S")
    return rows


@app.post("/api/ask")
def ask(req: AskReq):
    q = (req.question or "").strip()
    if not q:
        raise HTTPException(400, "question 이 비어 있습니다.")
    if not config.oc_is_set():
        raise HTTPException(500, "LAW_OC 미설정 (export LAW_OC=...)")
    st = get_state()
    try:
        st["client"].models.list()
    except Exception:
        raise HTTPException(503, "생성기(EXAONE :8001) 미실행 — vllm serve "
                                 "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct --port 8001 --trust-remote-code")

    with _lock:                          # vLLM 직렬화
        res = ap.run_agent(q, st["retriever"], st["client"], st["verifier"],
                           cross_client=st.get("exaone"),
                           gen_model=config.GEN_MODEL, cross_model=config.CROSS_MODEL)

    data = to_api(res)
    if req.dryrun:
        data["tool_log"] = build_tool_log(res, "", dryrun=True)
        data["docx_url"] = None
        data["pdf_url"] = None
        return data
    # docx (+pdf) 생성 (백엔드) → 다운로드 링크
    payload = ap.build_payload(res)
    out = str(DOCS / f"법률답변서_{slug(q)}.docx")
    paths = json.loads(create_legal_document(
        question=payload["question"], gist=payload["gist"],
        conclusion=payload["conclusion"], citations=payload["citations"],
        summary=payload["summary"], out_path=out, out_format="both"))
    docx_name = os.path.basename(paths["docx"])
    pdf_name = os.path.basename(paths["pdf"]) if paths.get("pdf") else None
    data["tool_log"] = build_tool_log(res, docx_name, pdf_name=pdf_name)
    data["docx_url"] = "/documents/" + docx_name
    data["pdf_url"] = ("/documents/" + pdf_name) if pdf_name else None
    return data


@app.get("/documents/{name}")
def get_doc(name: str):
    p = DOCS / os.path.basename(name)
    if not p.exists():
        raise HTTPException(404, "문서 없음")
    mime = ("application/pdf" if p.suffix.lower() == ".pdf"
            else "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    return FileResponse(p, filename=p.name, media_type=mime)


@app.get("/")
def index():
    if not HTML.exists():
        raise HTTPException(404, "legal_qa_agent_demo.html 없음")
    return FileResponse(HTML, media_type="text/html")


@app.get("/health")
def health():
    return {"ok": True, "oc": config.oc_is_set(), "warmed": bool(_state)}

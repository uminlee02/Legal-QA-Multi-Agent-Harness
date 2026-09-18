"""web/app.py — LangGraph 법률 QA 하네스 웹 노출(FastAPI). 새 로직 없음, 기존 파이프라인 래핑만.

  POST /api/ask {question, mode}  → build_targeted_graph(⑥)/build_proposed_graph(③) 호출 → 검증 답변서 JSON + docx/pdf
  GET  /                          → index.html
  GET  /documents/{name}          → 생성된 docx/pdf 다운로드
  GET  /config, /health           → 구성/상태

  전제: vLLM 생성(:8010) + 검증(:8011) 기동. 법제처 실제 API(LAW_OC) — 없으면 mock 폴백(화면 표기).
  실행: ../run_web.sh  (vLLM + uvicorn + cloudflared)
"""
import os
import sys
import pathlib
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

_HERE = pathlib.Path(__file__).resolve().parent          # legal_agent/web
_LA = _HERE.parent                                        # legal_agent
for p in (str(_LA), str(_LA.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

import config_lg as C
from formatting import section, summarize_citations
from graph import build_targeted_graph, build_proposed_graph, initial_state

DOCS = C.DOCS_DIR
HTML = _HERE / "index.html"

_state = {}
_lock = threading.Lock()          # vLLM/검증기 직렬화 (GPU MAD 공유 → 동시 처리 방지)


class _MockVerifier:
    """LAW_OC 미설정 시 폴백 — 모든 인용 'uncertain'(판정 제외). 화면에 mock 표기."""
    def verify(self, c, answer=None, strict=False, gold_by_article=None):
        from lawcheck import Verdict
        return Verdict("uncertain", "⚠", c.law_name, f"{c.display} (mock — 법제처 미조회)")


def get_state():
    if not _state:
        from runtime import Runtime
        rt = Runtime()                          # KoE5 검색기 + 법제처 검증기 + LLM 클라이언트 (1회 웜업)
        if not C.oc_is_set():
            rt.verifier = _MockVerifier()
            _state["law_api"] = "mock"
        else:
            _state["law_api"] = "real"
        _state["rt"] = rt
        # 사용자용 실서비스 = 항상 최고 성능 구성(⑥ 타겟 편집). 비교 모드는 웹앱 미노출(eval 전용).
        _state["graph"] = build_targeted_graph(rt)
    return _state


@asynccontextmanager
async def lifespan(app):
    get_state()                                 # 시작 시 1회 웜업(검색기 로드)
    yield


app = FastAPI(title="법률 질의응답 시스템", lifespan=lifespan)


class AskReq(BaseModel):
    question: str                 # 실서비스는 항상 타겟 편집 — 비교 모드 없음


def _exist(status):
    return True if status == "real" else (False if status == "fake" else None)


def _content(status, content):
    if status != "real":
        return None
    return True if content == "match" else (False if content == "mismatch" else None)


def to_api(final, mode):
    answer = final.get("answer", "")
    gist = section(answer, "답변 요지") or answer.strip()[:200]
    conclusion = section(answer, "결론") or gist
    citations = final.get("citations", [])

    provisions, seen = [], set()
    for c in citations:
        key = (c.get("law_name", ""), c.get("label", ""))
        if key in seen:
            continue
        seen.add(key)
        provisions.append({
            "law": c.get("law_name", ""),
            "label": c.get("label", ""),
            "exist": _exist(c.get("status")),
            "content": _content(c.get("status"), c.get("content")),
            "text": c.get("article_text", "") or "",     # 법제처 조문 원문(모델 생성 아님)
            "note": c.get("explanation", "") or "",
        })
    summ = summarize_citations(citations)

    elog = final.get("edit_log", [])
    edits = {
        "mode": mode,
        "n_edits": len(elog),
        "edit_fraction_mean": round(sum(e["edit_fraction"] for e in elog) / len(elog), 3) if elog else 0.0,
        "lines_edited": sum(e.get("lines_edited", 0) for e in elog),
        "lines_deleted": sum(e.get("lines_deleted", 0) for e in elog),
        "line_edits": [le for e in elog for le in e.get("line_edits", [])],
    }
    from urllib.parse import quote
    doc = final.get("document") or {}
    docx = quote(os.path.basename(doc["docx"])) if doc.get("docx") else None   # 한글 파일명 인코딩
    pdf = quote(os.path.basename(doc["pdf"])) if doc.get("pdf") else None
    return {
        "question": final.get("question"),
        "domain": final.get("domain"),
        "domain_ko": C.DOMAIN_KO.get(final.get("domain"), final.get("domain")),
        "mode": mode,
        "summary": gist,
        "conclusion": conclusion,
        "provisions": provisions,
        "verify_summary": {"n_total": summ["n_total"], "n_real": summ["n_real"],
                           "n_match": summ["n_match"], "n_fake": summ["n_fake"]},
        "retry_count": final.get("retry_count", 0),
        "edits": edits,
        "tool_log": final.get("trace", []),
        "law_api": _state.get("law_api", "real"),
        "docx_url": ("/documents/" + docx) if docx else None,
        "pdf_url": ("/documents/" + pdf) if pdf else None,
    }


@app.post("/api/ask")
def ask(req: AskReq):
    q = (req.question or "").strip()
    if not q:
        raise HTTPException(400, "question 이 비어 있습니다.")
    st = get_state()
    try:
        st["rt"].gen_client.models.list()
    except Exception:
        raise HTTPException(503, "생성기(Qwen3.5-27B :8010) 미실행 — run_web.sh 로 vLLM 기동 필요")
    with _lock:                       # vLLM 직렬화
        final = st["graph"].invoke(initial_state(q, "proposed"), config={"recursion_limit": 60})
    return to_api(final, "targeted")


@app.get("/documents/{name}")
def get_doc(name: str):
    p = DOCS / os.path.basename(name)
    if not p.exists():
        raise HTTPException(404, "문서 없음")
    mime = ("application/pdf" if p.suffix.lower() == ".pdf"
            else "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    return FileResponse(p, filename=p.name, media_type=mime)


@app.get("/config")
def cfg():
    return {
        "generator": C.GEN_NAME,
        "cross_verify": C.LOGIC_NAME,
        "retriever": "nlpai-lab/KoE5",
        "fact_verify": "law.go.kr Open API",
        "repair": "targeted-edit (surgical)",
        "deploy": "on-prem · H200",
        "law_api": _state.get("law_api", "real" if C.oc_is_set() else "mock"),
        "tools": ["koe5_search", "law_go_kr_verify", "deepseek_review",
                  "targeted_edit", "create_legal_document"],
    }


@app.get("/")
def index():
    if not HTML.exists():
        raise HTTPException(404, "index.html 없음")
    return FileResponse(HTML, media_type="text/html")


@app.get("/health")
def health():
    return {"ok": True, "oc": C.oc_is_set(), "warmed": bool(_state)}

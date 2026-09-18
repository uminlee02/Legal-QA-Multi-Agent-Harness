"""
run_demo5.py — 시연 질문 5개 배치 실행 (PHASE2 완료용).

  각 질문: ①검색 → ②생성 → ③검증+자율수정 → ④docx MCP 출력.
  Retriever/검증기/LLM 클라이언트는 1회만 로드, MCP 세션도 1개로 5개 워드 생성.

  실행: python run_demo5.py      (vLLM + LAW_OC + 임베딩 캐시 필요)
  산출: documents/법률답변서_*.docx 5개 + 콘솔 트레이스 + 요약표
"""
import os
import sys
import asyncio

from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import config
from demo_cli import Retriever
from lawcheck import LawVerifier
import agent_pipeline as ap

QUESTIONS = [
    "타인의 명예를 훼손하면 어떤 책임을 지나요?",
    "타인의 물건을 훔치면 어떤 처벌을 받나요?",
    "타인을 기망하여 재물을 편취한 경우 어떤 처벌을 받나요?",
    "주택 임차인이 보증금을 보호받으려면 어떤 권리가 있나요?",
    "교통사고로 타인에게 손해를 입혔을 때 손해배상 책임의 근거는 무엇인가요?",
]


async def main():
    # 생성 = EXAONE(한국어 네이티브)
    client = OpenAI(base_url=config.GEN_BASE_URL, api_key="EMPTY")
    try:
        client.models.list()
    except Exception:
        sys.exit("✗ 생성기(EXAONE) 미실행. vllm serve LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct "
                 "--port 8001 --trust-remote-code")
    if not config.oc_is_set():
        sys.exit("✗ OC 키 미설정: export LAW_OC=...")

    # 교차검증 = Qwen(이종, 있으면 사용)
    cross = OpenAI(base_url=config.CROSS_BASE_URL, api_key="EMPTY")
    try:
        cross.models.list()
        print(f"[교차검증] {config.CROSS_NAME} (이종 모델)")
    except Exception:
        cross = None
        print("[교차검증] 교차모델 미실행 → 단일 모델 자기검증")

    retriever = Retriever()
    verifier = LawVerifier()

    results = []
    for i, q in enumerate(QUESTIONS, 1):
        print(f"\n{'='*72}\n[질문 {i}/{len(QUESTIONS)}] {q}\n{'='*72}")
        results.append(ap.run_agent(q, retriever, client, verifier, cross_client=cross,
                                    gen_model=config.GEN_MODEL, cross_model=config.CROSS_MODEL))

    # MCP 세션 1개로 5개 워드 생성 (생성 에이전트 → 문서작성 MCP 호출)
    print(f"\n{'='*72}\n[④ docx MCP] 5개 문서 생성\n{'='*72}")
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(config.ROOT / "docx_mcp_server.py")],
        env=dict(os.environ))
    docpaths = []
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            for res in results:
                payload = ap.build_payload(res)
                out = str(config.ROOT / "documents" / f"법률답변서_{ap.slug(res['question'])}.docx")
                r = await session.call_tool("create_legal_document", dict(payload, out_path=out))
                txt = next((b.text for b in r.content if getattr(b, "text", None)), None)
                import json as _json
                paths = _json.loads(txt) if txt else {"docx": None, "pdf": None}
                docpaths.append(paths.get("docx"))
                print(f"  ✓ {os.path.basename(paths.get('docx') or '?')}"
                      + (f"  + {os.path.basename(paths['pdf'])}" if paths.get("pdf") else "  (PDF 생략)"))

    # 요약표
    print(f"\n{'='*72}\n[요약] 시연 5개\n{'='*72}")
    print(f"{'질문':<34}{'인용':>4}{'실존':>5}{'내용일치':>7}{'환각':>5}{'자율수정':>7}")
    print("-" * 72)
    tot = dict(n=0, real=0, match=0, content=0, fake=0, fix=0)
    for res in results:
        s = ap.build_payload(res)["summary"]
        q = res["question"][:30]
        print(f"{q:<34}{s['n_total']:>4}{s['n_real']:>5}"
              f"{s['n_match']:>4}/{s['n_content']:<3}{s['n_fake']:>4}{s['iterations']:>7}")
        tot["n"] += s["n_total"]; tot["real"] += s["n_real"]; tot["match"] += s["n_match"]
        tot["content"] += s["n_content"]; tot["fake"] += s["n_fake"]; tot["fix"] += s["iterations"]
    print("-" * 72)
    print(f"{'합계':<34}{tot['n']:>4}{tot['real']:>5}{tot['match']:>4}/{tot['content']:<3}"
          f"{tot['fake']:>4}{tot['fix']:>7}")
    print(f"\n워드 {len([p for p in docpaths if p])}개 생성: documents/")


if __name__ == "__main__":
    asyncio.run(main())

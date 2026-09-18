# 작업 지시서 (codex용) — H200 로컬 "법제처 근거 법률 QA 에이전트" 웹앱

## 목표
기존 파이프라인 부품을 **재사용**해 FastAPI 웹앱을 만든다.
질문 → 검색(KoE5) → 생성(Qwen2.5-7B, RAG) → 법제처 실존·내용 검증 → 자율수정 →
검증 배지가 붙은 답변서(웹 카드 + .docx). **새 모델/측정 없음. 검증기 동결.**

## 환경 (작업 디렉터리: /root/legal-citation-faithfulness/)
- Python: `/opt/conda/bin/python` (3.10). 설치됨: vllm 0.23, fastapi, uvicorn, python-docx 1.2, mcp, torch, transformers, datasets.
- vLLM: `vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 --gpu-memory-utilization 0.35 --seed 42` (OpenAI 호환 :8000). **GPU는 MAD 프로젝트와 공유** → gpu-util 0.35로 공존, 요청 직렬화(lock).
- 법제처 키: 환경변수 `LAW_OC` (값: `lawbot2026injae`). **검증은 반드시 type=JSON** (이 키는 XML이 "미신청" 에러 — lawcheck에 이미 반영됨).
- 임베딩 캐시: `results/statute_emb_fp16.npy` (KoE5 233k, 이미 빌드됨). 없으면 `python build_retrieval.py` 1회 실행.

## 재사용 부품 (import만, 로직 수정 금지)
- `agent_pipeline.py`
  - `run_agent(question, retriever, client, verifier)` → `{question, answer, provs, cites, checks, log}` (자율수정 루프 포함)
  - 헬퍼: `build_payload`, `section(text, name)`, `explain_map(haeseol, cites)`, `title_from_note`, `slug`
- `lawcheck.py` : `LawVerifier.verify(c, answer=, strict=True, gold_by_article=)` → `Verdict`
  - `Verdict.status` = "real"|"fake"|"uncertain", `.content` = "match"|"mismatch"|"unknown"|None,
    `.article_text` = **법제처 조문 원문**, `.official_name`, `.note`. **로직 동결, 건드리지 말 것.**
- `demo_cli.py` : `Retriever()` (KoE5 + 233k 임베딩 캐시 로드, `.search(q, k=3)` → `[{hierarchy, content}]`)
- `docx_mcp_server.py` : `create_legal_document(question, gist, conclusion, citations, summary, out_path)` → .docx 경로 (직접 import 호출 가능; FastMCP stdio 서버이기도 함)

## 만들 것
### 1) `app.py` (FastAPI)
- `POST /api/ask {question}` → `run_agent` 실행 → 아래 JSON. 이어서 docx 생성해 `docx_url` 포함.
- `GET /` → `legal_qa_agent_demo.html` (FileResponse).
- `GET /documents/{name}` → .docx 다운로드(FileResponse, docx mime).
- Retriever/verifier/client는 **서버 시작 시 1회 로드(lifespan)**, vLLM은 **threading.Lock 으로 직렬화**.
- **JSON 계약(엄수):**
  ```json
  { "question": str, "summary": str, "conclusion": str, "self_correct": int,
    "provisions": [ { "law": str|null, "article": str, "title": str,
                      "exist": true|false|null, "content": true|false|null,
                      "text": str, "note": str } ],
    "docx_url": str }
  ```
  - `summary` = `section(answer,"답변 요지")` (없으면 answer 앞부분), `conclusion` = `section(answer,"결론")` 없으면 summary로 폴백.
  - `provisions[]`: `cites`/`checks` zip, (law, article) 기준 중복 제거.
    - `law`=official_name||law_name, `article`=`c.display`, `title`=title_from_note(real일 때),
    - `exist`: real→true, fake→false, uncertain→null. `content`: match→true, mismatch→false, 그 외→null.
    - `text`=`verdict.article_text` (**법제처 원문, 모델 생성 아님**), `note`=`explain_map(...)`의 해당 조문 해설.
  - `self_correct` = `len(res["log"]) - 1`.

### 2) 프론트 `legal_qa_agent_demo.html`
- **사용자가 제공하는 완성 디자인 HTML을 사용. 디자인/CSS/레이아웃/검증 배지 스타일 절대 변경 금지.**
- 그 HTML의 하드코딩 CASES 제거 → 칩(예시 5개) + 자유 입력창 → `fetch('/api/ask')` → 응답 JSON으로 **동일한 답변서 카드** 렌더.
- 파이프라인 애니메이션(검색→생성→검증→문서)은 요청 중 표시, 응답 오면 카드 교체. docx 다운로드 링크(`/documents/...`).

### 3) `run_app.sh`
- vLLM 떠 있는지 확인, 없으면 기동(위 명령) → `uvicorn app:app --host 0.0.0.0 --port 7860`.

## 원칙
- **검증기 동결**: lawcheck 판정 불변(보고 수치 v1 기준).
- **조문 원문은 반드시 법제처(verdict.article_text)**. 모델 생성 아님. 요지/해설/결론만 생성.
- 새 측정/carry-forward 없음. 발표용 시스템. **RAG 항상 ON.**

## 함정 (실제로 겪음)
- `legal_qa_agent_demo.html`이 repo에 **없을 수 있음** → 사용자에게 받아라. 없다고 멋대로 새 디자인 만들지 말 것(데이터 배선만 교체가 목표).
- `pkill -f "vllm..."` 는 그 문자열이 든 자기 셸까지 죽임 → **PID로 종료**.
- Retriever 로드 ~15s(시작/첫 요청). 첫 질의 느림 → lifespan에서 웜업.
- 7B 모델이 가끔 영/중 코드스위칭, `【결론】` 마커 누락 → 결론은 요지로 폴백(서식 레이어에서).
- 음주운전 질문은 검색이 도교법 제148조를 주는데 실제 처벌은 제148조**의2**라 모델이 헷갈림 → 데모 예시에서 제외/주의.

## 완료 기준
- 브라우저에서 질문 입력 → 실제 Qwen 답변 + 법제처 검증 배지 + 조문 원문이 뜨고, .docx 다운로드됨.
- 예시 5개(명예훼손/절도/사기/임대차/교통사고) 정상 동작.

## 참고: 이미 동작하는 레퍼런스 구현이 repo에 있음
`app.py`, `run_app.sh`, 스탠드인 `legal_qa_agent_demo.html` 이 이미 있고 예시 5개 + docx 다운로드까지 검증됨.
**그대로 쓰거나 개선해도 됨.** 핵심 신규 작업은 사용자의 **실제 디자인 HTML**을 위 백엔드 계약(`/api/ask`)에 배선하는 것.
실행: `cd /root/legal-citation-faithfulness && LAW_OC=lawbot2026injae ./run_app.sh` → http://<호스트>:7860

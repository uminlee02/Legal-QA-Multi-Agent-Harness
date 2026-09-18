# 핸드오프 / 정리 — 한국 법률 QA 충실도 측정 + 발표용 에이전트 시스템 (H200)

> 한 줄: **소형 한국어 LLM이 다는 법조문 인용이 진짜인지** 법제처 API로 자동 검증하고(PHASE1 측정, Qwen2.5-7B),
> 그 검증기를 검색·생성·자율수정·문서생성 에이전트 + 웹앱으로 묶고(PHASE2), 이종 교차검증(PHASE3 EXAONE),
> 데모 생성기를 EXAONE로 바꿔 코드스위칭을 없애고 PDF 출력을 더했다(PHASE4). **측정은 Qwen 기준 전 구간 동결.**
> 환경: H200 143GB, /opt/conda py3.10, vllm 0.23, LibreOffice. 법제처 OC키 = `LAW_OC`. 검증기 v1 **동결**.

---

## PHASE 1 — 측정 (검증기 v1 기준 동결, 새 측정 없음)

### 검증기 (lawcheck.py, "심장")
- 인용 추출(정규식, 「」·앞문맥 노이즈 강건) → 법제처 검증.
- 법제처 API(추측 아님, chrisryugj/korean-law-mcp 소스 이식): 검색 `lawSearch.do?target=law`, 조문 `lawService.do?target=eflaw&JO=<조4+가지2>`. **이 OC는 type=XML이 "미신청" → 전부 type=JSON.**
- 판정: **느슨(실존)** ✓real/✗fake/⚠uncertain + **엄격(내용)** = 조문 실존 후 모델 주장(제N조(제목)·문맥)을 **법제처 조문 원문(article_text)**과 대조(match/mismatch/unknown).

### 결과 (Qwen2.5-7B, 가짜 인용율)
| 데이터셋 | 조건 | 느슨(실존) | 엄격(내용) | 비고 |
|---|---|---|---|---|
| 시드 5 | baseline | 7.7% | — | 첫 숫자 |
| KCL n=30 | baseline | **18.8%** (13/69) | 59%* | *내용 mismatch 0 → 분모착시(MCQA는 내용판정불가 84%) |
| KCL n=30 | verify-ON | 11.1% (5/45) | — | n↑면 사후검증이 느슨율↓(단 인용 85→63 회피) |
| **KoBLEX n=226** | **맨몸(OFF)** | **31.2%** (130/416) | 63.6% | 내용환각 **13건**, gold정확인용 **1.8%**(7/398) |
| KoBLEX n=226 | 사후검증(ON) | 25.0% | — | 내용환각 13→**15**(못 고침) |
| **KoBLEX n=226** | **사전검색 RAG** | **5.3%** (10/187) | **6.4%** | 내용환각 **1건**, gold정확인용 **17.3%**(69/398), 엄격커버 54→93% |

**핵심 결론: 사전검색(RAG) ≫ 사후검증.** RAG가 환각을 6배↓·내용환각 13→1·gold인용 10배↑. 사후검증(피드백 재생성)은 실존만 약간 손대고 내용환각은 못 고침.
- 인용 감소 분석(analyze_reduction.py): baseline 환각 13 → 제거 12·잔존 1, **신규 4 생성** → verify-ON 5. **교란: temp=0 런간 비재현** → KoBLEX는 **before/after paired**로 회피.
- 엄격 모드: KCL(터스 MCQA)에선 무력(mismatch 0), **KoBLEX(서술형)에서 실재화**(내용환각 13건 적발).

### 데이터/검색
- 평가셋: KCL `lbox/kcl`(kcl_mcqa), **KoBLEX `JihyungL/KoBLEX-koblex`(226, gold 조문 포함)**.
- RAG 검색기: **KoE5(nlpai-lab/KoE5)** dense, statute 코퍼스 233,544조 임베딩(`results/statute_emb_fp16.npy`, 캐시). gold recall@5 58.3%/hit@5 82.3%.

---

## PHASE 2 — 발표용 에이전트 시스템 (측정 아님, RAG 항상 ON)

질문 → ①검색(KoE5) → ②생성(Qwen RAG) → ③검증+자율수정 루프 → ④docx 생성.
- **③ agent_pipeline.py**: lawcheck.verify로 실존·내용 검증 → ✗ 발견 시 "검색 근거 다시 보라" 피드백 → 재생성 → 재검증(max_iter=2). RAG 접지 수정. `test_selfcorrect.py`로 결정적 증명(민법 제9999조→제750조).
- **④ docx_mcp_server.py**: FastMCP(stdio) `create_legal_document` → 검증배지(✓실존/✓내용일치)+법제처 원문 워드. (조문 원문=법제처, 요지/해설/결론=모델)
- **웹앱 app.py**: `POST /api/ask{question}` → JSON{question,summary,conclusion,self_correct,provisions[{law,article,title,exist,content,text,note}],**tool_log**(MCP 호출 트레이스),docx_url}. `GET /`=HTML, `/documents/{f}`=docx. 드라이런 옵션.
- **프론트 legal_qa_agent_demo.html**: 선배 A2A·MCP 스타일(사이드바 모델구성/툴/실행·초기화/드라이런 + 메인 답변서 + 접히는 JSON MCP로그 + docx). ⚠ **실제 디자인 HTML은 H200에 없어 스탠드인 제작** — 실제 디자인 주면 script(fetch/렌더)만 이식.
- 시연 5개(명예훼손/절도/사기/임대차/교통사고) + 자유질문 정상 동작 확인.

---

## PHASE 3 — 진짜 멀티에이전트: 이종 모델 교차검증 (EXAONE)

생성=Qwen2.5-7B(:8000), **교차검증=EXAONE-3.5-7.8B(:8001, 별도 vLLM)**. 검증기 v1·측정 동결.
- 구조: 질문→검색→생성→ ③-a 법제처 사실검증(기존) + **③-b EXAONE 논리·적용 검증(신규)** → 둘 중 문제 시 Qwen 수정 → 재검증(max_iter=2)→docx.
- `agent_pipeline.exaone_review()`: 질문+답변+근거조문 입력 → `{verdict:"ok|revise", issues:[...]}`. **안전장치: EXAONE은 cites/checks(측정값) 불변** — 사실은 법제처가 최종 심판, EXAONE은 논리 피드백·재생성 트리거만. `run_agent(..., cross_client=, use_rag=)`.
- 웹앱: tool_log에 `M4·cross-verifier/exaone_review` 엔트리, 사이드바에 EXAONE 모델·툴 추가.
- **신규 측정 (run_crossverify.py, KoBLEX n=30, paired)**:

| 설정 | 자기검증(단일·Qwen) 느슨가짜율 | 교차검증(+EXAONE) | EXAONE revise |
|---|---|---|---|
| **맨몸(non-RAG)** | **14.9%**(11/74) | **6.5%**(5/77) | 87%(26/30) |
| RAG | 0.0% | 0.0% (단 논리 revise) | 63%(19/30) |

→ **환각 있는 맨몸 설정에서 이종 교차검증이 단일 자기검증보다 환각 절반↓(14.9→6.5%)** = 가설 검증. RAG에선 사실 환각이 이미 0이라 둘 다 0%이나 EXAONE은 논리 문제(법제처 못잡는)를 63% 잡음. 단서: 개선 일부는 재생성 증가(cross 49 vs self 19), EXAONE 과검출 가능(수정 *질*은 미측정). 사례: cases/analysis_crossverify.md.

---

## PHASE 4 — 데모 답변 품질(코드스위칭 근절) + PDF 출력 (측정/검증기 동결 유지)

### (a) 생성기 EXAONE 전환 = 코드스위칭 근본 해결
- **문제**: Qwen2.5-7B는 중국 모델이라 한국어 답변에 한자(求償·登記·擔保物權者)·중국어 문장을 흘림. temp=0 결정적이라 프롬프트로 못 막음(3가지 프롬프트 변형 모두 실패, 시연 5개 중 4개 깨짐).
- **해결**(사용자 제안): 답변 **생성**을 한국어 네이티브 **EXAONE-3.5-7.8B**가 담당 → 생성문 한자 0(5/5). **역할 swap**:

| 역할 | 모델 | 비고 |
|---|---|---|
| **생성**(요지/해설/결론) | **EXAONE-3.5-7.8B** (:8001, `--trust-remote-code` 필수) | 한국어 네이티브 → 코드스위칭 0 |
| 교차검증(논리·적용) | Qwen2.5-7B (:8000) | 이종 모델 유지 |
| 사실판단(실존·내용) | 법제처 API | **동결** |
| **측정**(가짜인용율) | Qwen2.5-7B | **동결** |

- **측정 동결 보장**: `config.GEN_*/CROSS_*` 역할 변수 추가. `generate(model=)`·`exaone_review(model=)`·`run_agent(gen_model=, cross_model=)` 파라미터화하되 **기본값 = config.MODEL(Qwen)/EXAONE_MODEL** → 측정 스크립트(run_baseline/run_koblex/run_crossverify)는 `qwen`을 생성기로 명시 전달하므로 **숫자 무변**. 데모(app/run_demo5/CLI)만 EXAONE 생성으로 분기.
- 라벨 정정: app.py tool_log(M2 생성=EXAONE·M4 교차검증=Qwen), HTML 사이드바(`generator: exaone-3.5-7.8b / cross-verify: qwen2.5-7b`), docx footer, run_app.sh(EXAONE 필수, `NO_CROSS=1`로 Qwen 끔).
- **검증**: 시연 5개 + 웹앱 /api/ask → 생성문 한자 **0**, **환각 0**, 내용일치 12/12, 충실(요지 2-3·해설 2-3·결론 2-3문장). 남은 한자는 **법제처 조문 원문**(주택임대차보호법·민법 등 실제 구법, 동결 대상이라 그대로 — 모델 생성 아님). 스크립트 test_exaone_gen.py.

### (b) docx-MCP PDF 출력 추가
- `docx_mcp_server._docx_to_pdf()` = **LibreOffice headless**(`soffice --convert-to pdf`, 호출별 `-env:UserInstallation` 임시프로필로 동시실행 충돌 방지, 90s 타임아웃, 실패 시 None+stderr 로그). `create_legal_document`가 JSON `{"docx","pdf"}` 반환, `out_format`="docx|pdf|both"(기본 both).
- app.py: `pdf_url` + `/documents/{f}`가 PDF(`application/pdf`) 서빙, HTML "📄 PDF 내려받기" 버튼, tool_log에 formats 표기. **LibreOffice 7.3 설치 시 함정: apt 목록이 낡아 404 → `apt-get update` 먼저** 후 `libreoffice-writer fonts-nanum`.
- PDF에 NanumGothic/Myeongjo 폰트 임베드 → 한글 정상(boxes 아님). PDF=docx 동일(LibreOffice가 워드 그대로 변환).

## 실행 / 종료
```bash
cd /root/legal-citation-faithfulness
LAW_OC="lawbot2026injae" ./run_app.sh          # ① EXAONE(:8001 생성)+Qwen(:8000 교차)+앱(:7860 듀얼스택)
./cloudflared tunnel --url http://localhost:7860 # ② 브라우저 접속용 공개터널(URL은 출력/로그로; 사용자가 실행)
./stop_app.sh                                    # ③ 전부 종료, GPU 반환
```
- 임베딩 없으면 1회: `python build_retrieval.py`. 오프라인 점검: `python selftest.py`. 라이브 3건: `LAW_OC=.. python live_test.py`.
- 측정 재현: `python run_baseline.py --dataset kcl --n 30`, `python run_koblex.py [--rag results/koblex_retrieval_k5.json]`, `python compare_3way.py`.

## 함정/노하우 (재발 방지)
- 법제처: type=**JSON**만(XML 미신청). 검색 '민법→난민법' 오매칭 → 관련도점수+정확매칭. 법령명 앞노이즈('형사상으로는 형법') → search_law가 앞토큰 제거+트레일링토큰 정확매칭, _loose_match에 endswith.
- **VS Code(code tunnel) 포트포워딩 이 환경선 안 됨** → cloudflared 공개터널로 브라우저 접속(QUIC 기본, 1033은 전파지연 1~2분). 듀얼스택(launch_dual.py)으로 IPv4+IPv6 동시 리슨.
- temp=0 비재현 → paired before/after. `pkill -f <패턴>`이 자기 셸 죽임 → PID/스크립트로(stop_app.sh는 `pkill cloudflared`처럼 -f 없이 또는 PID).
- **Qwen2.5-7B 코드스위칭(한자/중국어)은 프롬프트로 못 막음 → PHASE4에서 생성기를 EXAONE로 바꿔 근본 해결**(데모만; 측정은 Qwen 동결). docx에 남는 한자는 법제처 조문 원문(실제 구법, 정상).
- EXAONE는 vLLM 기동 시 `--trust-remote-code` 없으면 ValidationError. cloudflared 출력은 사용자 터미널로만 가 Claude가 못 읽음 → `> cf.log` 리다이렉트하면 URL 추출 가능(공개터널 실행 자체는 분류기가 Claude에 막음).

## 다음 단계
- carry-forward(법령명 없는 bare 제N조 ⚠ ~170건 줄이기) — 측정 동결 해제 시.
- RAG k/프롬프트 튜닝으로 gold 활용↑(현재 검색상한의 30%만 인용). 엄격 mismatch 임계 튜닝.
- KoBLEX strict 프롬프트(조문 내용 진술 강제)로 엄격 커버리지↑. 더 큰 모델 비교. 실제 디자인 HTML 배선.

## 산출물
- 코드: lawcheck/agent_pipeline/docx_mcp_server/app/build_retrieval/run_koblex/run_baseline/compare_3way/analyze_reduction/demo_cli/questions/config + run_app/stop_app/launch_dual/tcp_proxy + selftest/live_test/test_selfcorrect/**test_exaone_gen(EXAONE 생성기 검증)**.
- 측정: `results/*.json|csv` (kcl_*, koblex_n226, koblex_rag, seed_*, crossverify_n30). 검색: koblex_retrieval_k5.json.
- 정성 사례: `cases/` (3way_rag, citation_reduction, kcl_fabricated, koblex_content_hallucination, q4_verifyON_reciting, analysis_crossverify).
- 문서 샘플: `documents/법률답변서_*.docx` **+ .pdf**(PHASE4, EXAONE 생성). 브리프: README.md, CODEX_BRIEF.md, BRIEF는 본 HANDOFF.
- 제외(재생성): statute_emb_fp16.npy(457MB, build_retrieval.py), cloudflared(38MB, 재다운로드), *.log.
- 런타임 의존: vLLM 0.23, python-docx, mcp SDK, **LibreOffice(soffice, PDF변환)**, fonts-nanum.

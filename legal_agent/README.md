# legal_agent — 정확성-우선 자율 법률 QA (LangGraph 하네스)

한국어 법률 질의에 대해 **큰 모델이 답변을 생성**하고, **인용 법조문을 법제처로 검증**하며,
검증을 통과할 때까지 **재생성**하는 정확성 최우선 멀티에이전트. 최종 답변은 MCP 로 워드/PDF 답변서로 출력.

> 오케스트레이션은 **LangGraph**(차용), 검증 절차는 **IRAC·CoVe·ReAct**(표준 차용, 자작 금지),
> 사실판단 최종권한은 **법제처 Open API**, 생성/논리검증은 **이종(異種)** 모델.
> 검색기(KoE5)·검증기(lawcheck)·문서생성(MCP)은 부모 레포의 **동결(frozen) 모듈을 그대로 재사용**한다.

## 아키텍처 (§3·§4)

```
[질문]
 → route        도메인 판별(형사/민사/기타)      [Qwen3.5-9B 또는 생성기 공유, guided_choice]
 → retrieve     법령 조문 top-k 검색(RAG)         [KoE5 · demo_cli.Retriever 재사용]
 → generate     답변+인용 생성(+IRAC/CoVe 주입)   [Qwen3.5-27B]
 → fact_verify  인용 조문 실존/내용 검증 (게이트)  [법제처 · lawcheck.LawVerifier 재사용] ★최종권한
 → logic_verify 조문 적용 논리 교차검증(보조)      [DeepSeek-R1-Distill-32B, 이종]
 → [gate]  사실검증 실패 & retry<MAX → generate(피드백 주입)   ┐ verify-loop (≤ MAX_RETRY=3)
           통과 → document(docx+pdf) → END                    ┘  [MCP · agent_pipeline.make_docx 재사용]
```

`baseline`(대조군, §8) = `retrieve → generate` 만. 검증·게이트·스킬·이종검증·하네스 전부 제거(순수 생성).

## 모델 (§2, 2026-07 HF 실존 확인 · vllm 0.23 지원 확인)

| 역할 | 모델 | 정밀도 | ~VRAM | 포트 |
|---|---|---|---|---|
| 생성 | `Qwen/Qwen3.5-27B` (dense, `qwen3_5`) | FP16 | 56GB | 8010 |
| 논리검증(이종) | `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` | FP16 | 65GB | 8011 |
| 라우팅 | `Qwen/Qwen3.5-9B` (기본은 생성기 공유) | FP16/FP8 | 19/9GB | (8010)/8012 |
| 사실검증 | 법제처 Open API (모델 아님) | — | — | — |

H200 1장(141GB): 생성+논리검증 두 FP16 서버 합계 ~121GB 가중치 + KV. **VRAM 초과 시 생성 `--quantization fp8`.**

## 실행

```bash
# 0) 임베딩 캐시 확인 (없으면 build_retrieval.py) — 이미 있음: ../results/statute_emb_fp16.npy
# 1) 서빙 (최초 실행 시 모델 ~140GB HF 다운로드; 순차 기동, 경합 방지)
bash serving/vllm_launch.sh
# 2) 법제처 키 (사실검증 게이트 필수)
export LAW_OC=발급받은키          # open.law.go.kr → OPEN API 신청
# 3) 엔드투엔드: 질문 → 검증 답변서(docx/pdf)
python run.py "타인의 명예를 훼손하면 어떤 책임을 지나요?"
# 4) 평가: 베이스라인 vs 제안 (가짜인용율·감소율)
python eval/run_eval.py --n 30 --mode both --out results/lg_eval_n30.json
```

### 서버·키 없이 하네스 로직만 검증 (CI/개발용)
```bash
python smoke_test.py     # 목 LLM/검증기로 라우팅·게이트·재생성루프·MAX_RETRY·문서종료 결정적 증명
```

## 설정 (config_lg.py, 전부 env 오버라이드)
`LG_GEN_MODEL/LG_GEN_BASE_URL`, `LG_LOGIC_MODEL/LG_LOGIC_BASE_URL`, `LG_ROUTER_MODEL/LG_ROUTER_BASE_URL`,
`LG_MAX_RETRY`(=3), `LG_LOGIC_TRIGGERS_RETRY`(=1), `LAW_OC`(부모 config 공유).

## 파일
```
legal_agent/
├── graph.py            # LangGraph 그래프(proposed/baseline) + gate + MAX_RETRY
├── state.py            # LegalState
├── config_lg.py        # 큰 모델 3종 역할/엔드포인트 (부모 config 재사용, 측정 동결 불침범)
├── clients.py          # vLLM(OpenAI) 플러밍: guided_choice, <think> strip
├── runtime.py          # 실서비스 Runtime — 동결 모듈(Retriever/LawVerifier/MCP) 래핑
├── formatting.py       # 파싱/라벨 헬퍼(정규식만, torch 미의존)
├── tracing.py          # 툴 호출 트레이스 엔트리
├── nodes/{router,retriever,generator,fact_verify,logic_verify,document}.py
├── skills/irac-generic.md        # 일반 IRAC/CoVe (형사/민사는 ../skills/*-law-verification.md 재사용)
├── serving/vllm_launch.sh        # H200 서빙(순차 기동, fp8 폴백 주석)
├── eval/run_eval.py              # 베이스라인 vs 제안 (frozen 채점기 재사용)
├── run.py                        # 엔드투엔드
└── smoke_test.py                 # 목 하네스 테스트(서버/키 불필요)
```

## 설계 원칙 준수 (§1)
1. 정확성>속도: 재생성 루프·FP16·다중 검증 허용. 2. 하네스=LangGraph 차용. 3. 스킬=IRAC/CoVe/ReAct 표준만.
4. 이종 교차검증: 생성(Qwen)≠논리검증(DeepSeek) — `assert_heterogeneous()`. 5. 사실 최종판단=법제처(logic 은 덮어쓰기 불가).
6. 게이트 우회 불가: 그래프 구조상 종료 전 `fact_verify` 강제(`fact_verified=True`).

## 측정 동결과의 관계
본 하네스는 부모 레포의 **측정 파이프라인(run_koblex/lawcheck v1)을 건드리지 않는다.** 채점은
frozen `rates/fr/gold_recall` + KoBLEX gold 로 재채점하여 기존 표와 정의를 일치시킨다(그래프 내부
검증은 RAG 접지 gold 사용, 평가 채점은 annotated gold 사용 — 구분 문서화됨).
```

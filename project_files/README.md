# 한국 법률 QA 충실도 측정 — 1차 마일스톤

소형 한국어 LLM이 법률 답변에 다는 **법조문 인용이 진짜인지**를 법제처 Open API로
자동 채점하고, **첫 "가짜 인용율"** 숫자를 뽑는 최소 측정 루프.

> **설계 원칙: 과설계 금지.** 멀티에이전트(LangGraph 등) 없이, 아래 4단계 측정 루프가 전부다.
> 멀티에이전트 래핑은 맨 마지막 일이다.

```
질문 → Qwen 답변(vLLM) → 인용추출(정규식) → 법제처 검증 → 가짜인용율 → CSV/JSON
```

## 0) 사람이 줄 것 — 법제처 OC 키 (필수)
- open.law.go.kr → **OPEN API 신청** → 무료 발급 (이메일 ID 형태).
- `export LAW_OC="발급키"`  (또는 `config.py`의 `OC` 직접 수정)

## 1) 설치
```bash
pip install openai datasets requests        # vllm 은 이미 설치됨(0.23.0)
```

## 2) 모델 서빙 (별도 터미널)
```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000     # H200 141GB → 7B 여유, 14B도 OK
# → http://localhost:8000/v1 (OpenAI 호환)
```

## 3) 오프라인 자체검증 (OC/vLLM 없이)
```bash
python selftest.py     # 인용추출 + JO 6자리 포맷 + 약칭정규화 단위검증
```

## 4) 측정 — 베이스라인(검증 OFF)
```bash
export LAW_OC="발급키"
python run_baseline.py --n 5                  # 시드 법률 질문 5개
python run_baseline.py --dataset kcl --n 20   # KCL(변호사시험)
```
콘솔에 **평균 가짜 인용율**이 찍히고 `results/*.json|csv` 로 저장됨 = 완료 기준.

## 5) 검증 ON (재생성 루프)
가짜로 뜬 인용을 모델에 되먹여 깨끗해질 때까지 재생성:
```bash
python run_baseline.py --n 5 --verify-loop
```
베이스라인 가짜율 vs 검증 ON 가짜율 비교 → 논문의 핵심 그래프.

## 파일
| 파일 | 역할 |
|---|---|
| `lawcheck.py` | **심장 ⭐** 인용추출(「」·앞노이즈 강건) + 법제처 검증(캐시·약칭·JO·느슨/엄격·gold대조) |
| `demo_cli.py` | **통합 데모** 질문→검색(KoE5)→생성(RAG)→인용검증 인터랙티브 CLI |
| `build_retrieval.py` | KoE5로 statute 233k 임베딩 + top-k 검색 인덱스(캐시) |
| `run_koblex.py` | KoBLEX 서술형 페어드(before/after) 측정 + `--rag` |
| `run_baseline.py` | 시드/KCL 4단계 측정 루프 + 느슨/엄격 가짜율 |
| `compare_3way.py` · `analyze_reduction.py` | 맨몸/사후검증/RAG 3-way · 인용감소 분석 |
| `questions.py` | 시드 5개 + KCL 로더 + `load_koblex`(gold 조문) |
| `config.py` · `selftest.py` · `live_test.py` | 설정 / 오프라인 단위검증 / 라이브 3건 |
| `cases/` | 논문용 정성 사례 + 분석 노트 |

## 데모 (검색→생성→검증 통합)
```bash
python build_retrieval.py     # 최초 1회: statute 임베딩 캐시 생성
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000
export LAW_OC="발급키"
python demo_cli.py            # 질문 입력 → 근거검색 + 답변 + 인용 실존/내용 검증
```

## 판정 기준 (lawcheck.py) — 느슨/엄격 두 가지 가짜율
**느슨한 가짜율 (실존만):**
- **✓ real** : 법령·조문(·항) 법제처 DB 실존
- **✗ fake** : 법령 없음 / 조문 없음 / 항 없음 → 가짜 인용
- **⚠ uncertain** : 법령명 미지정·부분매칭·API 실패 → 분모 제외
- `느슨한 가짜율 = ✗ / (전체 − ⚠)`

**엄격한 가짜율 (실존 + 내용 정합성):** `verify(c, answer, strict=True)`
- 조문이 실존해도 **모델이 말한 내용이 그 조문과 맞는지** 법제처 본문과 대조.
  - 모델이 `제N조(○○)`로 단 주장 제목 vs 실제 조문제목 (글자 바이그램 유사도)
  - 또는 인용 주변 문맥이 조문 본문 키워드와 겹치는지
- `내용 mismatch`(제목 명시했으나 실제 제목·본문 어느 쪽과도 안 겹침) → **엄격 가짜**
- `엄격한 가짜율 = (실존✗ + 내용mismatch) / (판정가능 − 내용판정불가)`
- 예: 모델 `도로교통법 제142조(음주운전)` → 실제 제142조="행정소송과의 관계" → 느슨 ✓, **엄격 ✗**.

`results/*.json`에 `loose_fake_rate`/`strict_fake_rate` 모두 저장. 정성 사례는 `cases/`.

## 함정 / 주의
- 법제처 응답은 `cache/`에 캐싱(레이트리밋 회피). 키 교체 시 cache 비우기.
- `제750조` vs `제750조의2`(가지번호), 항/호까지 구분 매칭.
- 외부 호출은 **법제처 공식 API뿐**. LLM은 전부 로컬 H200 (의뢰인 데이터 비유출).
- 검색 MST가 연혁본일 수 있어 `target=eflaw`(현행) 사용. 폐지/연혁 조문은 향후 보완.

## API 출처 (추측 아님)
엔드포인트·JO포맷·판정로직·약칭맵은 `github.com/chrisryugj/korean-law-mcp`
소스(`law-parser.ts`/`verify-citations.ts`/`search-normalizer.ts`)에서 확인해 이식.
- 검색: `GET /DRF/lawSearch.do?OC=&type=JSON&target=law&query=&display=100`
- 조문: `GET /DRF/lawService.do?OC=&type=JSON&target=eflaw&MST=&JO=<조4+가지2>`
- ⚠ **함정**: 이 OC는 `type=XML`이 "미신청" HTML 에러를 반환 → 검색·조문 모두 `type=JSON` 사용.
- 검색이 '민법'→'난민법'처럼 부분매칭을 섞어 반환 → 관련도 점수(`_score`)로 정확매칭 선별.

"""runtime.py — 실서비스 Runtime. 동결된 부모 모듈을 그대로 재사용(자작 금지, §1 원칙 2·3).

  검색   : demo_cli.Retriever (KoE5)          — 재사용
  사실검증: lawcheck.LawVerifier (법제처)       — 재사용(최종 권한)
  문서   : agent_pipeline.make_docx (MCP)      — 재사용
  생성/논리검증/라우팅: config_lg 의 큰 모델 3종(Qwen3.5-27B / DeepSeek-R1-32B / Qwen3.5-9B)

이 모듈만 torch/datasets/mcp 를 끌어온다. graph/nodes 는 rt 인터페이스만 의존 → 목 주입 가능.
"""
import time
import asyncio
import threading
import urllib.error

import config_lg as C
import clients


class ResilientVerifier:
    """법제처 API 일시적 오류(404/타임아웃) 재시도 래퍼(PHASE7 LockedVerifier 이식).

    lawcheck._get 는 429·타임아웃만 재시도하고 404 는 즉시 raise 하는데, 법제처가 부하 시
    간헐적 404 를 뱉어 긴 런을 죽인다. verify 를 그대로 4회 재시도(성공응답은 lawcheck 캐시),
    끝내 실패하면 'uncertain'(분모 제외) 폴백 → 외부 API 장애가 전체 평가를 죽이지 않게.
    lawcheck v1 판정 로직 불변(네트워크 복원력만 추가).
    """
    def __init__(self, v, lock=None):
        self._v = v
        self._l = lock or threading.Lock()

    def verify(self, *a, **kw):
        from lawcheck import Verdict
        last = None
        for attempt in range(4):
            try:
                with self._l:
                    return self._v.verify(*a, **kw)
            except (urllib.error.HTTPError, urllib.error.URLError,
                    TimeoutError, RuntimeError) as e:
                last = e
                time.sleep(0.6 * (attempt + 1))
        return Verdict("uncertain", "⚠", None,
                       f"법제처 API 반복 실패(외부 장애, 판정 제외): {str(last)[:80]}")

    def __getattr__(self, n):
        return getattr(self._v, n)


ROUTER_SYS = (
    "다음 한국 법률 질문의 도메인을 하나만 고르세요. "
    "형사(범죄·형벌·수사·공판 → criminal), 민사(계약·불법행위·재산·가족 → civil), "
    "그 외/행정/혼합(other). 반드시 criminal, civil, other 중 하나만 출력.")

# ③ 풀스택(근거 없는 논리검증): DeepSeek 가 자기 지식으로 논리·적용만 검토(내용은 법제처 몫).
LOGIC_SYS = (
    "당신은 한국 법률 답변을 교차검증하는 또 다른 법률 전문가입니다. [질문], [답변], "
    "[검색된 근거 조문]을 보고 답변의 논리와 조문 적용을 검토하세요. 조문의 실존·내용 정확성은 "
    "법제처가 별도로 최종 판단하니, 당신은 (1) 인용 조문이 이 질문 상황에 적절히 적용됐는지 "
    "(2) 답변 논리에 비약·오류·모순이 있는지만 지적하세요. 먼저 간단히 분석한 뒤, 마지막 줄에 "
    "JSON 한 개만 출력: {\"verdict\": \"ok|revise\", \"issues\": [\"...\"]}. "
    "문제 없으면 verdict=ok, issues=[]. 사소한 트집 금지, 실제 오류만.")

# ④ 풀스택(근거 기반 검증): DeepSeek 가 제공된 조문 원문에만 근거해 답변 인용을 '대조' 검증.
LOGIC_GROUNDED_SYS = (
    "당신은 한국 법률 답변을 교차검증하는 법률 전문가입니다. 아래 [근거 조문 원문]에만 근거하여 "
    "[답변]을 검증하세요. **당신의 사전 지식으로 추론하거나 근거에 없는 조문을 끌어오지 마세요.** "
    "확인 사항: (1) 답변이 인용한 조문이 제공된 근거에 실제로 있는가, (2) 답변이 서술한 그 조문의 "
    "내용이 근거 원문과 일치하는가(불일치·과장·왜곡=오류), (3) 그 조문을 이 질문 사안에 맞게 "
    "적용했는가. 근거 원문과 어긋나거나 근거에 없는 내용을 단언하면 verdict=revise 로 지적하세요. "
    "근거로 확인 불가한 부분은 트집잡지 말고 실제 근거-답변 불일치만 지적하세요. 조문 실존·내용의 "
    "최종 판단은 법제처가 별도로 하니 당신은 근거 대조 결과만 보고하세요. 먼저 근거와 답변을 대조해 "
    "분석한 뒤, 마지막 줄에 JSON 한 개만: {\"verdict\": \"ok|revise\", \"issues\": [\"...\"]}.")


# ⑥ 타겟 편집기(DeepSeek): 판사가 아니라 '틀린 인용만 국소 수정'하는 편집기.
EDITOR_SYS = (
    "당신은 법률 답변의 잘못된 인용만 국소적으로 고치는 편집기입니다. 주어진 [문장]에서 "
    "[법제처가 오류로 판정한 인용]만 수정하세요. **문장 전체를 새로 쓰지 말고, 틀린 인용 부분만 "
    "최소한으로 고치세요.** 규칙: (1) 법제처가 '없는 조문'이라 한 인용은 [검색 근거]의 실존 조문 중 "
    "이 문맥에 맞는 것으로 교체하거나, 마땅한 게 없으면 그 인용을 문장에서 삭제. (2) '내용 불일치'인 "
    "인용은 제공된 [조문 원문]에 맞게 서술을 고치거나, 부적절하면 검색 근거의 실존 조문으로 교체. "
    "(3) 나머지 표현과 다른(정상) 인용은 절대 건드리지 말고 그대로 두세요. 당신의 사전 지식으로 "
    "새 조문을 지어내지 말고 제공된 근거 안에서만 고치세요. 수정된 문장을 딱 한 줄로만 출력하세요"
    "(설명·따옴표·머리말 없이). 인용을 없애 문장 전체가 불필요해지면 정확히 '[삭제]'라고만 출력.")


class Runtime:
    def __init__(self, load_retriever=True, load_verifier=True):
        self.gen_client = clients.make_client(C.GEN_BASE_URL)
        self.logic_client = clients.make_client(C.LOGIC_BASE_URL)
        self.router_client = clients.make_client(C.ROUTER_BASE_URL)
        self._skills = {}
        self.retriever = None
        self.verifier = None
        if load_retriever:
            from demo_cli import Retriever          # heavy(torch/datasets) — 지연 import
            self.retriever = Retriever()
        self.logic_grounded = C.LOGIC_GROUNDED     # 조건 ④: 근거 기반 논리검증
        if load_verifier:
            from lawcheck import LawVerifier
            # 네트워크 복원력 래퍼(404 재시도) — 그래프 fact_check·eval verify_all 모두 rt.verifier 사용.
            self.verifier = ResilientVerifier(LawVerifier())

    # ── 프리플라이트 ────────────────────────────────────────────────────────
    def check_servers(self) -> dict:
        return {
            f"gen({C.GEN_NAME})":    clients.ping(self.gen_client),
            f"logic({C.LOGIC_NAME})": clients.ping(self.logic_client),
            f"router({C.ROUTER_NAME})": clients.ping(self.router_client),
        }

    # ── rt 인터페이스 ──────────────────────────────────────────────────────
    def route(self, question: str) -> str:
        out = clients.chat(self.router_client, C.ROUTER_MODEL, ROUTER_SYS, question,
                           max_tokens=8, guided_choice=list(C.DOMAINS), no_think=True)
        out = (out or "").strip().lower()
        for d in C.DOMAINS:
            if d in out:
                return d
        return "other"

    def retrieve(self, question: str) -> list:
        return self.retriever.search(question, k=C.RAG_TOPK)   # top-k 설정 가능(검색품질 실험)

    def gold_index(self, provs) -> dict:
        from agent_pipeline import gold_index
        return gold_index(provs)

    def skill(self, domain: str) -> str:
        if domain not in self._skills:
            path = C.SKILL_FILES.get(domain)
            try:
                self._skills[domain] = path.read_text(encoding="utf-8") if path and path.exists() else ""
            except Exception:
                self._skills[domain] = ""
        return self._skills[domain]

    def chat_gen(self, system: str, user: str) -> str:
        # Qwen3.5 는 하이브리드 추론 모델 → thinking 끄고 직접 한국어 구조화 답변 생성.
        return clients.chat(self.gen_client, C.GEN_MODEL, system, user,
                            max_tokens=C.GEN_MAX_TOKENS, no_think=True)

    def logic_review(self, question: str, answer: str, provs) -> dict:
        from formatting import article_label
        grounded = self.logic_grounded                 # ④=True 근거대조 / ③=False 논리만
        trunc = 600 if grounded else 300               # ④는 대조 위해 근거 원문 더 길게
        ctx = "\n".join(f"- {article_label(p.get('hierarchy', ''))}: {(p.get('content') or '').strip()[:trunc]}"
                        for p in provs)
        if grounded:
            sys_prompt = LOGIC_GROUNDED_SYS
            user = (f"[질문]\n{question}\n\n[답변]\n{answer}\n\n"
                    f"[근거 조문 원문 — 오직 이 원문에만 근거해 판정]\n{ctx}\n\n"
                    "위 답변이 인용·서술한 조문 내용을 제공된 근거 원문과 직접 대조해 판정하세요. 마지막에 JSON만.")
        else:
            sys_prompt = LOGIC_SYS
            user = (f"[질문]\n{question}\n\n[답변]\n{answer}\n\n[검색된 근거 조문]\n{ctx}\n\n"
                    "위 답변의 논리·조문적용을 교차검증해 마지막에 JSON으로만 판정하세요.")
        # R1 은 <think> 로 추론 → guided_json 대신 자유생성 후 strip_think+parse (추론능력 보존).
        raw = clients.chat(self.logic_client, C.LOGIC_MODEL, sys_prompt, user,
                           max_tokens=C.LOGIC_MAX_TOKENS)
        obj = clients.parse_json_obj(raw)
        issues = obj.get("issues", [])
        if not isinstance(issues, list):
            issues = [str(issues)]
        issues = [str(x).strip() for x in issues if str(x).strip()][:5]
        verdict = obj.get("verdict", "ok")
        if verdict not in ("ok", "revise"):
            verdict = "revise" if issues else "ok"
        return {"verdict": verdict, "issues": issues,
                "grounded": grounded, "evidence_n": len(provs)}

    def edit_line(self, line: str, bads: list, evidence: list) -> str:
        """⑥ 타겟 편집: 한 문장에서 법제처가 오류로 판정한 인용만 최소 수정.
        반환: 수정된 문장 / '' (삭제) / 원본(파싱 실패 시 안전 폴백)."""
        from formatting import article_label
        probs = []
        for c in bads:
            if c.get("status") == "fake":
                probs.append(f"- {c.get('law_name','')} {c.get('label','')}: 법제처 확인 결과 '없는 조문'. "
                             "검색 근거의 실존 조문으로 교체하거나 삭제.")
            else:  # mismatch
                at = (c.get("article_text") or "").strip()[:400]
                probs.append(f"- {c.get('law_name','')} {c.get('label','')}: '내용 불일치'. "
                             f"이 조문의 실제 원문=[{at}]. 원문에 맞게 고치거나 검색 근거의 다른 실존 조문으로 교체.")
        ctx = "\n".join(f"- {article_label(p.get('hierarchy',''))}: {(p.get('content') or '').strip()[:300]}"
                        for p in evidence)
        user = (f"[문장]\n{line}\n\n[이 문장에서 고쳐야 할 인용 (법제처 판정)]\n" + "\n".join(probs)
                + f"\n\n[검색 근거 (실존 조문 후보)]\n{ctx}\n\n"
                "위 문장에서 틀린 인용만 최소로 고쳐 수정된 문장 한 줄로 출력하세요('[삭제]' 가능).")
        raw = clients.chat(self.logic_client, C.LOGIC_MODEL, EDITOR_SYS, user,
                           max_tokens=C.LOGIC_MAX_TOKENS)
        out = clients.strip_think(raw).strip()
        if not out:
            return line                       # 절단/실패 → 원본 보존(안전)
        cand = [l.strip() for l in out.splitlines() if l.strip()]
        if not cand:
            return line
        if any(x.replace(" ", "").startswith("[삭제]") or x.replace(" ", "") == "[삭제]" for x in cand):
            return ""                         # 삭제
        return max(cand, key=len)             # 편집된 문장(가장 긴 비어있지 않은 줄)

    def fact_check(self, answer: str, gold: dict):
        from lawcheck import extract_citations
        cites = extract_citations(answer)
        checks = [self.verifier.verify(c, answer=answer, strict=True, gold_by_article=gold)
                  for c in cites]
        return cites, checks

    def make_document(self, payload: dict, out_path: str) -> dict:
        import os
        from agent_pipeline import make_docx
        # docx 푸터 모델 라벨을 이 하네스 실제 구성으로(MCP 서브프로세스가 env 상속 → 프리즈 서버 무수정 라벨만 교체).
        os.environ["DOC_GEN_LABEL"] = C.GEN_NAME
        os.environ["DOC_CROSS_LABEL"] = C.LOGIC_NAME
        return asyncio.run(make_docx(payload, out_path)) or {"docx": None, "pdf": None}

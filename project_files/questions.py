"""평가셋 — 1차는 직접 쓴 한국 법률 질문 5개로 충분 (브리프 권고).

확장:
  - KCL    : load_questions("kcl", n)  → lbox/kcl (한국 변호사시험 MCQA)
  - KoBLEX : load_koblex(n)            → JihyungL/KoBLEX-koblex (서술형 + gold 조문) ★엄격모드용
"""
import re

SEED = [
    "교통사고로 타인에게 손해를 입혔을 때, 민법상 손해배상 책임의 근거 조문과 요건을 설명해 주세요.",
    "근로자가 부당해고를 당했을 때 구제받을 수 있는 근로기준법상 절차와 근거 조문은 무엇인가요?",
    "주택 임차인이 보증금을 보호받기 위한 대항력과 우선변제권은 어떤 법 몇 조에 규정되어 있나요?",
    "음주운전의 처벌 기준과 근거가 되는 도로교통법 조문을 구체적으로 알려주세요.",
    "회사가 개인정보를 정보주체 동의 없이 제3자에게 제공하면 개인정보 보호법상 어떤 조항 위반이며 처벌은 어떻게 되나요?",
]


def load_questions(dataset: str = "seed", n: int = 5) -> list[str]:
    if dataset == "seed":
        return SEED[:n] if n else SEED

    if dataset == "kcl":
        # KCL kcl_mcqa 스키마: question + A~E(보기) + label(정답) + supporting_precedents
        from datasets import load_dataset
        ds = load_dataset("lbox/kcl", "kcl_mcqa", split="test")
        qs: list[str] = []
        for ex in ds:
            q = (ex.get("question") or "").strip()
            if not q:
                continue
            for opt in ("A", "B", "C", "D", "E"):
                if ex.get(opt):
                    q += f"\n{opt}. {ex[opt].strip()}"
            # 인용을 유도: 정답 선택 + 법조문 근거 설명 요구
            qs.append(q + "\n\n위 문제의 정답을 고르고, 관련 법조문(법령명 제N조)을 근거로 "
                          "구체적으로 설명하세요.")
            if len(qs) >= n:
                break
        return qs

    raise ValueError(f"unknown dataset: {dataset}")


def load_koblex(n: int | None = None) -> list[dict]:
    """KoBLEX(서술형 다단계 법률 QA) — {id, prompt, gold[], gold_answer}.

    gold[] = [{law, jo, branch, hierarchy, content}] (정답 근거 조문 + 본문).
    prompt 에는 RAG 없이 배경+질문만 넣는다(모델이 스스로 조문을 인용하게).
    """
    from datasets import load_dataset
    ds = load_dataset("JihyungL/KoBLEX-koblex", split="test")
    items: list[dict] = []
    for ex in ds:
        prompt = (f"[상황]\n{(ex.get('background') or '').strip()}\n\n"
                  f"[질문] {ex['question'].strip()}\n\n"
                  "위 상황에 적용되는 대한민국 법조문을 '법령명 제N조' 형식으로 "
                  "구체적으로 인용하여 근거와 함께 답하세요.")
        gold = []
        for ctx in ex.get("contexts") or []:
            h = (ctx.get("hierarchy") or "").strip()
            pm = re.match(r"^(.*?)\s*(\d+)\s*조(?:의\s*(\d+))?(.*)$", h)
            if not pm:
                continue
            gold.append({"law": pm.group(1).strip(), "jo": int(pm.group(2)),
                         "branch": int(pm.group(3) or 0), "hierarchy": h,
                         "content": ctx.get("content") or ""})
        items.append({"id": ex.get("id"), "prompt": prompt, "gold": gold,
                      "gold_answer": ex.get("answer") or ""})
        if n and len(items) >= n:
            break
    return items

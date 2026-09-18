"""
test_selfcorrect.py — ③ 자율 수정 루프가 실제로 작동함을 결정적으로 보여주는 데모.

가짜 인용을 내뱉는 모델을 흉내 내, 에이전트가 [검증 → 피드백 → 수정 → 재검증]으로
가짜를 스스로 고치는지 확인한다. 생성(LLM)만 모킹하고, 검증은 실제 법제처 API 사용.
→ vLLM 불필요, LAW_OC 만 있으면 됨.

  LAW_OC=... python test_selfcorrect.py
"""
import types

import config
from lawcheck import LawVerifier
import agent_pipeline as ap


class FakeClient:
    """scripted 답변을 순서대로 반환하는 가짜 OpenAI 클라이언트."""
    def __init__(self, answers):
        self.answers = answers
        self.i = 0
        self.chat = types.SimpleNamespace(completions=self)

    def create(self, **kw):
        a = self.answers[min(self.i, len(self.answers) - 1)]
        self.i += 1
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=a))])


class FakeRetriever:
    """검색을 흉내: 민법 제750조(실존) 근거 1건 반환."""
    def search(self, q, k=3):
        return [{"hierarchy": "민법 750조 불법행위의 내용",
                 "content": "고의 또는 과실로 인한 위법행위로 타인에게 손해를 가한 자는 "
                            "그 손해를 배상할 책임이 있다."}]


def main():
    if not config.oc_is_set():
        raise SystemExit("LAW_OC 필요 (검증은 실제 법제처 API 사용)")

    # round0: 지어낸 조문(민법 제9999조)  →  round1: 근거의 실존 조문(민법 제750조)
    fake = FakeClient([
        "불법행위로 인한 손해배상은 민법 제9999조(불법행위)에 따라 가해자가 책임을 집니다.",
        "불법행위로 인한 손해배상은 민법 제750조(불법행위의 내용)에 따라 가해자가 책임을 집니다.",
    ])
    res = ap.run_agent("타인의 불법행위로 손해를 입었을 때 배상받을 근거는?",
                       FakeRetriever(), fake, LawVerifier())

    print("\n=== 자율 수정 루프 로그 (before/after) ===")
    for e in res["log"]:
        mark = "✗ 문제" if e["bad"] else "✓ 통과"
        print(f"  iter{e['iter']}: {mark} | 인용={e['citations']} | 적발={e['bad']}")

    assert res["log"][0]["bad"], "round0 에 가짜(민법 제9999조)가 적발돼야 함"
    assert not res["log"][-1]["bad"], "최종 답변은 깨끗해야 함"
    print("\n✓ 검증 통과: 에이전트가 가짜(민법 제9999조)를 적발 → 근거 재참조 → "
          "실존(민법 제750조)으로 자율 수정 후 통과.")


if __name__ == "__main__":
    main()

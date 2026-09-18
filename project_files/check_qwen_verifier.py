"""
check_qwen_verifier.py — Qwen 교차(논리)검증기 품질 점검 (측정 아님, 결과 파일 불변).
  생성=EXAONE(한국어), 검증=Qwen. 각 질문에 Qwen 검증기의 verdict/issues 원문을 덤프 →
  사람이 (1) 한국어 제대로 썼나 (2) 지적이 타당한가 (3) 오독 흔적 평가.
  round-0만(생성 1 + 검증 1) — 검증기 1차 판단 품질만 본다.
"""
import re
from openai import OpenAI
import config
from demo_cli import Retriever, article_label
import agent_pipeline as ap

QUESTIONS = [
    "타인의 명예를 훼손하면 어떤 책임을 지나요?",
    "타인의 물건을 훔치면 어떤 처벌을 받나요?",
    "타인을 기망하여 재물을 편취한 경우 어떤 처벌을 받나요?",
    "주택 임차인이 보증금을 보호받으려면 어떤 권리가 있나요?",
    "교통사고로 타인에게 손해를 입혔을 때 손해배상 책임의 근거는 무엇인가요?",
    "상속인의 순위는 어떻게 되나요?",
    "이혼 시 재산분할은 어떻게 이루어지나요?",
    "근로자가 부당하게 해고당했을 때 어떻게 구제받나요?",
    "임대인이 보증금을 돌려주지 않으면 어떻게 하나요?",
    "유언장은 어떤 방식으로 작성해야 효력이 있나요?",
]
HAN = re.compile(r'[一-鿿]{2,}')   # 연속 2자 이상 한자(중국어/한자 혼입)


def main():
    gen = OpenAI(base_url=config.GEN_BASE_URL, api_key="EMPTY")      # EXAONE
    cross = OpenAI(base_url=config.CROSS_BASE_URL, api_key="EMPTY")  # Qwen
    gen.models.list(); cross.models.list()
    retriever = Retriever()

    for i, q in enumerate(QUESTIONS, 1):
        provs = retriever.search(q)
        answer = ap.generate(gen, ap.rag_prompt(q, provs), model=config.GEN_MODEL)
        review = ap.exaone_review(cross, q, answer, provs, model=config.CROSS_MODEL)  # Qwen 검증
        gist = ap.section(answer, "답변 요지") or answer[:140]

        print("\n" + "═" * 80)
        print(f"[{i}] {q}")
        print("═" * 80)
        print(f"  검색: {', '.join(article_label(p['hierarchy']) for p in provs)}")
        print(f"  [EXAONE 답변 요지] {gist[:170]}")
        print(f"  [Qwen 검증기] verdict = {review['verdict']}")
        if review["issues"]:
            for j, iss in enumerate(review["issues"], 1):
                cjk = HAN.findall(iss)
                flag = f"  ⟨한자혼입:{' '.join(cjk[:3])}⟩" if cjk else ""
                print(f"     issue{j}: {iss}{flag}")
        else:
            print("     issues: (없음)")


if __name__ == "__main__":
    main()

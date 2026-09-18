"""
check_qwen_verifier_flawed.py — Qwen 검증기 변별력 테스트 (측정 아님).
  명백한 법리 오류를 '심은' 한국어 답변을 Qwen 검증기에 넣어,
  (1) 잡아내는가(revise) (2) issue가 제대로 된 한국어인가 (3) 지적이 타당한가 평가.
  생성기 없이, 손으로 만든 결함 답변 + 실제 검색 조문으로 Qwen exaone_review만 호출.
"""
import re
from openai import OpenAI
import config
from demo_cli import Retriever, article_label
import agent_pipeline as ap

# (질문, 결함답변, 심은오류 설명) — 명백·일의적 오류만
CASES = [
    ("타인의 물건을 훔치면 어떤 처벌을 받나요?",
     "【답변 요지】 타인의 물건을 훔치는 절도는 형법 제347조(사기죄)에 해당하여 처벌됩니다. 절도는 사기의 한 종류이기 때문입니다.\n【결론】 따라서 절도범은 사기죄로 처벌됩니다.",
     "절도(형법 제329조)를 사기죄(347조)라 하고 '절도는 사기의 일종'이라는 허위 법리"),
    ("교통사고로 사람을 다치게 하면 어떻게 처벌되나요?",
     "【답변 요지】 교통사고로 사람을 다치게 하면 형법 제335조(준강도)가 적용되어 처벌됩니다.\n【결론】 교통사고 가해자는 준강도죄로 처벌됩니다.",
     "준강도(335조)를 교통사고에 적용하는 틀린 맥락"),
    ("명예훼손을 하면 어떤 책임을 지나요?",
     "【답변 요지】 명예훼손은 형사처벌 대상이 전혀 아닙니다. 다만 형법 제307조에 따라 2년 이하의 징역 또는 500만원 이하의 벌금에 처합니다.\n【결론】 형사책임은 없습니다.",
     "'형사처벌 대상 아님'과 '징역형에 처함'이 정면 모순"),
    ("주택 임차인이 보증금을 보호받으려면 어떻게 하나요?",
     "【답변 요지】 임차인은 전입신고나 확정일자 같은 아무 요건 없이도 당연히 최우선변제권을 가집니다. 어떤 절차도 필요 없습니다.\n【결론】 요건 불문하고 보증금을 우선 변제받습니다.",
     "대항요건·확정일자 없이 우선변제권이 당연 발생한다는 오류"),
    ("타인을 기망하여 재물을 편취하면 어떤 처벌을 받나요?",
     "【답변 요지】 사기죄는 형법 제347조에 따라 최대 100년 이하의 징역에 처합니다.\n【결론】 사기범은 최대 100년 징역입니다.",
     "사기죄 법정형을 '100년 이하'로 과장(실제 10년 이하)"),
    ("상속인의 순위는 어떻게 되나요?",
     "【답변 요지】 상속 1순위는 피상속인의 형제자매이고, 직계비속(자녀)은 4순위입니다.\n【결론】 형제자매가 자녀보다 먼저 상속합니다.",
     "상속순위 뒤바꿈(직계비속이 1순위, 형제자매 3순위)"),
]


HAN = re.compile(r'[一-鿿]{2,}')


def main():
    cross = OpenAI(base_url=config.CROSS_BASE_URL, api_key="EMPTY")  # Qwen
    cross.models.list()
    retriever = Retriever()
    caught = 0
    for i, (q, ans, planted) in enumerate(CASES, 1):
        provs = retriever.search(q)
        review = ap.exaone_review(cross, q, ans, provs, model=config.CROSS_MODEL)  # Qwen 검증
        hit = review["verdict"] == "revise"
        caught += hit
        print("\n" + "═" * 80)
        print(f"[{i}] {q}")
        print(f"  심은 오류: {planted}")
        print(f"  결함답변 : {ans.replace(chr(10),' / ')[:150]}")
        print(f"  검색조문 : {', '.join(article_label(p['hierarchy']) for p in provs)}")
        print(f"  ▶ Qwen verdict = {review['verdict']}  ({'✓ 잡음' if hit else '✗ 놓침(rubber-stamp)'})")
        for j, iss in enumerate(review["issues"], 1):
            cjk = HAN.findall(iss)
            flag = f"  ⟨한자혼입:{' '.join(cjk[:3])}⟩" if cjk else ""
            print(f"     issue{j}: {iss}{flag}")
        if not review["issues"]:
            print("     issues: (없음)")
    print("\n" + "═" * 80)
    print(f"요약: Qwen 검증기가 명백한 오류 {len(CASES)}건 중 {caught}건 적발 (revise)")


if __name__ == "__main__":
    main()

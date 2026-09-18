"""
test_exaone_gen.py — EXAONE(:8001)를 '생성기'로 써서 5개 시연 질문 답변을 만들고
  ① 코드스위칭(한자 연속 2자 이상) ② 형식(요지/해설/결론 파싱) ③ 환각(법제처 검증)을 측정.
  검증기·측정 동결: lawcheck/Retriever 그대로 사용. 생성 모델만 EXAONE로 바꿔 비교.
"""
import re
from openai import OpenAI
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
HAN = re.compile(r'[一-鿿]{2,}')   # 연속 2자 이상 한자 = 중국어/한자 혼입


def gen_exaone(client, prompt):
    r = client.chat.completions.create(
        model=config.EXAONE_MODEL,
        messages=[{"role": "system", "content": ap.SYSTEM},
                  {"role": "user", "content": prompt}],
        temperature=0.0, max_tokens=1500, seed=42)
    return r.choices[0].message.content or ""


def sents(t):
    return len([s for s in re.split(r'(?<=[다요\.])\s+', (t or "").strip()) if len(s) > 3])


def main():
    client = OpenAI(base_url=config.EXAONE_BASE_URL, api_key="EMPTY")
    client.models.list()
    retriever = Retriever()
    verifier = LawVerifier()
    print(f"\n{'질문':<30}{'한자혼입':>7}{'요지':>5}{'해설':>5}{'결론':>5}{'인용':>5}{'실존':>5}{'환각':>5}")
    print("-" * 72)
    tot = dict(cjk=0, n=0, real=0, fake=0)
    for q in QUESTIONS:
        provs = retriever.search(q)
        gold = ap.gold_index(provs)
        ans = gen_exaone(client, ap.rag_prompt(q, provs))
        cites, checks = ap.verify_answer(verifier, ans, gold)
        n_fake = sum(v.status == "fake" for v in checks)
        n_real = sum(v.status == "real" for v in checks)
        cjk = HAN.findall(ans)
        gist = ap.section(ans, "답변 요지"); hae = ap.section(ans, "조문 해설"); con = ap.section(ans, "결론")
        n_hae = len([l for l in hae.splitlines() if re.search(r'제\s*\d+\s*조', l)])
        tag = "✓없음" if not cjk else f"✗{len(cjk)}곳"
        print(f"{q[:28]:<30}{tag:>7}{sents(gist):>5}{n_hae:>5}{sents(con):>5}"
              f"{len(cites):>5}{n_real:>5}{n_fake:>5}")
        tot["cjk"] += len(cjk); tot["n"] += len(cites); tot["real"] += n_real; tot["fake"] += n_fake
        # 첫 질문은 본문 샘플 출력
        if q == QUESTIONS[2]:   # 기망(Qwen이 깨졌던 질문)
            print(f"    [기망 요지] {gist[:150]}")
            print(f"    [기망 결론] {con[:150]}")
    print("-" * 72)
    print(f"{'합계':<30}{('✓0' if not tot['cjk'] else '✗'+str(tot['cjk'])):>7}"
          f"{'':>15}{tot['n']:>5}{tot['real']:>5}{tot['fake']:>5}")


if __name__ == "__main__":
    main()

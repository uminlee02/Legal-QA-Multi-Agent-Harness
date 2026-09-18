"""
demo_cli.py — 한국 법률 QA 충실도 데모 (검색→생성→인용검증 통합 파이프라인).

  [1] 검색  : KoE5 dense로 법제처 statute 코퍼스에서 관련 조문 top-k
  [2] 생성  : Qwen2.5-7B(vLLM) + 검색 조문 컨텍스트 주입(RAG)
  [3] 검증  : 각 인용을 법제처 API로 실존 + 내용일치 확인

  실행 전제:
    export LAW_OC="발급키"
    vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000
    (검색 임베딩 캐시 results/statute_emb_fp16.npy 필요 → 없으면 build_retrieval.py 먼저)

  실행: python demo_cli.py     (질문 입력, 빈 줄/Ctrl-D 로 종료)
"""
import re
import sys
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from openai import OpenAI

import config
from build_retrieval import hier_key, EMB_PATH, MODEL as KOE5_MODEL, DEV
from lawcheck import extract_citations, LawVerifier

K = 3


@torch.no_grad()
def encode_query(model, tok, text):
    enc = tok(["query: " + text], padding=True, truncation=True,
              max_length=512, return_tensors="pt").to(DEV)
    o = model(**enc)
    mask = enc.attention_mask.unsqueeze(-1).float()
    emb = (o.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return F.normalize(emb, dim=1)[0]


def article_label(hier: str) -> str:
    m = re.match(r"^(.*?)\s*(\d+)\s*조(?:의\s*(\d+))?", hier or "")
    if not m:
        return (hier or "")[:24]
    return f"{m.group(1).strip()} 제{m.group(2)}조" + (f"의{m.group(3)}" if m.group(3) else "")


def title_from_note(note: str) -> str:
    m = re.search(r"\(([^)]*)\)\s*$", note or "")
    return m.group(1) if m else ""


class Retriever:
    def __init__(self):
        print("· statute 코퍼스/임베딩 로드...", flush=True)
        corp = load_dataset("JihyungL/KoBLEX-statute", split="corpus")
        self.hiers = list(corp["hierarchy"])
        self.contents = list(corp["content"])
        self.P = torch.from_numpy(np.load(EMB_PATH)).to(DEV).float()
        print("· KoE5 검색기 로드...", flush=True)
        self.tok = AutoTokenizer.from_pretrained(KOE5_MODEL)
        self.model = AutoModel.from_pretrained(KOE5_MODEL).to(DEV).half().eval()

    def search(self, query, k=K):
        q = encode_query(self.model, self.tok, query)
        idx = torch.topk(q @ self.P.T, k).indices.cpu().tolist()
        return [{"hierarchy": self.hiers[j], "content": self.contents[j]} for j in idx]


def generate(client, prompt):
    r = client.chat.completions.create(
        model=config.MODEL,
        messages=[{"role": "system", "content":
                   "당신은 대한민국 법률 전문가입니다. 반드시 한국어로만 답하세요. "
                   "주어진 근거 조문 중 관련된 것만 '법령명 제N조(조문제목)' 형식으로 정확히 "
                   "인용하고, 없는 조문은 지어내지 마세요."},
                  {"role": "user", "content": prompt}],
        temperature=0.0, max_tokens=700, seed=42)
    return r.choices[0].message.content or ""


def answer(query, retriever, client, verifier):
    print("\n[1] 검색 중... (법제처 KoE5)", flush=True)
    provs = retriever.search(query)
    labels = []
    for p in provs:                       # 같은 조문(항만 다름) 표시 중복 제거
        lab = article_label(p["hierarchy"])
        if lab not in labels:
            labels.append(lab)
    print(f"    근거 조문 {len(labels)}건: " + ", ".join(labels))

    print("\n[2] 답변 생성 중... (Qwen2.5-7B + 근거 주입)", flush=True)
    # '제N조' 표준형으로 제시 → 모델이 표준 형식으로 인용하도록 유도
    ctx = "\n".join(f"- {article_label(p['hierarchy'])}: {(p['content'] or '').strip()[:380]}"
                    for p in provs)
    prompt = ("[참고 조문] 아래 검색된 조문 중 관련된 것만 근거로 인용하라. 여기 없는 조문을 지어내지 마라.\n"
              + ctx + f"\n\n[질문] {query}")
    ans = generate(client, prompt)

    # 검증: 검색된 조문을 내용대조 gold 로도 사용
    gold = {}
    for p in provs:
        hk = hier_key(p["hierarchy"])
        if hk:
            gold[hk] = (p["content"] or "") + " " + p["hierarchy"]
    cites = extract_citations(ans)
    checks = [verifier.verify(c, answer=ans, strict=True, gold_by_article=gold) for c in cites]

    bar = "─" * 38
    print("\n" + bar)
    print("답변: " + ans.strip())
    if cites:
        print("근거:")
        n_real = n_match = n_content = 0
        for c, v in zip(cites, checks):
            ex = {"real": "✓실존", "fake": "✗미존재"}.get(v.status, "⚠확인필요")
            ct = ""
            if v.status == "real":
                n_real += 1
                if v.content in ("match", "mismatch"):
                    n_content += 1
                ct = {"match": "✓내용일치", "mismatch": "✗내용불일치",
                      "unknown": "·내용미확인"}.get(v.content, "")
                if v.content == "match":
                    n_match += 1
            name = v.official_name or c.law_name or "(미지정)"
            ttl = title_from_note(v.note)
            label = f"{name} {c.display}" + (f"({ttl})" if ttl else "")
            print(f"      {label:<32} {ex} {ct}")
        print(bar)
        n = len(cites)
        n_fake = sum(x.is_fake for x in checks)
        ok = "✓" if n_real == n else ""
        cok = "✓" if (n_content and n_match == n_content) else ""
        print(f"인용 검증: {n_real}/{n} 법제처 확인 {ok} | 내용 일치 {n_match}/{n_content} {cok} | 환각 {n_fake}")
    else:
        print("근거: (추출된 법조문 인용 없음)")
        print(bar)
    print()


def main():
    client = OpenAI(base_url=config.VLLM_BASE_URL, api_key="EMPTY")
    try:
        client.models.list()
    except Exception:
        sys.exit("✗ vLLM 미실행. 먼저: vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000")
    if not config.oc_is_set():
        sys.exit("✗ OC 키 미설정: export LAW_OC=발급키")

    retriever = Retriever()
    verifier = LawVerifier()
    print("\n준비 완료. 법률 질문을 입력하세요 (빈 줄/Ctrl-D 종료).\n")
    while True:
        try:
            q = input("질문> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            break
        answer(q, retriever, client, verifier)


if __name__ == "__main__":
    main()

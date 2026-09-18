"""
build_retrieval.py — 사전검색(RAG)용 인덱스 빌드. KoE5(dense)로 KoBLEX statute
코퍼스(233,544조) 전체를 임베딩하고, 각 질문에 top-k 조문을 검색해 저장.

  - gold를 직접 주지 않음(부정행위 방지): 전체 코퍼스에서 질문으로 검색해 gold를 찾아야 함.
  - 출력: results/koblex_retrieval_k{K}.json  ({id: [{hierarchy, content}]})
  - 부가: retrieval recall@k (gold 조문이 top-k에 포함되는 비율) 출력 — 리트리버 품질 진단.

KoE5는 E5 계열 → query/passage 프리픽스 + mean pooling + L2정규화 (sentence-transformers
설치는 vllm 의존성 충돌 위험이라 transformers로 직접 구동).
"""
import json
import re
import time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

import config
from questions import load_koblex

MODEL = "nlpai-lab/KoE5"
K = 5
BS = 128
EMB_PATH = config.RESULTS_DIR / "statute_emb_fp16.npy"
META_PATH = config.RESULTS_DIR / "statute_meta.json"
DEV = "cuda"


def hier_key(h: str):
    m = re.match(r"^(.*?)\s*(\d+)\s*조(?:의\s*(\d+))?", h or "")
    if not m:
        return None
    return (re.sub(r"\s+", "", m.group(1)), int(m.group(2)), int(m.group(3) or 0))


def mean_pool(hidden, mask):
    m = mask.unsqueeze(-1).float()
    return (hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)


@torch.no_grad()
def encode(model, tok, texts, prefix):
    out = []
    for i in range(0, len(texts), BS):
        batch = [prefix + (t or "") for t in texts[i:i + BS]]
        enc = tok(batch, padding=True, truncation=True, max_length=512,
                  return_tensors="pt").to(DEV)
        o = model(**enc)
        emb = F.normalize(mean_pool(o.last_hidden_state, enc.attention_mask), dim=1)
        out.append(emb.half().cpu())
        if i % (BS * 50) == 0:
            print(f"  ...{i}/{len(texts)}", flush=True)
    return torch.cat(out)


def main():
    t0 = time.time()
    print("== 코퍼스 로드 ==", flush=True)
    corp = load_dataset("JihyungL/KoBLEX-statute", split="corpus")
    hiers = list(corp["hierarchy"])
    contents = list(corp["content"])
    passages = [f"{h} {c}" for h, c in zip(hiers, contents)]
    print(f"  {len(passages)} 조문", flush=True)

    print("== KoE5 로드 ==", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(DEV).half().eval()

    if EMB_PATH.exists():
        print("== 캐시된 임베딩 로드 ==", flush=True)
        P = torch.from_numpy(np.load(EMB_PATH))
    else:
        print("== passage 임베딩 (시간 소요) ==", flush=True)
        P = encode(model, tok, passages, "passage: ")
        np.save(EMB_PATH, P.numpy())
        META_PATH.write_text(json.dumps(hiers, ensure_ascii=False), encoding="utf-8")
        print(f"  저장 {EMB_PATH.name} {tuple(P.shape)}  ({time.time()-t0:.0f}s)", flush=True)

    print("== 질문 임베딩 + 검색 ==", flush=True)
    items = load_koblex()
    queries = [it["prompt"].replace("위 상황에 적용되는", " ") for it in items]
    Q = encode(model, tok, queries, "query: ")

    Pg = P.to(DEV).float()
    retr = {}
    recall_hits = recall_tot = 0
    rr_at = {1: 0, 3: 0, 5: 0}
    for i, it in enumerate(items):
        sims = (Q[i].to(DEV).float() @ Pg.T)
        topk = torch.topk(sims, K).indices.cpu().tolist()
        prov = [{"hierarchy": hiers[j], "content": contents[j]} for j in topk]
        retr[it["id"]] = prov
        # recall@k vs gold
        goldset = {(re.sub(r"\s+", "", g["law"]), g["jo"], g["branch"]) for g in it["gold"]}
        retr_keys = [hier_key(p["hierarchy"]) for p in prov]
        for g in goldset:
            recall_tot += 1
            if g in retr_keys:
                recall_hits += 1
        for kk in rr_at:
            hit = any(hier_key(prov[j]["hierarchy"]) in goldset for j in range(min(kk, len(prov))))
            rr_at[kk] += int(bool(goldset) and hit)

    out = config.RESULTS_DIR / f"koblex_retrieval_k{K}.json"
    out.write_text(json.dumps(retr, ensure_ascii=False), encoding="utf-8")
    n = len(items)
    print("\n== 검색 품질 ==")
    print(f"  gold 조문 recall@{K} (조문 단위) = {recall_hits}/{recall_tot} = {recall_hits/recall_tot:.1%}")
    for kk in (1, 3, 5):
        print(f"  질문 단위 hit@{kk} (gold 중 하나라도 top-{kk}) = {rr_at[kk]}/{n} = {rr_at[kk]/n:.1%}")
    print(f"  저장: {out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()

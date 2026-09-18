"""(2) retrieve — KoE5 dense 로 법령 코퍼스에서 관련 조문 top-k. [부모 Retriever 재사용]

검색된 '조문 원문'을 생성 근거(evidence)로, 그리고 내용대조 gold 로 전달.
gold 는 검증(내용일치) 용도이며 모델에는 노출되지 않는다.
retrieve 는 baseline·proposed 공통(브리프 §8: RAG 는 두 조건 공통).
"""
from formatting import article_label
from tracing import entry, push


def retrieve(state, rt):
    q = state["question"]
    provs = rt.retrieve(q)
    gold = rt.gold_index(provs)
    labels = []
    for p in provs:
        lab = article_label(p.get("hierarchy", ""))
        if lab not in labels:
            labels.append(lab)
    return {"evidence": provs, "gold": gold,
            "trace": push(state, entry("M1·retriever", "koe5_search", "ok",
                                       f"근거 {len(labels)}건: " + ", ".join(labels[:8]),
                                       phase="retrieve", k=len(provs)))}

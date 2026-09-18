"""run.py — 엔드투엔드 실행: 질문 1건 → 검증 답변서(docx/pdf).

  python run.py "타인의 명예를 훼손하면 어떤 책임을 지나요?"

전제(없으면 안내 후 종료):
  · vLLM 생성/논리검증 서버 기동 (serving/vllm_launch.sh)
  · LAW_OC 환경변수 (법제처 사실검증 게이트, §1 원칙 6)
  · KoE5 임베딩 캐시 results/statute_emb_fp16.npy (없으면 build_retrieval.py)
"""
import sys
import pathlib

_HERE = pathlib.Path(__file__).resolve().parent
for p in (str(_HERE), str(_HERE.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

import config_lg as C
from graph import build_proposed_graph, initial_state


def main():
    question = " ".join(sys.argv[1:]).strip() or "타인의 명예를 훼손하면 어떤 책임을 지나요?"

    # ── 이종 교차검증 불변식(§1 원칙 4) ──
    gf, lf = C.assert_heterogeneous()
    print(f"· 이종 확인: 생성={gf} ≠ 논리검증={lf}")

    # ── 사실검증 게이트 전제(§1 원칙 6) ──
    if not C.oc_is_set():
        sys.exit("✗ LAW_OC 미설정 — 법제처 사실검증 게이트 필수. `export LAW_OC=발급키` 후 재실행.")

    from runtime import Runtime
    print("· Runtime 로드(KoE5 검색기 + 법제처 검증기)...")
    rt = Runtime()
    servers = rt.check_servers()
    for name, ok in servers.items():
        print(f"  {'✓' if ok else '✗'} {name}")
    if not servers.get(f"gen({C.GEN_NAME})"):
        sys.exit("✗ 생성 서버 미실행 — serving/vllm_launch.sh 로 기동 후 재실행.")

    app = build_proposed_graph(rt)
    print(f"\n[질문] {question}\n")
    final = app.invoke(initial_state(question, mode="proposed"),
                       config={"recursion_limit": 50})

    # ── 결과 요약 ──
    cites = final.get("citations", [])
    n_real = sum(c["status"] == "real" for c in cites)
    n_fake = sum(c["status"] == "fake" for c in cites)
    n_match = sum(c["status"] == "real" and c.get("content") == "match" for c in cites)
    print("\n" + "─" * 48)
    print(f"도메인: {C.DOMAIN_KO.get(final.get('domain'), final.get('domain'))} "
          f"| 재생성 {final.get('retry_count', 0)}회 | 사실검증 게이트 실행: {final.get('fact_verified')}")
    print(f"인용 {len(cites)} | 실존 {n_real} | 내용일치 {n_match} | 환각(미존재) {n_fake}")
    for c in cites:
        badge = {"real": "✓실존", "fake": "✗미존재"}.get(c["status"], "⚠확인필요")
        ct = {"match": " ✓내용일치", "mismatch": " ✗내용불일치"}.get(c.get("content"), "")
        print(f"  - {c['label']}: {badge}{ct}")
    doc = final.get("document") or {}
    print(f"\n✓ 워드: {doc.get('docx')}")
    print(f"✓ PDF : {doc.get('pdf') or '(변환 실패/생략)'}")


if __name__ == "__main__":
    main()

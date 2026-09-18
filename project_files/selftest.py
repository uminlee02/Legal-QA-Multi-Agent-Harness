"""selftest.py — OC/vLLM 없이 인용추출·JO포맷·약칭정규화 단위검증.

  python selftest.py        # 오프라인. 통과하면 핵심 로직 정상.
"""
from lawcheck import extract_citations, resolve_alias

SAMPLE = (
    "이 사안은 민법 제750조에 따라 불법행위 손해배상 책임이 성립합니다. "
    "또한 상법 제401조의2 제2항 제3호의 이사 책임도 문제됩니다. "
    "화관법 제28조 및 근로기준법 제23조 제1항도 참고하세요. "
    "한편 「개인정보 보호법」 제26조와 「도로교통법」 제148조의2처럼 "  # 모델이 쓰는 「」 형식
    "법령명을 홑낫표로 감싼 인용도 잡혀야 합니다. "
    "그리고 존재하지 않는 가공의조문법 제99999조도 추출되는지 확인합니다."
)


def main():
    cites = extract_citations(SAMPLE)
    print("추출된 인용:")
    for c in cites:
        print(f"  - {c.law_name or '(미지정)':<12} {c.display:<22} JO={c.jo_code}")

    by_jo = {c.jo_code: c for c in cites}

    # JO 6자리 포맷 (buildJO = 조4 + 가지2)
    assert "075000" in by_jo, "민법 제750조 → 075000"
    assert "040102" in by_jo, "상법 제401조의2 → 040102"
    assert by_jo["040102"].hang == 2 and by_jo["040102"].ho == 3, "항/호 파싱"
    assert by_jo["040102"].law_name == "상법", "법령명 역추적"

    # 약칭 정규화
    assert resolve_alias("화관법") == "화학물질관리법"
    assert resolve_alias("근기법") == "근로기준법"
    assert resolve_alias("민법") == "민법"  # 약칭 아님 → 그대로

    # 법령명 lookback (화관법 인용에 법령명이 붙는지)
    assert any(c.law_name == "화관법" for c in cites), "화관법 lookback"

    # 「」 홑낫표 형식 — 법령명이 괄호 밖 노이즈 없이 추출돼야 함
    assert any(c.law_name == "개인정보 보호법" for c in cites), "「개인정보 보호법」 추출"
    assert any(c.law_name == "도로교통법" and c.jo == 148 and c.jo_branch == 2
               for c in cites), "「도로교통법」 제148조의2 추출"

    print("\n✓ selftest 통과 — 추출/JO포맷/약칭 정상. (검증 end-to-end는 OC 키 필요)")


if __name__ == "__main__":
    main()

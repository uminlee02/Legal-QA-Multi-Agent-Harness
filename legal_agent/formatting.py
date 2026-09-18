"""formatting.py — 답변 파싱/라벨링 헬퍼(정규식만, torch/datasets 미의존).

부모 레포(demo_cli / agent_pipeline)에서 검증된 로직을 그대로 옮겨 담아,
LangGraph 노드가 무거운 검색·모델 모듈 없이도 동작하도록(=목 테스트 가능) 분리한다.
"""
import re


def article_label(hier: str) -> str:
    """계층문자열 → '법령명 제N조[의M]' 표준 라벨."""
    m = re.match(r"^(.*?)\s*(\d+)\s*조(?:의\s*(\d+))?", hier or "")
    if not m:
        return (hier or "")[:24]
    return f"{m.group(1).strip()} 제{m.group(2)}조" + (f"의{m.group(3)}" if m.group(3) else "")


def title_from_note(note: str) -> str:
    """검증 note 끝의 '(조문제목)' 추출."""
    m = re.search(r"\(([^)]*)\)\s*$", note or "")
    return m.group(1) if m else ""


def section(text: str, name: str) -> str:
    """【name】 섹션 본문 추출(다음 【 또는 끝까지)."""
    m = re.search(rf"【\s*{re.escape(name)}\s*】", text or "")
    if not m:
        return ""
    start = m.end()
    nxt = re.search(r"【", text[start:])
    return (text[start:start + nxt.start()] if nxt else text[start:]).strip()


def explain_map(haeseol: str, cites) -> dict:
    """【조문 해설】 각 줄 → (조,가지) → 해설텍스트."""
    out = {}
    for line in (haeseol or "").splitlines():
        line = line.strip()
        m = re.search(r"제\s*(\d+)\s*조(?:의\s*(\d+))?", line)
        if not m:
            continue
        key = (int(m.group(1)), int(m.group(2) or 0))
        parts = re.split(r"[:：]", line, maxsplit=1)
        expl = parts[1].strip() if len(parts) == 2 else re.sub(r"^[-•·\s]+", "", line)
        out.setdefault(key, expl)
    return out


def slug(q: str) -> str:
    s = re.sub(r"[^0-9A-Za-z가-힣]+", "_", q or "").strip("_")
    return s[:30] or "query"


def build_citation_payload(answer: str, cites, checks) -> list:
    """(cites, checks) → docx MCP citations 리스트. 법제처 원문(article_text)·모델 해설 결합.

    cites 원소는 lawcheck.Citation(.law_name,.display,.jo,.jo_branch),
    checks 원소는 lawcheck.Verdict(.status,.content,.official_name,.note,.article_text) 형태.
    """
    expl = explain_map(section(answer, "조문 해설"), cites)
    citations, seen = [], set()
    for c, v in zip(cites, checks):
        ttl = title_from_note(getattr(v, "note", "") or "")
        label = f"{(getattr(v, 'official_name', None) or getattr(c, 'law_name', None) or '(미지정)')} {c.display}"
        if ttl and v.status == "real":
            label += f"({ttl})"
        if label in seen:
            continue
        seen.add(label)
        citations.append({
            "label": label,
            "status": v.status,
            "content": getattr(v, "content", None),
            "article_text": getattr(v, "article_text", "") or "",     # 법제처 원문(코드 주입, 모델 생성 아님)
            "explanation": expl.get((getattr(c, "jo", None), getattr(c, "jo_branch", 0)), ""),
            "jo": getattr(c, "jo", None),                              # 타겟 편집(⑥) 줄 매칭용
            "jo_branch": getattr(c, "jo_branch", 0),
            "law_name": getattr(v, "official_name", None) or getattr(c, "law_name", None) or "",
        })
    return citations


def summarize_citations(citations: list) -> dict:
    return {
        "n_total": len(citations),
        "n_real":  sum(x["status"] == "real" for x in citations),
        "n_match": sum(x["status"] == "real" and x["content"] == "match" for x in citations),
        "n_content": sum(x["status"] == "real" and x["content"] in ("match", "mismatch")
                         for x in citations),
        "n_fake":  sum(x["status"] == "fake" for x in citations),
    }

"""(⑥) targeted_edit — 검증 실패 시 '전체 재생성' 대신 '틀린 인용 span만 국소 편집'.

③의 whack-a-mole(재생성이 멀쩡한 인용을 깸)을 피한다. 답변을 줄 단위로 보고, 법제처가
fake/mismatch로 판정한 인용이 들어간 줄만 DeepSeek 편집기로 최소 수정하고, 나머지 줄은
바이트 단위로 보존한다 → 정상 인용 보존 보장. 수정된 인용만 다음 fact_verify 에서 재확인.
"""
import re
import difflib

import config_lg as C
from tracing import entry, push

_LINE_JO = re.compile(r"제\s*(\d+)\s*조(?:의\s*(\d+))?")


def _bad_set(citations):
    return {(c.get("jo"), c.get("jo_branch") or 0)
            for c in citations if c.get("status") == "fake" or c.get("content") == "mismatch"}


def targeted_edit(state, rt):
    answer = state.get("answer", "")
    citations = state.get("citations", [])
    evidence = state.get("evidence", [])
    bad = [c for c in citations if c.get("status") == "fake" or c.get("content") == "mismatch"]
    retry = state.get("retry_count", 0) + 1

    if not bad:
        return {"retry_count": retry,
                "trace": push(state, entry("M4·editor", "targeted_edit", "ok",
                                           "편집 대상 없음", phase="edit"))}

    lines = answer.split("\n")
    new_lines, n_edited, n_deleted, line_edits = [], 0, 0, []
    for line in lines:
        jos = {(int(m.group(1)), int(m.group(2) or 0)) for m in _LINE_JO.finditer(line)}
        line_bads = [c for c in bad if (c.get("jo"), c.get("jo_branch") or 0) in jos]
        if line_bads and line.strip():
            edited = rt.edit_line(line, line_bads, evidence)      # DeepSeek 국소 편집
            if edited == "":                                       # 삭제
                n_deleted += 1
                line_edits.append({"before": line, "after": "", "op": "delete"})
                continue
            if edited != line:
                n_edited += 1
                line_edits.append({"before": line, "after": edited, "op": "edit"})
            new_lines.append(edited)
        else:
            new_lines.append(line)                                 # 정상 줄 → 바이트 보존
    new_answer = "\n".join(new_lines)

    # 편집 국소성 지표: 문자 단위 유사도(1=무변화). 전체 재생성이면 낮고, 국소 편집이면 높음.
    sim = difflib.SequenceMatcher(None, answer, new_answer).ratio()
    stats = {"iter": retry, "n_bad": len(bad), "lines_edited": n_edited,
             "lines_deleted": n_deleted, "edit_fraction": round(1 - sim, 3),
             "line_edits": line_edits}      # 웹 시각화용 before/after(변경 줄만, 소량)
    return {
        "answer": new_answer,
        "retry_count": retry,
        "edit_log": list(state.get("edit_log", [])) + [stats],
        "trace": push(state, entry("M4·editor", "targeted_edit", "retry",
                                    f"편집#{retry}: 대상 {len(bad)} | 수정 {n_edited}줄 삭제 {n_deleted}줄 "
                                    f"| 편집률 {stats['edit_fraction']}(0=무변화)", phase="edit")),
    }

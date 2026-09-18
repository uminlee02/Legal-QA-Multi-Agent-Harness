"""tracing.py — 선배 그림4 스타일 툴 호출 트레이스 엔트리(웹앱/디버그 공용)."""
import time
from datetime import datetime, timezone


def entry(agent: str, tool: str, status: str, result: str = "",
          phase: str = "", **arguments) -> dict:
    now = time.time()
    return {
        "ts": now,
        "ts_iso": datetime.fromtimestamp(now, timezone.utc).isoformat(),
        "agent": agent,
        "tool": tool,
        "phase": phase,
        "status": status,               # ok | fail | retry | skip
        "arguments": arguments or {},
        "result_excerpt": (result or "")[:400],
    }


def push(state, e: dict) -> list:
    """기존 trace 에 엔트리 추가한 새 리스트 반환(LangGraph 덮어쓰기 채널용)."""
    return list(state.get("trace", [])) + [e]

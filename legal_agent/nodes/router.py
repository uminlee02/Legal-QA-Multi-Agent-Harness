"""(1) route — 도메인 판별(형사/민사/기타). [Qwen3.5-9B 또는 생성기 공유]

제약 디코딩(guided_choice)으로 criminal/civil/other 중 하나만 강제 → 형식오류 0.
baseline 모드에서는 라우팅(하네스)을 끄고 'other'(스킬 미주입)로 둔다.
"""
import config_lg as C
from tracing import entry, push


def route(state, rt):
    q = state["question"]
    if state.get("config_mode") == "baseline":
        return {"domain": "other",
                "trace": push(state, entry("M0·router", "route_domain", "skip",
                                           "baseline: 라우팅/스킬 비활성", phase="route"))}
    try:
        dom = rt.route(q)
    except Exception as e:
        dom = "other"
        return {"domain": dom,
                "trace": push(state, entry("M0·router", "route_domain", "fail",
                                           f"라우팅 실패 → other ({e})", phase="route"))}
    if dom not in C.DOMAINS:
        dom = "other"
    return {"domain": dom,
            "trace": push(state, entry("M0·router", "route_domain", "ok",
                                       f"{dom} ({C.DOMAIN_KO.get(dom, dom)})", phase="route"))}

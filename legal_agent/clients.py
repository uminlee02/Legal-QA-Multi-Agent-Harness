"""clients.py — vLLM(OpenAI 호환) 저수준 플러밍.

역할별 큰 모델(생성/논리검증/라우팅)에 대한 얇은 래퍼.
  · guided_choice / guided_json (vLLM structured outputs) 로 라우팅·판정을 제약 디코딩
    → 제어채널 코드스위칭·형식오류 원천 차단(2번 실험에서 검증된 방어).
  · DeepSeek-R1 계열은 <think>...</think> 추론 블록을 뱉으므로 파싱 전 제거.
"""
import re
import json
from typing import Optional

from openai import OpenAI

import config_lg as C

_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def make_client(base_url: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key="EMPTY")


def ping(client: OpenAI) -> bool:
    try:
        client.models.list()
        return True
    except Exception:
        return False


def strip_think(text: str) -> str:
    """R1 계열의 <think> 추론 블록 제거(판정/JSON 파싱 전)."""
    if not text:
        return ""
    text = _THINK_RE.sub("", text)
    # 닫히지 않은 <think> (max_tokens 절단) 방어: </think> 뒤만 취함.
    if "</think>" in text:
        text = text.split("</think>")[-1]
    elif "<think>" in text:
        text = text.split("<think>")[-1]
    return text.strip()


def chat(client: OpenAI, model: str, system: str, user: str,
         max_tokens: int = 1024, temperature: Optional[float] = None,
         guided_choice: Optional[list] = None,
         guided_json: Optional[dict] = None,
         no_think: bool = False) -> str:
    """단발 chat. guided_* 는 vLLM extra_body 로 전달(제약 디코딩).

    no_think=True: Qwen3.5 등 하이브리드 추론 모델의 thinking 모드 비활성
    (chat_template_kwargs.enable_thinking=False). 생성/라우팅용 — 직접 한국어 답변, 토큰 절약.
    """
    extra_body = {}
    if guided_choice is not None:
        extra_body["guided_choice"] = guided_choice
    if guided_json is not None:
        extra_body["guided_json"] = guided_json
    if no_think:
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}
    kwargs = dict(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=C.TEMPERATURE if temperature is None else temperature,
        max_tokens=max_tokens,
        seed=C.SEED,
    )
    if extra_body:
        kwargs["extra_body"] = extra_body
    r = client.chat.completions.create(**kwargs)
    return r.choices[0].message.content or ""


def parse_json_obj(text: str) -> dict:
    """모델 출력에서 첫 JSON 오브젝트를 관대하게 추출."""
    text = strip_think(text)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}

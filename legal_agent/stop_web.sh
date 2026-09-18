#!/usr/bin/env bash
# stop_web.sh — 웹앱/터널/vLLM 전부 종료 → GPU 반환(MAD 공유). PID로 종료(자기셸 매칭 회피).
cd "$(dirname "$0")"
echo "· cloudflared 종료"; for pid in $(pgrep -f "[c]loudflared tunnel"); do kill "$pid" 2>/dev/null; done
echo "· uvicorn 종료";     for pid in $(pgrep -f "[u]vicorn web.app"); do kill "$pid" 2>/dev/null; done
echo "· vLLM 종료";        for pid in $(pgrep -f "[v]llm serve"); do kill "$pid" 2>/dev/null; done
sleep 6
# 고아 EngineCore 정리 (PPID=1 누수)
for pid in $(ps -eo pid,cmd | grep -iE "vllm|EngineCore" | grep -v grep | awk '{print $1}'); do kill -9 "$pid" 2>/dev/null; done
sleep 3
echo "=== GPU ==="; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
echo "GPU 반환 완료."

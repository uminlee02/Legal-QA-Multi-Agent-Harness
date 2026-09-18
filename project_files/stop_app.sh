#!/usr/bin/env bash
# stop_app.sh — 법률 QA 웹앱 + 터널 전부 종료 (GPU 반환)
pkill cloudflared 2>/dev/null && echo "✓ cloudflared(터널) 종료" || echo "· cloudflared 없음"
pkill -f launch_dual 2>/dev/null
pkill -f "uvicorn app:app" 2>/dev/null && echo "✓ 웹앱 종료" || echo "· 웹앱 없음"
pkill -f "vllm serve" 2>/dev/null && echo "✓ vLLM 종료" || echo "· vLLM 없음"
sleep 2
echo "남은 GPU 사용량:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv 2>/dev/null | tail -1

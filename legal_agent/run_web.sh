#!/usr/bin/env bash
# run_web.sh — LangGraph 법률 QA 웹앱: vLLM(생성+검증) + uvicorn(:7860) + cloudflared 공개 터널.
#   기존 검증된 파이프라인(⑥ 타겟편집)을 웹으로 노출만 함. 새 로직 없음.
#   사용: LAW_OC=발급키 bash run_web.sh   (LAW_OC 없으면 mock 폴백, 화면에 표기)
set -uo pipefail
cd "$(dirname "$0")"                      # legal_agent/
LOG=web/logs; mkdir -p "$LOG"

if [ -z "${LAW_OC:-}" ]; then
  echo "⚠ LAW_OC 미설정 → 법제처 mock 폴백으로 실행(인용 '확인필요' 표기). 실제 검증하려면 export LAW_OC=..."
fi

# 1) vLLM 두 서버 (gen Qwen3.5-27B :8010 + logic DeepSeek-R1-32B :8011) — GDN/seqs/thinking 픽스 내장
if curl -sf http://localhost:8010/v1/models >/dev/null 2>&1 && curl -sf http://localhost:8011/v1/models >/dev/null 2>&1; then
  echo "· vLLM 이미 기동됨 (:8010, :8011)"
else
  echo "· vLLM 기동 (serving/vllm_launch.sh, 최초 로드 수 분)..."
  bash serving/vllm_launch.sh
fi

# 2) uvicorn 웹서버 (:7860)
echo "· 웹서버(uvicorn :7860) 기동..."
setsid env LAW_OC="${LAW_OC:-}" python -m uvicorn web.app:app --host 0.0.0.0 --port 7860 \
  > "$LOG/web.log" 2>&1 < /dev/null &
for i in $(seq 1 90); do curl -sf http://localhost:7860/health >/dev/null 2>&1 && { echo "  ✓ 웹서버 준비됨"; break; }; sleep 2; done

# 3) cloudflared 공개 터널
echo "· cloudflared 공개 터널 기동(전파 1~2분)..."
setsid ../cloudflared tunnel --url http://localhost:7860 > "$LOG/cf.log" 2>&1 < /dev/null &   # cloudflared 는 부모 레포에
URL=""
for i in $(seq 1 40); do
  URL=$(grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" "$LOG/cf.log" | head -1)
  [ -n "$URL" ] && break; sleep 3
done

echo
echo "=================================================================="
echo "  공개 URL : ${URL:-'(추출 실패 — web/logs/cf.log 확인, 1~2분 후 재시도)'}"
echo "  로컬     : http://localhost:7860"
echo "  생성=Qwen3.5-27B(:8010)  검증=DeepSeek-R1-32B(:8011)  법제처=$([ -n "${LAW_OC:-}" ] && echo real || echo mock)"
echo "  종료     : bash stop_web.sh   (터널·vLLM 내리고 GPU 반환)"
echo "=================================================================="
echo "  ※ trycloudflare URL 은 공개 노출 — 데모용, 끝나면 stop_web.sh 로 종료할 것."

#!/usr/bin/env bash
# run_app.sh — 법률 QA 멀티에이전트 웹앱 실행 (EXAONE:8001 생성 + Qwen:8000 교차검증 + uvicorn)
#   사용: ./run_app.sh             (브라우저: http://<호스트>:7860)
#   환경: LAW_OC(법제처 키), PORT(기본 7860), NO_CROSS=1 (교차검증 Qwen 끄고 단일 모델)
#   ※ 생성 = EXAONE(한국어 네이티브 → 코드스위칭 없음, 필수). 교차검증 = Qwen(이종, 선택).
set -e
cd "$(dirname "$0")"
export LAW_OC="${LAW_OC:-lawbot2026injae}"
PORT="${PORT:-7860}"

# 생성기 EXAONE (:8001) — 필수
if ! curl -s -m 2 http://localhost:8001/v1/models 2>/dev/null | grep -q EXAONE; then
  echo "[vLLM] EXAONE-3.5-7.8B 기동 (생성, 한국어 네이티브)..."
  nohup /opt/conda/bin/vllm serve LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct --port 8001 \
        --gpu-memory-utilization 0.35 --max-model-len 8192 --seed 42 --trust-remote-code > exaone.log 2>&1 &
fi
# 교차검증 Qwen (:8000) — 선택(NO_CROSS=1 로 생략)
if [ -z "${NO_CROSS}" ] && ! curl -s -m 2 http://localhost:8000/v1/models 2>/dev/null | grep -q Qwen; then
  echo "[vLLM] Qwen2.5-7B 기동 (이종 교차검증)..."
  nohup /opt/conda/bin/vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 \
        --gpu-memory-utilization 0.35 --seed 42 > vllm.log 2>&1 &
fi
for i in $(seq 1 60); do
  curl -s -m 2 http://localhost:8001/v1/models 2>/dev/null | grep -q EXAONE && break
  sleep 3
done
curl -s -m 2 http://localhost:8001/v1/models 2>/dev/null | grep -q EXAONE \
  && echo "[vLLM] EXAONE READY (:8001) — 생성" \
  || { echo "[vLLM] ✗ EXAONE(:8001) 미준비 — 생성기 필수! exaone.log 확인"; }
[ -z "${NO_CROSS}" ] && {
  for i in $(seq 1 60); do
    curl -s -m 2 http://localhost:8000/v1/models 2>/dev/null | grep -q Qwen && break
    sleep 3
  done
  curl -s -m 2 http://localhost:8000/v1/models 2>/dev/null | grep -q Qwen \
    && echo "[vLLM] Qwen READY (:8000) — 교차검증 ON" || echo "[vLLM] Qwen 미준비 — 교차검증 없이 동작"
}
echo "[app]  http://localhost:${PORT}  (IPv4+IPv6 듀얼스택 · VS Code 포트포워딩 호환 · 첫 질의 ~15s 웜업)"
# launch_dual.py: IPV6_V6ONLY=0 듀얼스택 소켓 → uvicorn --fd (VS Code는 localhost를 ::1로 풀어서 IPv6 필요)
exec env PORT="${PORT}" /opt/conda/bin/python launch_dual.py

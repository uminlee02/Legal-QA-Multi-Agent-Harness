#!/usr/bin/env bash
# vllm_launch.sh — H200 1장(141GB)에 브리프 §2 큰 모델 서빙(OpenAI 호환).
#
#   생성   Qwen/Qwen3.5-27B                          :8010  (~56GB FP16)
#   논리검증 deepseek-ai/DeepSeek-R1-Distill-Qwen-32B  :8011  (~65GB FP16)   ← 이종
#   라우팅  (기본) 생성 서버 공유 :8010                        (VRAM 절약; 9B 별도는 아래 주석)
#
# VRAM 예산: 가중치 56+65 = 121GB. 남는 ~20GB 를 두 서버 KV/오버헤드로 분배.
#   → gpu-memory-utilization 0.44 + 0.50 = 0.94(≈132GB). max-model-len 은 KV 절약 위해 16384.
# 함정(2번 실험): 두 vLLM 동시 기동 시 메모리 프로파일 경합 → **순차 기동**(아래 헬스체크 대기).
# VRAM 초과/32768 필요 시: 생성 모델에 `--quantization fp8` 추가(가중치 ~28GB로 반감).
set -euo pipefail
cd "$(dirname "$0")/.."                       # legal_agent/
LOG_DIR="${LOG_DIR:-serving/logs}"; mkdir -p "$LOG_DIR"

GEN_MODEL="${LG_GEN_MODEL:-Qwen/Qwen3.5-27B}"
LOGIC_MODEL="${LG_LOGIC_MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-32B}"
GEN_PORT="${GEN_PORT:-8010}"
LOGIC_PORT="${LOGIC_PORT:-8011}"
MAXLEN="${MAXLEN:-16384}"
GEN_UTIL="${GEN_UTIL:-0.44}"
LOGIC_UTIL="${LOGIC_UTIL:-0.50}"

wait_ready () {  # $1=port  $2=name
  echo "· $2 (:$1) 헬스체크 대기..."
  for i in $(seq 1 180); do
    if curl -sf "http://localhost:$1/v1/models" >/dev/null 2>&1; then
      echo "  ✓ $2 준비됨"; return 0
    fi
    sleep 5
  done
  echo "  ✗ $2 타임아웃 — $LOG_DIR/$2.log 확인"; return 1
}

echo "[1/2] 생성기 $GEN_MODEL 기동 (:$GEN_PORT, util=$GEN_UTIL)"
# ⚠ Qwen3.5 는 Gated Delta-Net 하이브리드 어텐션 사용. H200(SM90)에서 'auto'는 flashinfer GDN
#   커널을 택하는데, 번들된 flashinfer 0.6.12 가 현재 CUDA 툴킷에서 sm90 GDN 커널 JIT 컴파일에
#   실패(cuda::ptx has no member 'fence_proxy_tensormap_generic')하여 무한 재시도로 기동 불가.
#   → --additional-config 로 GDN prefill 백엔드를 Triton/FLA(내장 폴백)로 강제해 우회.
# ⚠ GDN(선형/Mamba) 레이어는 디코드 시퀀스마다 Mamba 캐시 블록 1개 필요. util=0.44 에선 ~108개만
#   → 기본 max_num_seqs=1024 면 CUDA graph capture 불가(ValueError). 108 이하로 낮춘다(우리 워크로드는
#   순차/저동시성이라 64 충분). VRAM 더 주면 블록 수도 늘어 이 상한이 올라감.
setsid vllm serve "$GEN_MODEL" \
  --dtype bfloat16 --gpu-memory-utilization "$GEN_UTIL" \
  --max-model-len "$MAXLEN" --port "$GEN_PORT" \
  --enable-prefix-caching \
  --additional-config '{"gdn_prefill_backend":"triton"}' \
  --max-num-seqs "${GEN_MAX_SEQS:-64}" \
  >"$LOG_DIR/gen.log" 2>&1 < /dev/null &
wait_ready "$GEN_PORT" "gen"

echo "[2/2] 논리검증 $LOGIC_MODEL 기동 (:$LOGIC_PORT, util=$LOGIC_UTIL)  — 순차(경합 방지)"
setsid vllm serve "$LOGIC_MODEL" \
  --dtype bfloat16 --gpu-memory-utilization "$LOGIC_UTIL" \
  --max-model-len "$MAXLEN" --port "$LOGIC_PORT" \
  >"$LOG_DIR/logic.log" 2>&1 < /dev/null &
wait_ready "$LOGIC_PORT" "logic"

echo
echo "완료. 엔드포인트:"
echo "  생성   http://localhost:$GEN_PORT/v1   ($GEN_MODEL)"
echo "  논리검증 http://localhost:$LOGIC_PORT/v1 ($LOGIC_MODEL)"
echo "  라우팅  (기본) 생성 서버 공유 → LG_ROUTER_BASE_URL=http://localhost:$GEN_PORT/v1"
echo
echo "종료: pkill -f 'vllm serve'  (고아 EngineCore 누수 시 ps 로 개별 kill)"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader || true

# ── (선택) 라우팅 전용 Qwen3.5-9B 별도 서빙 ─ VRAM 여유(fp8 사용 등) 있을 때만 ──
#   생성/논리검증을 fp8 로 낮춰 자리를 확보한 뒤:
# setsid vllm serve Qwen/Qwen3.5-9B --dtype bfloat16 --quantization fp8 \
#   --gpu-memory-utilization 0.10 --max-model-len 8192 --port 8012 \
#   >"$LOG_DIR/router.log" 2>&1 < /dev/null &
# wait_ready 8012 "router"
#   그리고: export LG_ROUTER_BASE_URL=http://localhost:8012/v1 LG_ROUTER_MODEL=Qwen/Qwen3.5-9B

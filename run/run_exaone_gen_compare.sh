#!/usr/bin/env bash
# 보조실험: 생성기를 EXAONE로 바꿔 KoBLEX n=226 측정 (맨몸 + RAG).
#   기존 Qwen 결과(koblex_n226/koblex_rag) 불변 — 새 파일(--out)로만 저장.
#   생성기만 EXAONE로 교체(env MODEL/VLLM_BASE_URL), 측정프롬프트·lawcheck v1·페어드 동일.
cd "$(dirname "$0")"
export LAW_OC="${LAW_OC:-lawbot2026injae}"
export MODEL="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
export VLLM_BASE_URL="http://localhost:8001/v1"

echo "############ [1/2] EXAONE 맨몸(non-RAG) n=226 $(date +%H:%M:%S) ############"
/opt/conda/bin/python run_koblex.py --out koblex_exaone_gen
echo "############ [2/2] EXAONE RAG n=226 $(date +%H:%M:%S) ############"
/opt/conda/bin/python run_koblex.py --rag results/koblex_retrieval_k5.json --out koblex_exaone_gen_rag
echo "############ 완료 $(date +%H:%M:%S) ############"

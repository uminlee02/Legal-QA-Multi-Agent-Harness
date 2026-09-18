#!/usr/bin/env bash
# run_phaseA.sh — 본실험 페이즈 A(주결과): 생성=Qwen × {A,B,C} × {RAG,맨몸} × 3런, n=226.
#   이미 있는 출력(main_qwen_*_r*.json)은 skip → 크래시 후 재실행하면 이어서 감.
set -e
cd "$(dirname "$0")"
export LAW_OC="${LAW_OC:-lawbot2026injae}"
for r in 1 2 3; do
  for cond in rag norag; do
    /opt/conda/bin/python run_main.py --gen qwen --cond "$cond" --run "$r"
  done
done
/opt/conda/bin/python analyze_main.py
echo "PHASE_A_DONE"
